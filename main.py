import torch
from torch.utils.data import DataLoader
from data_prep.dataset import SkeletonDataset, SkeletonDatasetFromDirectory
from models.proposed.st_gcn import Model as RtStgcn
from models.proposed.st_gcn import AggregateStgcn, ObservedAggregateStgcn, QAggregateStgcn
from models.original.st_gcn import Model as OriginalStgcn
from processor import Processor

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, broadcast_object_list, broadcast

import st_gcn_parser
import argparse
import os
import random
import time
import json


def common(rank: int, world_size: int, args):
    """Performs setup common to any ST-GCN model variant.

    Only needs to be invoked once for a given problem (train-test, benchmark, etc.). 
    Corresponds to the parts of the pipeline irrespective of the black-box model used.
    Creates DataLoaders, sets up processing device and random number generator,
    reads action classes file.

    TODO: use IP/port from SLURM to initialize DDP process group
    TODO: use pin_memory to load data straight into corresponding GPU

    Args:
        args : ``dict``
            Parsed CLI arguments.

    Returns:
        Dictionary of action classes.

        Train and validation DataLoaders.
    """

    # preparing datasets for training and validation
    if args.dataset_type == 'file':
        train_data = SkeletonDataset('{0}/train_data.npy'.format(args.data), '{0}/train_label.pkl'.format(args.data), args.actions)
        val_data = SkeletonDataset('{0}/val_data.npy'.format(args.data), '{0}/val_label.pkl'.format(args.data), args.actions)
    elif args.dataset_type == 'dir':
        train_data = SkeletonDatasetFromDirectory('{0}/train/features'.format(args.data), '{0}/train/labels'.format(args.data), args.actions)
        val_data = SkeletonDatasetFromDirectory('{0}/val/features'.format(args.data), '{0}/val/labels'.format(args.data), args.actions)

    # trials of different length can not be placed in the same tensor when batching, have to manually iterate over them
    batch_size = 1 if args.dataset_type == 'dir' else args.batch_size

    # if using CUDA, initialize process group for DDP using environment variables from the SLURM job script
    if torch.cuda.is_available():
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    # prepare the DataLoaders
    if torch.cuda.is_available():
        train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        
        # TODO: use pin_memory to load each sample in corresponding GPU
        train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler)

        print("[rank {0}]: SLURM_JOB_NAME={1}; SLURM_ARRAY_TASK_ID={2}; train={3}, test={4}".format(rank, os.environ['SLURM_JOB_NAME'], os.environ['SLURM_ARRAY_TASK_ID'], len(train_dataloader), len(val_dataloader)), flush=True, file=args.log[0])
    else:
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # extract actions from the label file
    # NOTE: must be repeated by each process to create proper output tensor size for class distribution
    with open(args.actions, 'r') as action_names:
        actions = action_names.read().split('\n')

    # 0th class is always background action
    actions_dict = dict()
    for i, action in enumerate(actions):
        actions_dict[i+1] = action

    # do some setup only on Master Node and scatter results to other GPUs
    if rank == 0 or not torch.cuda.is_available():
        # extract skeleton graph data
        with open(args.graph, 'r') as graph_file:
            graph = json.load(graph_file)

        jobname = os.environ['SLURM_JOB_NAME'] if os.environ['SLURM_ARRAY_TASK_ID'] is None else os.environ['SLURM_JOB_NAME']+'_'+os.environ['SLURM_ARRAY_TASK_ID']

        # prepare a directory to store results
        save_dir = "{0}/{1}".format(
            args.out,
            jobname)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # create object list for scattering across processes
        objects = [{
            "graph": graph,
            "save_dir": save_dir,
            "jobname": jobname       
        }]

        # get class distribution
        class_dist = train_data.__get_distribution__(rank)
    elif rank != 0 and torch.cuda.is_available():
        objects = [None]
        class_dist = torch.zeros(len(actions_dict), dtype=torch.float32, device=rank)

    # scatter results to all other GPUs
    if torch.cuda.is_available():
        broadcast_object_list(objects, src=0)
        broadcast(class_dist, src=0)
    
    return train_dataloader, val_dataloader, class_dist, objects["graph"], actions_dict, objects["save_dir"], objects["jobname"]


def build_model(rank: int, args):
    """Builds the selected ST-GCN model variant.

    Args:
        args : ``dict``
            Parsed CLI arguments.

    Returns:
        PyTorch Model corresponding to the user-defined CLI parameters.

    Raises:
        ValueError:
            If GCN parameter list sizes do not match the number of stages.
    """

    if (len(args.in_ch) != args.stages or
        len(args.out_ch) != args.stages or
        len(args.stride) != args.stages or
        len(args.residual) != args.stages):
        raise ValueError(
            'GCN parameter list sizes do not match the number of stages. '
            'Check your config file.')
    elif (args.model == 'realtime' and args.buffer != 1):
        raise ValueError(
            'Selected the realtime model, but set buffer size to 1. '
            'Check your config file.')

    if args.model == 'original':
        model = OriginalStgcn(rank,**vars(args))
    else:
        # all 3 adapted versions are encapsulated in the same class, training is identical (batch mode),
        # usecase changes applied during inference
        model = RtStgcn(rank,**vars(args))

    # if using GPUs, convert the created model into the DistributedDataParallel
    if torch.cuda.is_available():
        model = DDP(model.to(rank), device_ids=[rank])

    return model


def train(rank: int, world_size: int, args):
    """Entry point for training functionality of a single selected model.

    Args:
        args : ``dict``
            Parsed CLI arguments.
    """

    # perform common setup around the model's black box
    train_dataloader, val_dataloader, class_dist, args.graph, actions, save_dir, args.jobname = common(rank, world_size, args)
    
    args.num_classes = len(actions)

    # construct the target model using the CLI arguments
    model = build_model(rank, args)
    # TODO: use barrier() to load checkpoint model to GPU and copy its initial state across GPUs
    # load the checkpoint if not trained from scratch
    # if args.checkpoint:
    #     # model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
    #     model.load_state_dict({
    #         k.split('module.')[1]: v 
    #         for k, v in
    #         torch.load(args.checkpoint)['model_state_dict'].items()})

    # construct a processing wrapper
    trainer = Processor(model, args.num_classes, class_dist, rank)

    if rank == 0 or not torch.cuda.is_available():
        start_time = time.time()

        # last dimension is the number of subjects in the scene (2 for datasets used)
        print("Training started", flush=True, file=args.log[0])

    # perform the training
    # (the model is trained on all skeletons in the scene, simultaneously)
    trainer.train(
        world_size=world_size,
        save_dir=save_dir,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **vars(args))

    if rank == 0 or not torch.cuda.is_available():
        print("Training completed in: {0}".format(time.time() - start_time), flush=True, file=args.log[0])

        # copy over resulting files of interest into the $VSC_DATA persistent storage
        if args.backup:
            backup_dir = "{0}/{1}".format(
                args.backup,
                args.jobname)

            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)

            for f in [
                'accuracy-curve.csv',
                'train-validation-curve.csv',
                'final.pt',
                'macro-F1@k.csv',
                'accuracy.csv',
                'edit.csv',
                'confusion-matrix.csv',
                *['segmentation-{0}.csv'.format(i) for i in args.demo]]:
                os.system('cp {0}/{1} {2}'.format(save_dir, f, backup_dir))

        os.system(
            'mail -s "[{0}]: COMPLETED" {1} <<< ""'
            .format(
                args.jobname,
                args.email))

    # cleanup
    if torch.cuda.is_available(): destroy_process_group()

    return


def test(rank: int, world_size: int, args):
    """Entry point for testing functionality of a single pretrained model.

    Args:
        args : ``dict``
            Parsed CLI arguments.
    """

    # perform common setup around the model's black box
    _, val_dataloader, class_dist, args.graph, actions, save_dir, args.jobname = common(rank, world_size, args)
    
    args.num_classes = len(actions)

    # construct the target model using the CLI arguments
    model = build_model(rank, args)
    # TODO: use barrier() to load checkpoint model to GPU and copy its initial state across GPUs
    # load the checkpoint if not trained from scratch
    # if args.checkpoint:
    #     # model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
    #     model.load_state_dict({
    #         k.split('module.')[1]: v 
    #         for k, v in
    #         torch.load(args.checkpoint, map_location=device)['model_state_dict'].items()})

    # construct a processing wrapper
    # NOTE: uses class distribution from training set
    trainer = Processor(model, args.num_classes, class_dist, rank)

    if rank == 0 or not torch.cuda.is_available():
        start_time = time.time()

        # last dimension is the number of subjects in the scene (2 for datasets used)
        print("Testing started", flush=True, file=args.log[0])

    # perform the testing
    trainer.test(
        world_size=world_size,
        save_dir=save_dir,
        dataloader=val_dataloader,
        **vars(args))

    if rank == 0 or not torch.cuda.is_available():
        print("Testing completed in: {0}".format(time.time() - start_time), flush=True, file=args.log[0])

        # copy over resulting files of interest into the $VSC_DATA persistent storage
        if args.backup:
            backup_dir = "{0}/{1}".format(
                args.backup,
                args.jobname)

            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)

            for f in [
                'macro-F1@k.csv',
                'accuracy.csv',
                'edit.csv',
                'confusion-matrix.csv',
                *['segmentation-{0}.csv'.format(i) for i in args.demo]]:
                os.system('cp {0}/{1} {2}'.format(save_dir, f, backup_dir))

        os.system(
            'mail -s "[{0}]: COMPLETED" {1} <<< ""'
            .format(
                args.jobname,
                args.email))

    # cleanup
    if torch.cuda.is_available(): destroy_process_group()

    return


def benchmark(rank: int, world_size: int, args):
    """Entry point for benchmarking inference of a model.

    TODO: adapt for DDP.

    Args:
        args : ``dict``
            Parsed CLI arguments.
    """

    # perform common setup around the model's black box
    # TODO: add entries for original ST-GCN modules
    _, val_dataloader, class_dist, args.graph, actions, save_dir, args.jobname = common(rank, world_size, args)
    
    args.num_classes = len(actions)
    args.prepare_dict = {
        "float_to_observed_custom_module_class": {
            "static": {
                AggregateStgcn: ObservedAggregateStgcn,
            }
        }
    }
    args.convert_dict = {
        "observed_to_quantized_custom_module_class": {
            "static": {
                ObservedAggregateStgcn: QAggregateStgcn,
            }
        }
    }

    # construct the target model using the CLI arguments
    model = build_model(rank, args)
    # TODO: use barrier() to load checkpoint model to GPU and copy its initial state across GPUs
    # load the checkpoint if not trained from scratch
    # if args.checkpoint:
    #     # model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
    #     model.load_state_dict({
    #         (k.split('module.')[1] if k.split('.')[1] != 'edge_importance' else 'st_gcn.{0}.edge_importance'.format(k.split('.')[2])): v
    #         for k, v in
    #         torch.load(args.checkpoint, map_location=device)['model_state_dict'].items()})

    # construct a processing wrapper
    trainer = Processor(model, args.num_classes, class_dist, rank)

    start_time = time.time()

    # last dimension is the number of subjects in the scene (2 for datasets used)
    print("Benchmarking started", flush=True, file=args.log[0])

    # perform the testing
    trainer.benchmark(save_dir, val_dataloader, **vars(args))

    print("Benchmarking completed in: {0}".format(time.time() - start_time), flush=True, file=args.log[0])

    if args.backup:
        backup_dir = "{0}/{1}".format(
            args.backup,
            args.jobname)

        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        for f in [
            'accuracy.csv',
            'loss.csv',
            'macro-F1@k.csv',
            'edit.csv',
            'latency.csv',
            'model-size.csv',
            'confusion-matrix_fp32.csv',
            'confusion-matrix_int8.csv',
            *['segmentation-{0}_fp32.csv'.format(i) for i in args.demo],
            *['segmentation-{0}_int8.csv'.format(i) for i in args.demo]]:
            os.system('cp {0}/{1} {2}'.format(save_dir, f, backup_dir))

    os.system(
        'mail -s "[{0}]: COMPLETED" {1} <<< ""'
        .format(
            args.jobname,
            args.email))

    # cleanup
    if torch.cuda.is_available(): destroy_process_group()

    return


if __name__ == '__main__':
    # top-level custom CLI parser
    # TODO: add `required` parameters to sub-parsers, where needed
    parser = st_gcn_parser.Parser(
        prog='main',
        description='Script for human action segmentation processing using ST-GCN networks.',
        epilog='TODO: add the epilog')

    subparsers= parser.add_subparsers(
        title='commands',
        dest='command',
        required=True)

    # train command parser (must manually update usage after changes
    # to the argument list or provide a custom formatter)
    parser_train = subparsers.add_parser(
        'train',
        usage="""%(prog)s [-h]
            \r\t[--config FILE]
            \r\t[--model MODEL {realtime|original}]
            \r\t[--strategy STRATEGY {uniform|distance|spatial}]
            \r\t[--in_feat IN_FEAT]
            \r\t[--stages STAGES]
            \r\t[--buffer BUFFER]
            \r\t[--kernel [KERNEL]]
            \r\t[--segment SEGMENT]
            \r\t[--importance]
            \r\t[--latency]
            \r\t[--receptive_field FIELD]
            \r\t[--layers [LAYERS]]
            \r\t[--in_ch [IN_CH,[...]]]
            \r\t[--out_ch [OUT_CH,[...]]]
            \r\t[--stride [STRIDE,[...]]]
            \r\t[--residual [RESIDUAL,[...]]]
            \r\t[--dropout [DROPOUT,[...]]]
            \r\t[--iou_threshold [THRESHOLDS]]
            \r\t[--graph FILE]

            \r\t[--seed SEED]
            \r\t[--epochs EPOCHS]
            \r\t[--checkpoints [CHECKPOINTS]]
            \r\t[--learning_rate RATE]
            \r\t[--learning_rate_decay RATE_DECAY]
            \r\t[--batch_size BATCH]

            \r\t[--data DATA_DIR]
            \r\t[--dataset_type TYPE]
            \r\t[--actions FILE]
            \r\t[--out OUT_DIR]
            \r\t[--backup BACKUP_DIR]
            \r\t[--checkpoint CHECKPOINT]
            \r\t[--log O_FILE E_FILE]
            \r\t[--email EMAIL]
            \r\t[-v[vv]]""",
        help='train target ST-GCN network',
        epilog='TODO: add the epilog')

    parser_train_model = parser_train.add_argument_group(
        'model',
        'arguments for configuring the ST-GCN model. '
        'If an argument is not provided, defaults to value inside config file. '
        'User can provide own config JSON file using --config argument, '
        'but it is the user\'s responsibility to provide all needed parameters')
    parser_train_optim = parser_train.add_argument_group(
        'optimizer',
        'arguments for configuring training')
    parser_train_io = parser_train.add_argument_group(
        'IO',
        'all miscallenous IO, log, file and path arguments')

    # model arguments
    parser_train_model.add_argument(
        '--config',
        type=str,
        default='config/pku-mmd/realtime_local.json',
        metavar='',
        help='path to the NN config file. Must be the last argument if combined '
            'with other CLI arguments. Provides default values for all arguments, except --log '
            '(default: config/pku-mmd/realtime_local.json)')
    parser_train_model.add_argument(
        '--model',
        choices=['realtime','original'],
        metavar='',
        help='type of NN model to use (default: realtime)')
    parser_train_model.add_argument(
        '--strategy',
        choices=['uniform','distance','spatial'],
        metavar='',
        help='type of graph partitioning strategy to use (default: spatial)')
    parser_train_model.add_argument(
        '--in_feat',
        type=int,
        metavar='',
        help='number of features/channels in data samples (default: 3)')
    parser_train_model.add_argument(
        '--stages',
        type=int,
        metavar='',
        help='number of ST-GCN stages to stack (default: 1)')
    parser_train_model.add_argument(
        '--buffer',
        type=int,
        metavar='',
        help='number of frames to buffer before batch processing. '
            'Applied only when --model=buffer_realtime (default: 1)')
    parser_train_model.add_argument(
        '--kernel',
        type=int,
        nargs='+',
        metavar='',
        help='list of temporal kernel sizes (Gamma) per stage (default: [9])')
    parser_train_model.add_argument(
        '--segment',
        type=int,
        metavar='',
        help='size of overlapping segments of frames to divide a trial into for '
            'parallelizing computation (creates a new batch dimension). '
            'Currently only supports datasets with different length trials. torch.cuda.device_count()ghting '
            '(default: True)')
    parser_train_model.add_argument(
        '--latency',
        default=False,
        action='store_true',
        help='flag specifying whether ST-GCN layers have half-buffer latency when --model!=original, '
            'or non-overlapping receptive field window when --model=original (default: False)')
    parser_train_model.add_argument(
        '--receptive_field',
        type=int,
        metavar='',
        help='number of frames in a sliding window across raw inputs. '
            'Applied only when --model=original. Should be selected proportionate to '
            'the kernel size to avoid operations with mostly zeroes (default: 50)')
    parser_train_model.add_argument(
        '--layers',
        type=int,
        nargs='+',
        metavar='',
        help='list of number of ST-GCN layers per stage (default: [9])')
    parser_train_model.add_argument(
        '--in_ch',
        type=int,
        nargs='+',
        action='append',
        metavar='',
        help='list of number of input channels per ST-GCN layer per stage. '
            'For multi-stage, pass --in_ch parameter multiple times '
            '(default: [[64,64,64,64,128,128,128,256,256]])')
    parser_train_model.add_argument(
        '--out_ch',
        type=int,
        nargs='+',
        action='append',
        metavar='',
        help='list of number of output channels per ST-GCN layer per stage. '
            'For multi-stage, pass --out_ch parameter multiple times '
            '(default: [[64,64,64,128,128,128,256,256,256]])')
    parser_train_model.add_argument(
        '--stride',
        type=int,
        nargs='+',
        action='append',
        metavar='',
        help='list of size of stride in temporal accumulation per ST-GCN layer per stage. '
            'For multi-stage, pass --stride parameter multiple times '
            '(default: [[1,1,1,2,1,1,2,1,1]])')
    parser_train_model.add_argument(
        '--residual',
        type=int,
        nargs='+',
        action='append',
        metavar='',
        help='list of binary flags specifying residual connection per ST-GCN layer per stage. '
            'For multi-stage, pass --residual parameter multiple times '
            '(default: [[0,1,1,1,1,1,1,1,1]])')
    parser_train_model.add_argument(
        '--dropout',
        type=float,
        nargs='+',
        action='append',
        metavar='',
        help='list of dropout values per ST-GCN layer per stage. '
            'For multi-stage, pass --dropout parameter multiple times '
            '(default: [[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])')
    parser_train_model.add_argument(
        '--iou_threshold',
        type=float,
        nargs='+',
        metavar='',
        help='list of IoU thesholds for F1@k metric (default: [0.1,0.25,0.5])')
    parser_train_model.add_argument(
        '--graph',
        type=str,
        metavar='',
        help='path to the skeleton graph specification file '
            '(default: data/skeletons/openpose.json)')
    # optimizer arguments
    parser_train_optim.add_argument(
        '--seed',
        type=int,
        metavar='',
        help='seed for the random number generator (default: 1538574472)')
    parser_train_optim.add_argument(
        '--epochs',
        type=int,
        metavar='',
        help='number of epochs to train the NN over (default: 100)')
    parser_train_optim.add_argument(
        '--checkpoints',
        type=int,
        nargs='+',
        metavar='',
        help='list of epochs to checkpoint the model at '
            '(default: [19, 39, 59, 79, 99])')
    parser_train_optim.add_argument(
        '--learning_rate',
        type=float,
        metavar='',
        help='learning rate of the optimizer (default: 0.01)')
    parser_train_optim.add_argument(
        '--learning_rate_decay',
        type=float,
        metavar='',
        help='learning rate decay factor of the optimizer (default: 0.1)')
    parser_train_optim.add_argument(
        '--batch_size',
        type=int,
        metavar='',
        help='number of captures to process in a minibatch (default: 16)')
    # IO arguments
    parser_train_io.add_argument(
        '--data',
        metavar='',
        help='path to the dataset directory (default: data/pku-mmd)')
    parser_train_io.add_argument(
        '--dataset_type',
        metavar='',
        help='type of the dataset (default: file)')
    parser_train_io.add_argument(
        '--actions',
        metavar='',
        help='path to the action classes file (default: data/pku-mmd/actions.txt)')
    parser_train_io.add_argument(
        '--out',
        metavar='',
        help='path to the output directory (default: pretrained_models/pku-mmd)')
    parser_train_io.add_argument(
        '--backup',
        metavar='',
        help='path to the backup directory to copy over final files after completion (default: None)')
    parser_train_io.add_argument(
        '--checkpoint',
        type=str,
        metavar='',
        default=None,
        help='path to the checkpoint to restore states from (default: None)')
    parser_train_io.add_argument(
        '--log',
        nargs=2,
        type=argparse.FileType('w'),
        # const=[t1+t2+'.txt' for t1, t2 in zip(['log.o.','log.e.'],2*[str(time.time())])],
        default=[None, None],
        metavar='',
        help='files to log the script to. Only argument without default option in --config '
            '(default: stdout, stderr)')
    parser_train_io.add_argument(
        '--email',
        type=str,
        metavar='',
        default=None,
        help='email address to send update notifications to (default: None)')
    parser_train_io.add_argument(
        '-v', '--verbose', dest='verbose',
        action='count',
        default=0,
        help='level of log detail (default: 0)')

    # test command parser
    parser_test = subparsers.add_parser(
        'test',
        usage="""%(prog)s\n\t[-h]
            \r\t[--config FILE]
            \r\t[--model MODEL {realtime|original}]
            \r\t[--strategy STRATEGY {uniform|distance|spatial}]
            \r\t[--in_feat IN_FEAT]
            \r\t[--stages STAGES]
            \r\t[--buffer BUFFER]
            \r\t[--kernel [KERNEL]]
            \r\t[--segment SEGMENT]
            \r\t[--importance]
            \r\t[--latency]
            \r\t[--receptive_field FIELD]
            \r\t[--layers [LAYERS]]
            \r\t[--in_ch [IN_CH,[...]]]
            \r\t[--out_ch [OUT_CH,[...]]]
            \r\t[--stride [STRIDE,[...]]]
            \r\t[--residual [RESIDUAL,[...]]]
            \r\t[--dropout [DROPOUT,[...]]]
            \r\t[--iou_threshold [THRESHOLDS]]
            \r\t[--graph FILE]

            \r\t[--data DATA_DIR]
            \r\t[--dataset_type TYPE]
            \r\t[--actions FILE]
            \r\t[--out OUT_DIR]
            \r\t[--backup BACKUP_DIR]
            \r\t[--checkpoint CHECKPOINT]
            \r\t[--log O_FILE E_FILE]
            \r\t[--email EMAIL]
            \r\t[-v[vv]]""",
        help='test target ST-GCN network',
        epilog='TODO: add the epilog')

    parser_test_model = parser_test.add_argument_group(
        'model',
        'arguments for configuring the ST-GCN model. '
        'If an argument is not provided, defaults to value inside config file. '
        'User can provide own config JSON file using --config argument, '
        'but it is the user\'s responsibility to provide all needed parameters')
    parser_test_io = parser_test.add_argument_group(
        'IO',
        'all miscallenous IO, log, file and path arguments')

    # model arguments
    parser_test_model.add_argument(
        '--config',
        type=str,
        default='config/pku-mmd/realtime_local.json',
        metavar='',
        help='path to the NN config file. Must be the last argument if combined '
            'with other CLI arguments. Provides default values for all arguments, except --log '
            '(default: config/pku-mmd/realtime_local.json)')
    parser_test_model.add_argument(
        '--model',
        choices=['realtime','original'],
        metavar='',
        help='type of NN model to use (default: realtime)')
    parser_test_model.add_argument(
        '--strategy',
        choices=['uniform','distance','spatial'],
        metavar='',
        help='type of graph partitioning strategy to use (default: spatial)')
    parser_test_model.add_argument(
        '--in_feat',
        type=int,
        metavar='',
        help='number of features/channels in data samples (default: 3)')
    parser_test_model.add_argument(
        '--stages',
        type=int,
        metavar='',
        help='number of ST-GCN stages to stack (default: 1)')
    parser_test_model.add_argument(
        '--buffer',
        type=int,
        metavar='',
        help='number of frames to buffer before batch processing. '
            'Applied only when --model=buffer_realtime (default: 1)')
    parser_test_model.add_argument(
        '--kernel',
        type=int,
        nargs='+',
        metavar='',
        help='list of temporal kernel sizes (Gamma) per stage (default: [9])')
    parser_test_model.add_argument(
        '--segment',
        type=int,
        metavar='',
        help='size of overlapping segments of frames to divide a trial into for '
            'parallelizing computation (creates a new batch dimension). '
            'Currently only supports datasets with different length trials. '
            'Applied only when --model != original and --dataset_type=dir (default: 100)')
    parser_test_model.add_argument(
        '--importance',
        default=True,
        action='store_true',
        help='flag specifying whether ST-GCN layers have edge importance weighting '
            '(default: True)')
    parser_test_model.add_argument(
        '--latency',
        default=False,
        action='store_true',
        help='flag specifying whether ST-GCN layers have half-buffer latency when --model!=original, '
            'or non-overlapping receptive field window when --model=original (default: False)')
    parser_test_model.add_argument(
        '--receptive_field',
        type=int,
        metavar='',
        help='number of frames in a sliding window across raw inputs. '
            'Applied only when --model=original. Should be selected proportionate to '
            'the kernel size to avoid operations with mostly zeroes (default: 50)')
    parser_test_model.add_argument(
        '--layers',
        type=int,
        nargs='+',
        metavar='',
        help='list of number of ST-GCN layers per stage (default: [9])')
    parser_test_model.add_argument(
        '--in_ch',
        type=int,
        nargs='+',
        action='append',
        metavar='',
        help='list of number of input channels per ST-GCN layer per stage. '
            'For multi-stage, pass --in_ch parameter multiple times '
            '(default: [[64,64,64,64,128,128,128,256,256]])')
    parser_test_model.add_argument(
        '--out_ch',
        type=int,
        nargs='+',
        action='append',
        metavar='',
        help='list of number of output channels per ST-GCN layer per stage. '
            'For multi-stage, pass --out_ch parameter multiple times '
            '(default: [[64,64,64,128,128,128,256,256,256]])')
    parser_test_model.add_argument(
        '--stride',
        type=int,
        nargs='+',
        action='append',
        metavar='',
        help='list of size of stride in temporal accumulation per ST-GCN layer per stage. '
            'For multi-stage, pass --stride parameter multiple times '
            '(default: [[1,1,1,2,1,1,2,1,1]])')
    parser_test_model.add_argument(
        '--residual',
        type=int,
        nargs='+',
        action='append',
        metavar='',
        help='list of binary flags specifying residual connection per ST-GCN layer per stage. '
            'For multi-stage, pass --residual parameter multiple times '
            '(default: [[0,1,1,1,1,1,1,1,1]])')
    parser_test_model.add_argument(
        '--dropout',
        type=float,
        nargs='+',
        action='append',
        metavar='',
        help='list of dropout values per ST-GCN layer per stage. '
            'For multi-stage, pass --dropout parameter multiple times '
            '(default: [[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])')
    parser_test_model.add_argument(
        '--iou_threshold',
        type=float,
        nargs='+',
        metavar='',
        help='list of IoU thesholds for F1@k metric (default: [0.1,0.25,0.5])')
    parser_test_model.add_argument(
        '--graph',
        type=str,
        metavar='',
        help='path to the skeleton graph specification file '
            '(default: data/skeletons/openpose.json)')
    # IO arguments
    parser_test_io.add_argument(
        '--data',
        metavar='',
        help='path to the dataset directory (default: data/pku-mmd)')
    parser_test_io.add_argument(
        '--dataset_type',
        metavar='',
        help='type of the dataset (default: file)')
    parser_test_io.add_argument(
        '--actions',
        metavar='',
        help='path to the action classes file (default: data/pku-mmd/actions.txt)')
    parser_test_io.add_argument(
        '--out',
        metavar='',
        help='path to the output directory (default: pretrained_models/pku-mmd)')
    parser_test_io.add_argument(
        '--backup',
        metavar='',
        help='path to the backup directory to copy over final files after completion (default: None)')
    parser_test_io.add_argument(
        '--checkpoint',
        type=str,
        metavar='',
        default=None,
        help='path to the checkpoint to restore states from (default: None)')
    parser_test_io.add_argument(
        '--log',
        nargs=2,
        type=argparse.FileType('w'),
        # const=[t1+t2+'.txt' for t1, t2 in zip(['log.o.','log.e.'],2*[str(time.time())])],
        default=[None, None],
        metavar='',
        help='files to log the script to. Only argument without default option in --config '
            '(default: stdout, stderr)')
    parser_test_io.add_argument(
        '--email',
        type=str,
        metavar='',
        default=None,
        help='email address to send update notifications to (default: None)')
    parser_test_io.add_argument(
        '-v', '--verbose', dest='verbose',
        action='count',
        default=0,
        help='level of log detail (default: 0)')

    # benchmark command parser
    parser_benchmark = subparsers.add_parser(
        'benchmark',
        usage="""%(prog)s\n\t[-h]
            \r\t[--config FILE]
            \r\t[--model MODEL {realtime|original}]
            \r\t[--strategy STRATEGY {uniform|distance|spatial}]
            \r\t[--in_feat IN_FEAT]
            \r\t[--stages STAGES]
            \r\t[--buffer BUFFER]
            \r\t[--kernel [KERNEL]]
            \r\t[--segment SEGMENT]
            \r\t[--importance]
            \r\t[--latency]
            \r\t[--receptive_field FIELD]
            \r\t[--layers [LAYERS]]
            \r\t[--in_ch [IN_CH,[...]]]
            \r\t[--out_ch [OUT_CH,[...]]]
            \r\t[--stride [STRIDE,[...]]]
            \r\t[--residual [RESIDUAL,[...]]]
            \r\t[--dropout [DROPOUT,[...]]]
            \r\t[--iou_threshold [THRESHOLDS]]
            \r\t[--graph FILE]
            \r\t[--backend ARCH {x86|qnnpack}]

            \r\t[--data DATA_DIR]
            \r\t[--dataset_type TYPE]
            \r\t[--actions FILE]
            \r\t[--out OUT_DIR]
            \r\t[--checkpoint CHECKPOINT]
            \r\t[--log O_FILE E_FILE]
            \r\t[--email EMAIL]
            \r\t[-v[vv]]""",
        help='benchmark target ST-GCN network (accuracy, scores, latency).',
        epilog='TODO: add the epilog')

    parser_benchmark_model = parser_benchmark.add_argument_group(
        'model',
        'arguments for configuring the ST-GCN model. '
        'If an argument is not provided, defaults to value inside config file. '
        'User can provide own config JSON file using --config argument, '
        'but it is the user\'s responsibility to provide all needed parameters')
    parser_benchmark_io = parser_benchmark.add_argument_group(
        'IO',
        'all miscallenous IO, log, file and path arguments')

    # model arguments
    parser_benchmark_model.add_argument(
        '--config',
        type=str,
        default='config/pku-mmd/realtime_local.json',
        metavar='',
        help='path to the NN config file. Must be the last argument if combined '
            'with other CLI arguments. Provides default values for all arguments, except --log '
            '(default: config/pku-mmd/realtime_local.json)')
    parser_benchmark_model.add_argument(
        '--model',
        choices=['realtime','original'],
        metavar='',
        help='type of NN model to use (default: realtime)')
    parser_benchmark_model.add_argument(
        '--strategy',
        choices=['uniform','distance','spatial'],
        metavar='',
        help='type of graph partitioning strategy to use (default: spatial)')
    parser_benchmark_model.add_argument(
        '--in_feat',
        type=int,
        metavar='',
        help='number of features/channels in data samples (default: 3)')
    parser_benchmark_model.add_argument(
        '--stages',
        type=int,
        metavar='',
        help='number of ST-GCN stages to stack (default: 1)')
    parser_benchmark_model.add_argument(
        '--buffer',
        type=int,
        metavar='',
        help='number of frames to buffer before batch processing. '
            'Applied only when --model=buffer_realtime (default: 1)')
    parser_benchmark_model.add_argument(
        '--kernel',
        type=int,
        nargs='+',
        metavar='',
        help='list of temporal kernel sizes (Gamma) per stage (default: [9])')
    parser_benchmark_model.add_argument(
        '--segment',
        type=int,
        metavar='',
        help='size of overlapping segments of frames to divide a trial into for '
            'parallelizing computation (creates a new batch dimension). '
            'Currently only supports datasets with different length trials. '
            'Applied only when --model != original and --dataset_type=dir (default: 100)')
    parser_benchmark_model.add_argument(
        '--importance',
        default=True,
        action='store_true',
        help='flag specifying whether ST-GCN layers have edge importance weighting '
            '(default: True)')
    parser_benchmark_model.add_argument(
        '--latency',
        default=False,
        action='store_true',
        help='flag specifying whether ST-GCN layers have half-buffer latency when --model!=original, '
            'or non-overlapping receptive field window when --model=original (default: False)')
    parser_benchmark_model.add_argument(
        '--receptive_field',
        type=int,
        metavar='',
        help='number of frames in a sliding window across raw inputs. '
            'Applied only when --model=original. Should be selected proportionate to '
            'the kernel size to avoid operations with mostly zeroes (default: 50)')
    parser_benchmark_model.add_argument(
        '--layers',
        type=int,
        nargs='+',
        metavar='',
        help='list of number of ST-GCN layers per stage (default: [9])')
    parser_benchmark_model.add_argument(
        '--in_ch',
        type=int,
        nargs='+',
        action='append',
        metavar='',
        help='list of number of input channels per ST-GCN layer per stage. '
            'For multi-stage, pass --in_ch parameter multiple times '
            '(default: [[64,64,64,64,128,128,128,256,256]])')
    parser_benchmark_model.add_argument(
        '--out_ch',
        type=int,
        nargs='+',
        action='append',
        metavar='',
        help='list of number of output channels per ST-GCN layer per stage. '
            'For multi-stage, pass --out_ch parameter multiple times '
            '(default: [[64,64,64,128,128,128,256,256,256]])')
    parser_benchmark_model.add_argument(
        '--stride',
        type=int,
        nargs='+',
        action='append',
        metavar='',
        help='list of size of stride in temporal accumulation per ST-GCN layer per stage. '
            'For multi-stage, pass --stride parameter multiple times '
            '(default: [[1,1,1,2,1,1,2,1,1]])')
    parser_benchmark_model.add_argument(
        '--residual',
        type=int,
        nargs='+',
        action='append',
        metavar='',
        help='list of binary flags specifying residual connection per ST-GCN layer per stage. '
            'For multi-stage, pass --residual parameter multiple times '
            '(default: [[0,1,1,1,1,1,1,1,1]])')
    parser_benchmark_model.add_argument(
        '--dropout',
        type=float,
        nargs='+',
        action='append',
        metavar='',
        help='list of dropout values per ST-GCN layer per stage. '
            'For multi-stage, pass --dropout parameter multiple times '
            '(default: [[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])')
    parser_benchmark_model.add_argument(
        '--iou_threshold',
        type=float,
        nargs='+',
        metavar='',
        help='list of IoU thesholds for F1@k metric (default: [0.1,0.25,0.5])')
    parser_benchmark_model.add_argument(
        '--graph',
        type=str,
        metavar='',
        help='path to the skeleton graph specification file '
            '(default: data/skeletons/openpose.json)')
    parser_benchmark_model.add_argument(
        '--backend',
        choices=['x86','qnnpack'],
        type=str,
        default='x86',
        metavar='',
        help='quantization backend (default: x86)')
    # IO arguments
    parser_benchmark_io.add_argument(
        '--data',
        metavar='',
        help='path to the dataset directory (default: data/pku-mmd)')
    parser_benchmark_io.add_argument(
        '--dataset_type',
        metavar='',
        help='type of the dataset (default: file)')
    parser_benchmark_io.add_argument(
        '--actions',
        metavar='',
        help='path to the action classes file (default: data/pku-mmd/actions.txt)')
    parser_benchmark_io.add_argument(
        '--out',
        metavar='',
        help='path to the output directory (default: pretrained_models/pku-mmd)')
    parser_benchmark_io.add_argument(
        '--checkpoint',
        type=str,
        metavar='',
        default=None,
        help='path to the checkpoint to restore states from (default: None)')
    parser_benchmark_io.add_argument(
        '--log',
        nargs=2,
        type=argparse.FileType('w'),
        # const=[t1+t2+'.txt' for t1, t2 in zip(['log.o.','log.e.'],2*[str(time.time())])],
        default=[None, None],
        metavar='',
        help='files to log the script to. Only argument without default option in --config '
            '(default: stdout, stderr)')
    parser_benchmark_io.add_argument(
        '--email',
        type=str,
        metavar='',
        default=None,
        help='email address to send update notifications to (default: None)')
    parser_benchmark_io.add_argument(
        '-v', '--verbose', dest='verbose',
        action='count',
        default=0,
        help='level of log detail (default: 0)')

    parser_train.set_defaults(func=train)
    parser_test.set_defaults(func=test)
    parser_benchmark.set_defaults(func=benchmark)

    # parse the arguments
    args = parser.parse_args()

    # setting up random number generator for deterministic and meaningful benchmarking
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # enter the appropriate command
    # will use all available GPUs for DistributedDataParallel model and spawn K processes, 1 for each GPU
    # otherwise will run as a CPU model
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        mp.spawn(args.func, args=(world_size, args), nprocs=world_size)
    else:
        args.func(None, None, args)
