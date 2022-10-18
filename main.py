import torch
from torch.utils.data import DataLoader
from data_prep.dataset import SkeletonDataset, SkeletonDatasetFromDirectory
from models.proposed.st_gcn import Stgcn
from models.original.st_gcn import Model as OriginalStgcn
from processor import Processor
import st_gcn_parser
import argparse
import os
import sys
import random
import time
import json

# TODO:
# 1. Implement benchmark functions for multi-GPU functions
# 2. Setup all CLI arguments for the 3 commands

def common(args):
    """Performs setup common to any ST-GCN model variant.
    
    Only needs to be invoked once for a given problem (train-test, benchmark, etc.). 
    Corresponds to the parts of the pipeline irrespective of the black-box model used.
    Creates DataLoaders, sets up processing device and random number generator,
    reads action classes file.

    Args:
        args : ``dict``
            Parsed CLI arguments.

    Returns:
        Dictionary of action classes.

        PyTorch device (CPU or GPU).

        Train and validation DataLoaders.
    """
    
    # setting up random number generator for deterministic and meaningful benchmarking
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # preparing datasets for training and validation
    if args.dataset_type == 'file':
        train_data = SkeletonDataset('{0}/train_data.npy'.format(args.data), '{0}/train_label.pkl'.format(args.data))
        val_data = SkeletonDataset('{0}/val_data.npy'.format(args.data), '{0}/val_label.pkl'.format(args.data))
    elif args.dataset_type == 'dir':
        train_data = SkeletonDatasetFromDirectory('{0}/train/features'.format(args.data), '{0}/train/labels'.format(args.data))
        val_data = SkeletonDatasetFromDirectory('{0}/val/features'.format(args.data), '{0}/val/labels'.format(args.data))

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    
    # extract skeleton graph data
    with open(args.graph, 'r') as graph_file:
        graph = json.load(graph_file)

    # extract actions from the label file
    with open(args.actions, 'r') as action_names:
        actions = action_names.read().split('\n')

    # 0th class is always background action
    actions_dict = {0: "background"}
    for i, action in enumerate(actions):
        actions_dict[i+1] = action

    # prepare a directory to store results
    if not os.getenv('PBS_JOBID'):
        with open('.vscode/pbs_jobid.txt', 'r+') as f:
            job_id = f.readline()
            os.environ['PBS_JOBID'] = job_id
            f.seek(0)
            f.write(str(int(job_id)+1))
            f.truncate()
    
    save_dir = "{0}/{1}/run_{2}".format(args.out, args.model, os.getenv('PBS_JOBID').split('.')[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return graph, actions_dict, device, train_dataloader, val_dataloader, save_dir


def build_model(args):
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
        model = OriginalStgcn(**vars(args))
    else:
        # all 3 adapted versions are encapsulated in the same class, training is identical (batch mode),
        # usecase changes applied during inference
        model = Stgcn(**vars(args))
    
    return model


def train(args):
    """Entry point for training functionality of a single selected model.

    Args:
        args : ``dict``
            Parsed CLI arguments.
    """

    # perform common setup around the model's black box
    args.graph, actions, device, train_dataloader, val_dataloader, save_dir = common(args)
    args.num_classes = len(actions)
    
    # record the length of captures
    data, _ = next(iter(train_dataloader))

    # construct the target model using the CLI arguments
    model = build_model(args)
    # load the checkpoint if not trained from scratch
    if args.checkpoint:
        model.load_state_dict({
            k.split('module.')[1]: v 
            for k, v in
            torch.load(args.checkpoint, map_location=device)['model_state_dict'].items()})

    # construct a processing wrapper
    trainer = Processor(model, args.num_classes)

    start_time = time.time()

    # last dimension is the number of subjects in the scene (2 for datasets used)
    print("Training started", flush=True, file=args.log[0])
    
    # perform the training
    # (the model is trained on all skeletons in the scene, simultaneously)
    trainer.train(
        save_dir=save_dir,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,    
        **vars(args))
    
    print("Training completed in: {0}".format(time.time() - start_time), flush=True, file=args.log[0])
    
    os.system(
        'mail -s "[{0}]: $PBS_JOBNAME - COMPLETED" {1} <<< ""'
        .format(
            os.getenv('PBS_JOBID').split('.')[0],
            args.email))

    return


def test(args):
    """Entry point for testing functionality of a single pretrained model.

    Args:
        args : ``dict``
            Parsed CLI arguments.
    """

    # perform common setup around the model's black box
    args.graph, actions, device, _, val_dataloader, save_dir = common(args)
    args.num_classes = len(actions)
    
    # split between the subjects in the captures
    data, _ = next(iter(val_dataloader))
    args.capture_length = data.shape[2]

    # construct the target model using the CLI arguments
    model = build_model(args)
    # load the checkpoint if not trained from scratch
    if args.checkpoint:
        model.load_state_dict({
            k.split('module.')[1]: v 
            for k, v in
            torch.load(args.checkpoint, map_location=device)['model_state_dict'].items()})

    # construct a processing wrapper
    trainer = Processor(model, args.num_classes)

    start_time = time.time()

    # last dimension is the number of subjects in the scene (2 for datasets used)
    print("Testing started", flush=True, file=args.log[0])
    
    # perform the testing
    trainer.test(save_dir, val_dataloader, device, **vars(args))
    
    print("Testing completed in: {0}".format(time.time() - start_time), flush=True, file=args.log[0])
    
    os.system(
        'mail -s "[{0}]: $PBS_JOBNAME - COMPLETED" {1} <<< ""'
        .format(
            os.getenv('PBS_JOBID').split('.')[0], 
            args.email))

    return


def benchmark(args):
    """Entry point for benchmarking functionality of multiple pretrained models.

    TODO: complete

    Args:
        args : ``dict``
            Parsed CLI arguments.
    """

    # perform common setup around the model's black box
    args.graph, actions, device, _, val_dataloader, save_dir = common(args)
    args.num_classes = len(actions)
    
    # split between the subjects in the captures
    data, _ = next(iter(val_dataloader))
    args.capture_length = data.shape[2]

    # construct the target models using the CLI arguments
    models = []
    for m in args.models:
        model = build_model(m)
        
        model.load_state_dict({
            k.split('module.')[1]: v 
            for k, v in
            torch.load(args.checkpoint, map_location=device)['model_state_dict'].items()})

        models.append(model)

    # construct a processing wrapper
    trainer = Processor(model, args.num_classes)

    start_time = time.time()

    # last dimension is the number of subjects in the scene (2 for datasets used)
    print("Testing started", flush=True, file=args.log[0])
    
    # perform the testing
    trainer.test(save_dir, val_dataloader, device, **vars(args))

    print("Benchmarking completed in: {0}".format(time.time() - start_time), flush=True, file=args.log[0])
    
    os.system(
        'mail -s "[{0}]: $PBS_JOBNAME - COMPLETED" {1} <<< ""'
        .format(
            os.getenv('PBS_JOBID').split('.')[0],
            args.email))

    return


if __name__ == '__main__':
    # top-level custom CLI parser
    parser = st_gcn_parser.Parser(
        prog='main',
        description='Script for human action segmentation processing using ST-GCN networks.',
        epilog='TODO: add the epilog')
    
    subparsers= parser.add_subparsers(metavar='command')
        
    # train command parser (must manually update usage after changes 
    # to the argument list or provide a custom formatter)
    parser_train = subparsers.add_parser(
        'train',
        usage="""%(prog)s [-h]
            \r\t[--config FILE]            
            \r\t[--model MODEL {realtime|buffer_realtime|batch|original}]
            \r\t[--strategy STRATEGY {uniform|distance|spatial}]
            \r\t[--in_feat IN_FEAT]
            \r\t[--stages STAGES]
            \r\t[--buffer BUFFER]
            \r\t[--kernel [KERNEL]]
            \r\t[--importance]
            \r\t[--latency]
            \r\t[--layers [LAYERS]]
            \r\t[--in_ch [IN_CH,[...]]]
            \r\t[--out_ch [OUT_CH,[...]]]
            \r\t[--stride [STRIDE,[...]]]
            \r\t[--residual [RESIDUAL,[...]]]
            \r\t[--dropout [DROPOUT,[...]]]
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
        default='config/kinetics/realtime_local.json',
        metavar='',
        help='path to the NN config file. Must be the last argument if combined '
            'with other CLI arguments. Provides default values for all arguments, except --log '
            '(default: config/kinetics/realtime_local.json)')
    parser_train_model.add_argument(
        '--model',
        choices=['realtime','buffer_realtime','batch','original'],
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
        '--importance',
        default=True,
        action='store_true',
        help='flag specifying whether ST-GCN layers have edge importance weighting '
            '(default: True)')
    parser_train_model.add_argument(
        '--latency',
        default=False,
        action='store_true',
        help='flag specifying whether ST-GCN layers have half-buffer latency '
            '(default: False)')
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
        help='path to the dataset directory (default: data/kinetics)')
    parser_train_io.add_argument(
        '--dataset_type',
        metavar='',
        help='type of the dataset (default: file)')
    parser_train_io.add_argument(
        '--actions',
        metavar='',
        help='path to the action classes file (default: data/kinetics/actions.txt)')
    parser_train_io.add_argument(
        '--out',
        metavar='',
        help='path to the output directory (default: pretrained_models/kinetics)')
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
        default=[sys.stdout, sys.stderr],
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
            \r\t[--model MODEL {realtime|buffer_realtime|batch|original}]
            \r\t[--strategy STRATEGY {uniform|distance|spatial}]
            \r\t[--in_feat IN_FEAT]
            \r\t[--stages STAGES]
            \r\t[--buffer BUFFER]
            \r\t[--kernel [KERNEL]]
            \r\t[--importance]
            \r\t[--latency]
            \r\t[--layers [LAYERS]]
            \r\t[--in_ch [IN_CH,[...]]]
            \r\t[--out_ch [OUT_CH,[...]]]
            \r\t[--stride [STRIDE,[...]]]
            \r\t[--residual [RESIDUAL,[...]]]
            \r\t[--dropout [DROPOUT,[...]]]
            \r\t[--graph FILE]

            \r\t[--data DATA_DIR]
            \r\t[--dataset_type TYPE]
            \r\t[--actions FILE]
            \r\t[--out OUT_DIR]
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
        default='config/kinetics/realtime_local.json',
        metavar='',
        help='path to the NN config file. Must be the last argument if combined '
            'with other CLI arguments. Provides default values for all arguments, except --log '
            '(default: config/kinetics/realtime_local.json)')
    parser_test_model.add_argument(
        '--model',
        choices=['realtime','buffer_realtime','batch','original'],
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
        '--importance',
        default=True,
        action='store_true',
        help='flag specifying whether ST-GCN layers have edge importance weighting '
            '(default: True)')
    parser_test_model.add_argument(
        '--latency',
        default=False,
        action='store_true',
        help='flag specifying whether ST-GCN layers have half-buffer latency '
            '(default: False)')
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
        '--graph',
        type=str,
        metavar='',
        help='path to the skeleton graph specification file '
            '(default: data/skeletons/openpose.json)')
    # IO arguments
    parser_test_io.add_argument(
        '--data',
        metavar='',
        help='path to the dataset directory (default: data/kinetics)')
    parser_test_io.add_argument(
        '--dataset_type',
        metavar='',
        help='type of the dataset (default: file)')
    parser_test_io.add_argument(
        '--actions',
        metavar='',
        help='path to the action classes file (default: data/kinetics/actions.txt)')
    parser_test_io.add_argument(
        '--out',
        metavar='',
        help='path to the output directory (default: pretrained_models/kinetics)')
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
        default=[sys.stdout, sys.stderr],
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

    ##################################################################
    # benchmark command parser
    # TODO: setup all the needed CLI arguments
    parser_benchmark = subparsers.add_parser(
        'benchmark',
        usage="""%(prog)s\n\t[-h]
            
            """,
        help='benchmark target ST-GCN network against baseline(s)',
        epilog='TODO: add the epilog')
    ##################################################################

    parser_train.set_defaults(func=train)
    parser_test.set_defaults(func=test)
    parser_benchmark.set_defaults(func=benchmark)

    # parse the arguments
    args = parser.parse_args()

    # enter the appropriate command
    args.func(args)
    