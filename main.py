import torch
import torch.multiprocessing as mp

from processor import Processor, setup, cleanup
from models import MODELS
from utils import LOSS, SEGMENT_GENERATOR, STATISTICS
from utils import Parser
from utils.metrics import F1Score, EditScore, ConfusionMatrix

import os
import random


def pick_model(args):
    """Returns a constructor for the selected model variant.

    Args:
        args : ``dict``
            Parsed CLI arguments.

    Returns:
        PyTorch Model corresponding to the user-defined CLI parameters.
    """

    return \
        MODELS[args.processor['model']], \
        LOSS[args.processor['model']], \
        SEGMENT_GENERATOR[args.processor['model']], \
        STATISTICS[args.processor['model']]


def assert_parameters(args):
    """Performs model and job configuration parameter checks."""

    # do some desired parameter checking
    if (False):
        raise ValueError(
            'GCN parameter list sizes do not match the number of stages. '
            'Check your config file.')
    return None


def train(rank, world_size, args):
    """Entry point for training a single selected model.

    Args:
        rank :
            Local GPU index.

        world_size :
            Number of used GPUs.

        args : ``dict``
            Parsed CLI arguments.
    """

    # return reference to the user selected model constructor
    Model, Loss, SegmentGenerator, Statistics = pick_model(args)

    # perform common setup around the model's black box
    model, loss, segment_generator, statistics, train_dataloader, val_dataloader, args = setup(Model, Loss, SegmentGenerator, Statistics, rank, world_size, args)

    # list metrics that Processor should record
    metric_rank = rank if args.processor['is_ddp'] or not torch.cuda.is_available() else torch.device("cuda:0")
    metric_world_size = world_size if args.processor['is_ddp'] and torch.cuda.is_available() else 1
    metrics = [
        F1Score(metric_rank, metric_world_size, args.arch['num_classes'], args.processor['iou_threshold']),
        EditScore(metric_rank, metric_world_size, args.arch['num_classes']),
        ConfusionMatrix(metric_rank, metric_world_size, args.arch['num_classes'])]

    # construct a processing wrapper
    processor = Processor(rank, world_size, model, loss, statistics, segment_generator, metrics)

    # perform the training
    # (the model is trained on all skeletons in the scene, simultaneously)
    processor.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        proc_conf=args.processor,
        optim_conf=args.optimizer,
        job_conf=args.job)

    if rank == 0 or not torch.cuda.is_available() or not args.processor['is_ddp']:
        # copy over resulting files of interest into the $VSC_DATA persistent storage
        if args.processor.get('backup'):
            for f in [
                'accuracy-curve.csv',
                'train-validation-curve.csv',
                'final.pt',
                'macro-F1@k.csv',
                'accuracy.csv',
                'edit.csv',
                'confusion-matrix.csv',
                *['segmentation-{0}.csv'.format(i) for i in args.processor['demo']]]:
                os.system('cp {0}/{1} {2}'.format(args.processor['save_dir'], f, args.processor['backup_dir']))

        os.system(
            'mail -s "[{0}]: COMPLETED" {1} <<< ""'
            .format(
                args.job['jobname'],
                args.job['email']))

    # perform common cleanup
    cleanup(args)

    return None


def test(rank, world_size, args):
    """Entry point for testing performance of a single pretrained model.

    Args:
        rank :
            Local GPU index.

        world_size :
            Number of used GPUs.

        args : ``dict``
            Parsed CLI arguments.
    """

    # return reference to the user selected model constructor
    Model, Loss, SegmentGenerator, Statistics = pick_model(args)

    # perform common setup around the model's black box
    model, loss, segment_generator, statistics, train_dataloader, val_dataloader, args = setup(Model, Loss, SegmentGenerator, Statistics, rank, world_size, args)

    # list metrics that Processor should record
    metrics = [
        F1Score(rank, world_size, args.arch['num_classes'], args.processor['iou_threshold']),
        EditScore(rank, world_size, args.arch['num_classes']),
        ConfusionMatrix(rank, world_size, args.arch['num_classes'])]

    # construct a processing wrapper
    processor = Processor(rank, world_size, model, loss, statistics, segment_generator, metrics)

    # perform the testing
    processor.test(
        dataloader=val_dataloader,
        proc_conf=args.processor,
        job_conf=args.job)

    if rank == 0 or not torch.cuda.is_available() or not args.processor['is_ddp']:
        # copy over resulting files of interest into the $VSC_DATA persistent storage
        if args.processor.get('backup'):
            for f in [
                'macro-F1@k.csv',
                'accuracy.csv',
                'edit.csv',
                'confusion-matrix.csv',
                *['segmentation-{0}.csv'.format(i) for i in args.processor['demo']]]:
                os.system('cp {0}/{1} {2}'.format(args.processor['save_dir'], f, args.processor['backup_dir']))

        os.system(
            'mail -s "[{0}]: COMPLETED" {1} <<< ""'
            .format(
                args.job['jobname'],
                args.job['email']))

    # perform common cleanup
    cleanup(args)

    return None


def benchmark(rank, world_size, args):
    """Entry point for benchmarking inference of a model, including quantization.

    TODO: add custom quantization conversion modules for other models

    Args:
        rank :
            Local GPU index.

        world_size :
            Number of used GPUs.

        args : ``dict``
            Parsed CLI arguments.
    """

    # return reference to the user selected model constructor
    Model, Loss, SegmentGenerator, Statistics = pick_model(args)

    # perform common setup around the model's black box
    model, loss, segment_generator, statistics, train_dataloader, val_dataloader, args = setup(Model, Loss, SegmentGenerator, Statistics, rank, world_size, args)

    # get custom quantization details if the model needs any
    # maps custom quantization replacement modules
    args.arch.prepare_dict = model.prepare_dict()
    args.arch.convert_dict = model.convert_dict()

    # list metrics that Processor should record
    metrics = [
        F1Score(rank, world_size, args.arch['num_classes'], args.processor['iou_threshold']),
        EditScore(rank, world_size, args.arch['num_classes']),
        ConfusionMatrix(rank, world_size, args.arch['num_classes'])]

    # construct a processing wrapper
    processor = Processor(rank, world_size, model, loss, statistics, segment_generator, metrics)

    # perform the testing
    processor.benchmark(
        dataloader=val_dataloader,
        proc_conf=args.processor,
        arch_conf=args.arch,
        job_conf=args.job)

    if rank == 0 or not torch.cuda.is_available() or not args.processor['is_ddp']:
        if args.processor.get('backup'):
            for f in [
                'accuracy.csv',
                'loss.csv',
                'macro-F1@k.csv',
                'edit.csv',
                'latency.csv',
                'model-size.csv',
                'confusion-matrix_fp32.csv',
                'confusion-matrix_int8.csv',
                *['segmentation-{0}_fp32.csv'.format(i) for i in args.processor['demo']],
                *['segmentation-{0}_int8.csv'.format(i) for i in args.processor['demo']]]:
                os.system('cp {0}/{1} {2}'.format(args.processor['save_dir'], f, args.processor['backup_dir']))

        os.system(
            'mail -s "[{0}]: COMPLETED" {1} <<< ""'
            .format(
                args.job['jobname'],
                args.job['email']))

    # perform common cleanup
    cleanup(args)

    return None


def main(args):
    """Entrypoint into the script that routes to the correct function."""

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888' 
    
    # check user inputs using user logic
    assert_parameters(args)

    # setting up random number generator for deterministic and meaningful benchmarking
    random.seed(args.optimizer['seed'])
    torch.manual_seed(args.optimizer['seed'])
    torch.cuda.manual_seed_all(args.optimizer['seed'])
    torch.backends.cudnn.deterministic = True

    # enter the appropriate command

    # will use all available GPUs for DistributedDataParallel model and spawn K processes, 1 for each GPU
    # otherwise will run as a CPU model
    if torch.cuda.is_available() and args.processor['is_ddp']:
        world_size = torch.cuda.device_count()
        mp.spawn(args.func, args=(world_size, args), nprocs=world_size)
    else:
        args.func(None, None, args)

    return None


if __name__ == '__main__':
    # top-level custom CLI parser -> 
    parser = Parser(
        prog='main',
        description="""Script for continual human action recognition model processing.
            \nSupports: {{{0}}}""".format('|'.join(MODELS.keys())),
        epilog='Maxim Yudayev (maxim.yudayev@kuleuven.be)')

    subparsers= parser.add_subparsers(
        title='commands',
        dest='command',
        required=True)

    # train command parser
    parser_train = subparsers.add_parser(
        'train',
        usage="""%(prog)s [-h]
            \r\t[--config FILE]""",
        help='train target continual HAR network',
        epilog='Maxim Yudayev (maxim.yudayev@kuleuven.be)')
    parser_train.add_argument(
        '--config',
        type=str,
        default='config/pku-mmd/stgcn_local.json',
        metavar='',
        help='path to the NN config file. Must be the last argument if combined '
            'with other CLI arguments. Provides default values for all arguments, except --log '
            '(default: config/pku-mmd/stgcn_local.json)')
    
    # test command parser
    parser_test = subparsers.add_parser(
        'test',
        usage="""%(prog)s\n\t[-h]
            \r\t[--config FILE]""",
        help='test target continual HAR network',
        epilog='Maxim Yudayev (maxim.yudayev@kuleuven.be)')
    parser_test.add_argument(
        '--config',
        type=str,
        default='config/pku-mmd/stgcn_local.json',
        metavar='',
        help='path to the NN config file. Must be the last argument if combined '
            'with other CLI arguments. Provides default values for all arguments, except --log '
            '(default: config/pku-mmd/stgcn_local.json)')

    # benchmark command parser
    parser_benchmark = subparsers.add_parser(
        'benchmark',
        usage="""%(prog)s\n\t[-h]
            \r\t[--config FILE]""",
        help='benchmark target ST-GCN network (accuracy, scores, latency).',
        epilog='Maxim Yudayev (maxim.yudayev@kuleuven.be)')
    parser_benchmark.add_argument(
        '--config',
        type=str,
        default='config/pku-mmd/stgcn_local.json',
        metavar='',
        help='path to the NN config file. Must be the last argument if combined '
            'with other CLI arguments. Provides default values for all arguments, except --log '
            '(default: config/pku-mmd/stgcn_local.json)')

    parser_train.set_defaults(func=train)
    parser_test.set_defaults(func=test)
    parser_benchmark.set_defaults(func=benchmark)

    # parse the arguments
    main(parser.parse_args())
