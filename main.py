import torch

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


def train(world_size, args):
    """Entry point for training a single selected model.

    Args:
        rank :
            Local GPU index.

        world_size :
            Number of used GPUs.

        args : ``dict``
            Parsed CLI arguments.
    """

    output_device = world_size-1

    # return reference to the user selected model constructor
    Model, Loss, SegmentGenerator, Statistics = pick_model(args)

    # perform common setup around the model's black box
    model, loss, segment_generator, statistics, train_dataloader, val_dataloader, args = setup(Model, Loss, SegmentGenerator, Statistics, world_size, args)

    # list metrics that Processor should record
    metrics = [
        F1Score(output_device, args.arch['num_classes'], args.processor['iou_threshold']),
        EditScore(output_device, args.arch['num_classes']),
        ConfusionMatrix(output_device, args.arch['num_classes'])]

    # construct a processing wrapper
    processor = Processor(world_size, model, loss, statistics, segment_generator, metrics)

    # perform the training
    # (the model is trained on all skeletons in the scene, simultaneously)
    processor.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        proc_conf=args.processor,
        optim_conf=args.optimizer,
        job_conf=args.job)

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


def test(world_size, args):
    """Entry point for testing performance of a single pretrained model.

    Args:
        rank :
            Local GPU index.

        world_size :
            Number of used GPUs.

        args : ``dict``
            Parsed CLI arguments.
    """

    output_device = world_size-1

    # return reference to the user selected model constructor
    Model, Loss, SegmentGenerator, Statistics = pick_model(args)

    # perform common setup around the model's black box
    model, loss, segment_generator, statistics, train_dataloader, val_dataloader, args = setup(Model, Loss, SegmentGenerator, Statistics, world_size, args)

    # list metrics that Processor should record
    metrics = [
        F1Score(output_device, args.arch['num_classes'], args.processor['iou_threshold']),
        EditScore(output_device, args.arch['num_classes']),
        ConfusionMatrix(output_device, args.arch['num_classes'])]

    # construct a processing wrapper
    processor = Processor(world_size, model, loss, statistics, segment_generator, metrics)

    # perform the testing
    processor.test(
        dataloader=val_dataloader,
        proc_conf=args.processor,
        job_conf=args.job)

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


def benchmark(world_size, args):
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

    output_device = world_size-1

    # return reference to the user selected model constructor
    Model, Loss, SegmentGenerator, Statistics = pick_model(args)

    # perform common setup around the model's black box
    model, loss, segment_generator, statistics, train_dataloader, val_dataloader, args = setup(Model, Loss, SegmentGenerator, Statistics, world_size, args)

    # get custom quantization details if the model needs any
    # maps custom quantization replacement modules
    args.arch['prepare_dict'] = Model.prepare_dict()
    args.arch['convert_dict'] = Model.convert_dict()

    # list metrics that Processor should record
    metrics = [
        F1Score(output_device, args.arch['num_classes'], args.processor['iou_threshold']),
        EditScore(output_device, args.arch['num_classes']),
        ConfusionMatrix(output_device, args.arch['num_classes'])]

    # construct a processing wrapper
    processor = Processor(world_size, model, loss, statistics, segment_generator, metrics)

    # perform the testing
    processor.benchmark(
        dataloader=val_dataloader,
        proc_conf=args.processor,
        arch_conf=args.arch,
        job_conf=args.job)

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

    # check user inputs using user logic
    assert_parameters(args)

    # setting up random number generator for deterministic and meaningful benchmarking
    random.seed(args.optimizer['seed'])
    torch.manual_seed(args.optimizer['seed'])
    torch.cuda.manual_seed_all(args.optimizer['seed'])
    torch.backends.cudnn.deterministic = True

    # enter the appropriate command
    args.func(args)

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
