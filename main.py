#!/usr/bin/env python
import torch
from torch.utils.data import DataLoader
from data_prep.dataset import SkeletonDataset
from models.proposed.st_gcn import Model as ModelProposed
from models.original.st_gcn import Model as ModelOriginal
from processor import Processor
import argparse
import os
import random
import sys
import json

# TODO:
# 1. Implement train/test/benchmark functions for multi-GPU functions
# 2. Setup all CLI arguments for the 3 commands
# 3. Provide a way to monitor progress in realtime (e.g. TensorBoard)
# 4. Make the training and model parameters settable from the CLI config file

def common(args):
    # Setting up random number generator for deterministic and meaningful benchmarking
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Preparing datasets for training and validation
    train_data = SkeletonDataset('{0}/train_data.npy'.format(args.dir), '{0}/train_label.pkl'.format(args.dir))
    val_data = SkeletonDataset('{0}/val_data.npy'.format(args.dir), '{0}/val_label.pkl'.format(args.dir))

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

    # Extract actions from the label file
    with open('{0}/label_name.txt'.format(args.dir), 'r') as action_names:
        actions = action_names.read().split('\n')
    
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions, device, train_dataloader, val_dataloader

def construct(args, num_classes):
    # TODO: pass correct parameters to the constructors and **kwargs
    # edge importance, num_joints, graph
    # and update Model constructor
    if args.model == 'realtime':
        model =  ModelProposed(
            num_classes=num_classes,
            fifo_latency=False,
            **vars(args))
            # edge_importance_weighting=True,
            # num_joints=,
            # graph_args=args.graph)
    elif args.model == 'batch_realtime':
        model =  ModelProposed()
    elif args.model == 'stream':
        model =  ModelProposed()
    elif args.model == 'original':
        model =  ModelOriginal()
    # else is not needed because model choice is limited to the argparse options
    return model

def train(args):
    # TODO: Check all received arguments
    if not os.path.isdir(args.dir):
        print('Dataset path does not exist')
        sys.exit()
    
    ##################################################################

    # TODO: Read config files to setup the Graph


    # Construct the target model using the CLI arguments
    actions, device, train_dataloader, val_dataloader = common(args)
    num_classes = len(actions)
    model = construct(args, num_classes)

    trainer = Processor(model, num_classes)

    for i in range(1,2):
        print("Training subject: {0}".format(i), flush=True, file=args.log[1])
        
        model_dir = "./models/"+args.dataset+"/split_"+str(i)
        results_dir = "./results/"+args.dataset+"/split_"+str(i)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        trainer.train(
            model_dir, 
            batch_gen, 
            num_epochs=args.num_epochs, 
            batch_size=args.batch_size, 
            learning_rate=args.learning_rate, 
            device=device)        
    return


def test(args):
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device,)

    return


def benchmark(args):
    return


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            data = json.load(f)

        # Parse arguments in the file and store them in a blank namespace
        args = parser.parse_args(self.json_to_args(data).split(), namespace=None)

        # Set arguments in the target namespace if they haven’t been set yet
        for k, v in vars(args).items():
            if getattr(namespace, k, None) is not None:
                setattr(namespace, k, v)
    
        return

    def json_to_args(data):
        """Converts JSON config file to argument string for argparse"""
        arg = ''
        for group in data:
            for k, v in data[group].items():
                if type(v) is list and type(v[0]) is not list:
                    arg += ' --'+k+' '+' '.join([str(e) for e in v])
                elif type(v) is list and type(v[0]) is list:
                    for e in v:
                        arg += ' --'+k+' '+' '.join([str(ee) for ee in e])
                else:
                    arg += ' --'+k+' '+str(v)
        return arg


if __name__ == '__main__':
    # Top-level CLI parser
    parser = argparse.ArgumentParser(
        prog='main',
        description='Script for human action segmentation processing using ST-GCN networks.',
        epilog='TODO: add the epilog')
    subparsers = parser.add_subparsers(metavar='command')

    # Train command parser (must manually update usage after changes 
    # to the argument list or provide a custom formatter)
    parser_train = subparsers.add_parser(
        'train',
        usage="""%(prog)s [-h]
            \r\t[--config FILE]
            \r\t[--model MODEL {realtime|batch_realtime|stream|original}]
            \r\t[--in_features IN_FEATURES]
            \r\t[--stages STAGES]
            \r\t[--layers [LAYERS,[...]]]
            \r\t[--in_ch [IN_CH,[...]]]
            \r\t[--out_ch [OUT_CH,[...]]]
            \r\t[--stride [STRIDE,[...]]]
            \r\t[--residual [RESIDUAL,[...]]]
            \r\t[--dropout [DROPOUT,[...]]]
            \r\t[--kernel KERNEL]

            \r\t[--seed SEED]
            \r\t[--epochs EPOCHS]
            \r\t[--learning_rate RATE]
            \r\t[--batch_size BATCH]

            \r\t[--data DATA_DIR]
            \r\t[--out OUT_DIR]
            \r\t[--log O_FILE E_FILE]
            \r\t[-v[vv]]""",
        help='train target ST-GCN network',
        epilog='TODO: add the epilog')
    parser_train.set_defaults(func=train)

    parser_train_model = parser_train.add_argument_group(
        'model',
        'arguments for configuring the ST-GCN model; \
        if an argument is not provided, defaults to config file; \
        user can replace with own config JSON file using --config argument, \
        but it is responsibility of the user to enter all needed parameters in it')
    parser_train_optim = parser_train.add_argument_group(
        'optimizer',
        'arguments for configuring training')
    parser_train_io = parser_train.add_argument_group(
        'IO',
        'all IO, log, file and path arguments')

    # Model arguments
    parser_train_model.add_argument(
        '--config',
        type=open,
        action=LoadFromFile,
        default='config/kinetics/default.json',
        metavar='',
        help='path to NN config file; must be the last argument if not default \
            and other CLI arguments used (default: config/kinetics/default.json)')
    parser_train_model.add_argument(
        '--model',
        choices=['realtime','batch_realtime','stream','original'],
        default='realtime',
        metavar='',
        help='type of NN model to use (default: realtime)')
    parser_train_model.add_argument(
        '--in_features',
        type=int,
        default=3,
        metavar='',
        help='number of features/channels in data samples (default: 3')
    parser_train_model.add_argument(
        '--stages',
        type=int,
        default=1,
        metavar='',
        help='number of ST-GCN stages to stack (default: 1')
    parser_train_model.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=[9],
        metavar='',
        help='list of number of ST-GCN layers per stage (default: [9])')
    parser_train_model.add_argument(
        '--kernel',
        type=int,
        nargs='+',
        default=[9],
        metavar='',
        help='list of temporal kernel sizes (Gamma) per stage (default: [9])')
    parser_train_model.add_argument(
        '--in_ch',
        type=int,
        nargs='+',
        action='append',
        metavar='',
        help='list of number of input channels per ST-GCN layer per stage; \
            for multi-stage, pass --in_ch parameter multiple times \
            (default: [[64,64,64,64,128,128,128,256,256]])')
    parser_train_model.add_argument(
        '--out_ch',
        type=int, 
        nargs='+',
        action='append',
        metavar='',
        help='list of number of output channels per ST-GCN layer per stage; \
            for multi-stage, pass --out_ch parameter multiple times \
            (default: [[64,64,64,128,128,128,256,256,256]])')
    parser_train_model.add_argument(
        '--stride',
        type=int, 
        nargs='+',
        action='append',
        metavar='',
        help='list of size of stride in temporal accumulation per ST-GCN layer per stage; \
            for multi-stage, pass --stride parameter multiple times \
            (default: [[1,1,1,2,1,1,2,1,1]])')
    parser_train_model.add_argument(
        '--residual',
        type=int, 
        nargs='+',
        action='append',
        metavar='',
        help='list of binary flags specifying residual connection per ST-GCN layer per stage; \
            for multi-stage, pass --residual parameter multiple times \
            (default: [[0,1,1,1,1,1,1,1,1]])')
    parser_train_model.add_argument(
        '--dropout',
        type=float,
        nargs='+',
        action='append',
        metavar='',
        help='list of dropout values per ST-GCN layer per stage; \
            for multi-stage, pass --dropout parameter multiple times \
            (default: [[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])')
    # Optimizer arguments
    parser_train_optim.add_argument(
        '--seed',
        type=int,
        default=1538574472,
        metavar='',
        help='seed for the random number generator (default: 1538574472)')
    parser_train_optim.add_argument(
        '--epochs',
        type=int,
        default=100,
        metavar='',
        help='number of epochs to train the NN over (default: 100)')
    parser_train_optim.add_argument(
        '--learning_rate',
        type=float,
        default=0.0005,
        metavar='',
        help='learning rate of the optimizer (default: 0.0005)')
    parser_train_optim.add_argument(
        '--batch_size',
        type=int,
        default=16,
        metavar='',
        help='number of captures to process in a minibatch (default: 16)')
    # IO
    parser_train_io.add_argument(
        '--data',
        default='/scratch/leuven/341/vsc34153/rt_st_gcn/data/kinetics',
        metavar='',
        help='path to the dataset directory (default: $VSC_SCRATCH/rt_st_gcn/data/kinetics)')
    parser_train_io.add_argument(
        '--out',
        default='/scratch/leuven/341/vsc34153/rt_st_gcn/pretrained_models/kinetics',
        metavar='',
        help='path to the output directory (default: $VSC_SCRATCH/rt_st_gcn/pretrained_models/kinetics)')
    parser_train_io.add_argument(
        '--log',
        nargs=2,
        type=argparse.FileType('w'),
        # const=[t1+t2+'.txt' for t1, t2 in zip(['log.o.','log.e.'],2*[str(time.time())])],
        default=[sys.stdout, sys.stderr],
        metavar='',
        help='files to log the script to (default: stdout, stderr)')
    parser_train_io.add_argument(
        '-v', '--verbose', dest='verbose',
        action='count', 
        default=0,
        help='level of log detail (default: 0)')

    ##################################################################

    # Test command parser
    # TODO: setup all the needed CLI arguments
    parser_test = subparsers.add_parser(
        'test',
        usage="""%(prog)s\n\t[-h]
            \r\t[-m <model> {realtime|batch_realtime|stream|original}]
            \r\t[-d <data dir>]
            \r\t[-l <trained model>]
            \r\t[-b <batch size>]
            \r\t[-v[vv]]
            \r\t[-L <out file> <err file>]""",
        help='test target ST-GCN network',
        epilog='TODO: add the epilog')
    parser_test.set_defaults(func=test)

    parser_test.add_argument(
        '-m', '--model',
        choices=['realtime','batch_realtime','stream','original'],
        default='realtime',
        metavar='',
        help='type of NN model to use (default: realtime)')
    parser_test.add_argument(
        '-l', '--load',
        default='/scratch/leuven/341/vsc34153/rt_st_gcn/pretrained_models/kinetics',
        metavar='',
        help='path to pretrained weights of the model to load (default: $VSC_SCRATCH/...)')
    parser_test.add_argument(
        '-d', '--dir',
        default='/scratch/leuven/341/vsc34153/rt_st_gcn/data/kinetics',
        metavar='',
        help='path to the dataset directory (default: $VSC_SCRATCH/...)')
    parser_test.add_argument(
        '-b', '--batch_size', 
        type=int,
        default=256,
        metavar='',
        help='number of captures to process in a minibatch (default: 256)')
    parser_test.add_argument(
        '-v', '--verbose',
        action='count', 
        default=0,
        help='level of log detail (default: 0)')
    parser_test.add_argument(
        '-L', '--log',
        nargs=2,
        type=argparse.FileType('w'),
        # const=[t1+t2+'.txt' for t1, t2 in zip(['log.o.','log.e.'],2*[str(time.time())])],
        default=[sys.stdout, sys.stderr],
        metavar='',
        help='files to log the script to (default: stdout, stderr)')

    # Benchmark command parser
    # TODO: setup all the needed CLI arguments
    parser_benchmark = subparsers.add_parser(
        'benchmark',
        usage="""%(prog)s\n\t[-h]
            \r\t[-m <model> {realtime|batch_realtime|stream|original}]
            \r\t[-d <data dir>]
            \r\t[-l <trained model>]
            \r\t[-b <batch size>]
            \r\t[-v[vv]]
            \r\t[-L <out file> <err file>]""",
        help='benchmark target ST-GCN network against baseline(s)',
        epilog='TODO: add the epilog')
    parser_benchmark.set_defaults(func=benchmark)

    parser_benchmark.add_argument(
        '-m', '--model',
        choices=['realtime','batch_realtime','stream','original'],
        default='realtime',
        metavar='',
        help='type of NN model to use (default: realtime)')
    parser_benchmark.add_argument(
        '-l', '--load',
        default='/scratch/leuven/341/vsc34153/rt_st_gcn/pretrained_models/kinetics',
        metavar='',
        help='path to pretrained weights of the model to load (default: $VSC_SCRATCH/...)')
    parser_benchmark.add_argument(
        '-d', '--dir',
        default='/scratch/leuven/341/vsc34153/rt_st_gcn/data/kinetics',
        metavar='',
        help='path to the dataset directory (default: $VSC_SCRATCH/...)')
    parser_benchmark.add_argument(
        '-b', '--batch_size', 
        type=int,
        default=256,
        metavar='',
        help='number of captures to process in a minibatch (default: 256)')
    parser_benchmark.add_argument(
        '-v', '--verbose',
        action='count', 
        default=0,
        help='level of log detail (default: 0)')
    parser_benchmark.add_argument(
        '-L', '--log',
        nargs=2,
        type=argparse.FileType('w'),
        # const=[t1+t2+'.txt' for t1, t2 in zip(['log.o.','log.e.'],2*[str(time.time())])],
        default=[sys.stdout, sys.stderr],
        metavar='',
        help='files to log the script to (default: stdout, stderr)')

    # Parse the arguments
    args = parser.parse_args()
    # Scope will be routed to the default function corresponding to the selected command
    