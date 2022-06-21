import argparse
import sys
import json

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

with open('config/default.json','r') as f:
    data = json.load(f)

parser = argparse.ArgumentParser(description='Script for human action segmentation processing using ST-GCN networks.')
subparsers = parser.add_subparsers(metavar='command')
parser_train = subparsers.add_parser('train',help='train target ST-GCN network')
parser_train.add_argument('--model',choices=['realtime','batch_realtime','stream','original'],default='realtime',metavar='',help='type of NN model to use (default: realtime)')
parser_train.add_argument('--in_features',type=int,default=3,metavar='',help='number of features/channels in data samples (default: 3')
parser_train.add_argument('--stages',type=int,default=1,metavar='',help='number of ST-GCN stages to stack (default: 1')
parser_train.add_argument('--layers',type=int,nargs='+',default=[9],metavar='',help='list of number of ST-GCN layers per stage; (default: [9])')
parser_train.add_argument('--in_ch',type=int,nargs='+',action='append',metavar='',help='list of number of input channels per ST-GCN layer per stage; for multi-stage, pass --in_ch parameter multiple times (default: [[64,64,64,64,128,128,128,256,256]])')
parser_train.add_argument('--out_ch',type=int,nargs='+',action='append',metavar='',help='list of number of output channels per ST-GCN layer per stage; for multi-stage, pass --out_ch parameter multiple times (default: [[64,64,64,128,128,128,256,256,256]])')
parser_train.add_argument('--stride',type=int,nargs='+',action='append',metavar='',help='list of size of stride in temporal accumulation per ST-GCN layer per stage; for multi-stage, pass --stride parameter multiple times (default: [[1,1,1,2,1,1,2,1,1]])')
parser_train.add_argument('--residual',type=int,nargs='+',action='append',metavar='',help='list of binary flags specifying residual connection per ST-GCN layer per stage; for multi-stage, pass --residual parameter multiple times (default: [[0,1,1,1,1,1,1,1,1]])')
parser_train.add_argument('--dropout',type=float,nargs='+',action='append',metavar='',help='list of dropout values per ST-GCN layer per stage; for multi-stage, pass --dropout parameter multiple times (default: [[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]])')
parser_train.add_argument('--seed',type=int,default=1538574472,metavar='',help='seed for the random number generator (default: 1538574472)')
parser_train.add_argument('--epochs',type=int,default=100,metavar='',help='number of epochs to train the NN over (default: 100)')
parser_train.add_argument('--learning_rate',type=float,default=0.0005,metavar='',help='learning rate of the optimizer (default: 0.0005)')
parser_train.add_argument('--batch_size',type=int,default=16,metavar='',help='number of captures to process in a minibatch (default: 256)')
parser_train.add_argument('--data',default='/scratch/leuven/341/vsc34153/rt_st_gcn/data/kinetics',metavar='',help='path to the dataset directory (default: $VSC_SCRATCH/...)')
parser_train.add_argument('--out',default='/scratch/leuven/341/vsc34153/rt_st_gcn/pretrained_models/kinetics',metavar='',help='path to the output directory (default: $VSC_SCRATCH/...)')
parser_train.add_argument('--log',nargs=2,type=argparse.FileType('w'),default=[sys.stdout, sys.stderr],metavar='',help='files to log the script to (default: stdout, stderr)')
parser_train.add_argument('-v','--verbose',dest='verbose',action='count', default=0,help='level of log detail (default: 0)')

l = json_to_args(data)

a = parser_train.parse_args(l.split())
