import sys
sys.path.append('/home/hassin/Documents/University/Thesis/Realtime-ST-GCN')

import torch
import torch.nn as nn
import torch.nn.functional as F
from rtstgcn import Model as ModelUntilMatmul
from rtstgcn_unsqueeze_onwards import Model as ModelMatmulOnwards
import onnx
import json

with open("data/skeletons/pku-mmd.json", 'r') as f:
    graph_config = json.load(f)

config = {
        "strategy": "spatial",
        "in_feat": 3,
        "stages": 1,
        "kernel": 9,
        "output_type": "logits",
        "latency": False,
        "is_bn": True,
        "is_bn_stats": True,
        "graph": graph_config,
        "num_classes": 52,
        "normalization": "LayerNorm",
        "rt-st-gcn": {
            "latency": False,
            "importance": True,
            "in_feat": 3,
            "buffer": 1,
            "stages": 1,
            "layers": 1,
            "kernel": 9,
            "in_ch":
                [64], 
                # [64,64,64,64,128,128,128,256,256],
            "out_ch":
                [64],
                # [64,64,64,128,128,128,256,256,256],
            "stride":
                [1],
                # [1,1,1,2,1,1,2,1,1],
            "residual":
                [1],
                # [1,1,1,1,1,1,1,1,1],
            "dropout":
                [0],
                # [0,0,0,0,0,0,0,0,0]
        }
    }

def export_until_matmul():
    model = ModelUntilMatmul(**config)
    input = torch.randn([1, config["in_feat"], 1, 25])

    model._swap_layers_for_inference()
    model.eval_()
    model.eval()

    torch.onnx.export(model, input, "Tensil-generated/UntilMatmul/rtstgcn_until_matmul.onnx", opset_version=10)
    #Check the model
    m = onnx.load('Tensil-generated/UntilMatmul/rtstgcn_until_matmul.onnx')
    onnx.checker.check_model(m)

def export_matmul_onwards():
    model = ModelMatmulOnwards(**config)
    conv1 = torch.randn([1, 64, 1, 25])
    conv2 = torch.randn([1, 64, 1, 25])
    conv3 = torch.randn([1, 64, 1, 25])
    residual = torch.randn([1, 64, 1, 25])
    fifo = torch.zeros([1,9,64,25])

    model._swap_layers_for_inference()
    model.eval_()
    model.eval()

    torch.onnx.export(model, (residual, conv1,conv2, conv3, fifo), "Tensil-generated/MatmulOnwards/rtstgcn_matmul_onwards.onnx", opset_version=10)
    #Check the model
    m = onnx.load('Tensil-generated/MatmulOnwards/rtstgcn_matmul_onwards.onnx')
    onnx.checker.check_model(m)


def exportFullModel():
    export_until_matmul()
    export_matmul_onwards()

export_until_matmul()
