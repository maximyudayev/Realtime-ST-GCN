# Realtime ST-GCN
Correct implementation of ST-GCN inline with the original paper's proposed formulae, and adapted to realtime processing.

Classes are decoupled for modularity and readability, with separation of concern dictated by the user's JSON configuration files or CLI arguments.

## TODO
- [x] Implement ST-GCN correctly to the paper's spec (but in RT variant), using basic differentiable tensor operators and cutting out messy Modules combination (stacked GCN + TCN).
- [x] Validate the network (tensor dimension changes) manually by chaining PyTorch operators.
- [x] Write data preparation and loading backend for out-of-core big data files.
- [x] Write a script to leverage [KU Leuven HPC](https://www.vscentrum.be/) infrastructure for the PyTorch workflow in a simple automated process that originates on the local machine.
- [x] Add support for frame buffered realtime processing.
- [x] Validate the code.
- [x] Train the models.
- [ ] Add support for 2 FIFO latency variants.
- [ ] Quantize the model with the 8-bit dynamic fixed-point technique.
- [ ] Compare correct adapted quantized and floating-point models against the original floating-point baseline.
- [ ] Write a corrective review article on [Yan et al. (2018)](https://arxiv.org/abs/1801.07455) in NeurIPS, CVPR (origin of ST-GCN), or ICCV.
- [ ] Add ST-GCN eloborate explanatory document to the repository (or link to the preprint article). Also clarify why RT ST-GCN can be trained as a Batch model and later just copy the learned parameters over.
- [ ] Do design space exploration of the network parameters on the adapted network for software-hardware co-design of an action segmentation hardware accelerator.
- [ ] Do transfer learning for freezing-of-gait (FOG) prediction.

> **Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition**, Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018. [[Arxiv Preprint]](https://arxiv.org/abs/1801.07455)

## Directory Tree
```
root/
├── .vscode/
│   └── launch.json
├── config/
│   ├── kinetics/
│   │   ├── original_local.json
│   │   ├── original_vsc.json
│   │   ├── realtime_local.json
│   │   └── realtime_vsc.json
│   └── pku-mmd/
│       ├── original_local.json
│       ├── original_vsc.json
│       ├── adapted_local.json
│       ├── adapted_vsc.json
│       ├── realtime_local.json
│       └── realtime_vsc.json
├── data/
│   ├── kinetics/
│   │   ...
│   │   └── actions.txt
│   ├── ntu_rgb_d/
│   │   ├── xsub/
│   │   ├── xview/
│   │   └── actions.txt
│   ├── pku-mmd/
│   │   ├── train/
│   │   ├── val/
│   │   ├── train_xsubject/
│   │   ├── val_xsubject/
│   │   └── actions.txt
│   └── skeletons/
│       ├── openpose.json
│       ├── ntu-rgb+d.json
│       └── pku-mmd.json
├── data_prep/
│   ├── dataset.py
│   └── label_eval.py
├── models/
│   ├── original/
│   │   └── st_gcn.py
│   ├── adapted/
│   │   └── st_gcn.py
│   ├── proposed/
│   │   ├── st_gcn.py
│   │   └── test_stg_gcn.py
│   └── utils/
│       ├── graph.py
│       ├── test_graph.py
│       └── tgcn.py
├── pretrained_models/
│   ├── kinetics/
│   ├── pku-mmd/
│   ├── pku-mmd-xsubject/
│   └── train-validation-curve.ods
├── local/
│   └── st_gcn.sh
├── vsc/
│   ├── experiment_gamma_size.sh
│   ├── st_gcn_gpu_bigmem.pbs
│   ├── st_gcn_gpu_debug.pbs
│   ├── st_gcn_gpu_train.pbs
│   ├── st_gcn_gpu.pbs
│   └── vsc_cheatsheet.md
├── tools/
│   ├── get_models.sh
│   └── get_data.sh
├── .gitignore
├── main.py
├── processor.py
├── st_gcn_parser.py
├── README.md
├── ISSUE_TEMPLATE.md
└── LICENSE
```
### Data Structure
Data and pretrained models directories are not tracked by the repository and must be downloaded from the source. Refer to the [Data section](#data).

Datasets provide data as a 5D tensor in the format (N-batch, C-channels, L-length, V-nodes, M-skeletons), with labels as a 1D tensor (N-batch): meaning the datasets are limited to multiple skeletons in the scene either performing the same action or being involved in a mutual activity (salsa, boxing, etc.). In a real application, each skeleton in the scene may be independent from others requiring own prediction and label. Moreover, a skeleton may perform multiple different actions while in the always-on scene: action segmentation is done on a frame-by-frame basis, rather than on the entire video capture (training data guarantees only 1 qualifying action in each capture, but requires broadcasting of labels across time and skeletons for frame-by-frame model training, to be applicable to realtime always-on classifier).

* Kinetics dataset, 400 action classes, dimensions:
  * Train - (240436, 3, 300, 18, 2)
  * Validation - (19796, 3, 300, 18, 2)

* NTU-RGB-D, 60 action classes, cross-view dataset dimensions:
  * Train - (37646, 3, 300, 25, 2)
  * Validation - (18932, 3, 300, 25, 2)

* NTU-RGB-D, 60 action classes, cross-subject dataset dimensions:
  * Train - (40091, 3, 300, 25, 2)
  * Validation - (16487, 3, 300, 25, 2)

* PKU-MMD, 51 action classes, cross-view dataset dimensions (each trial may differ in duration):
  * Train - (671, 3, ..., 25, 1)
  * Validation - (338, 3, ..., 25, 1)

* PKU-MMD, 51 action classes, cross-subject dataset dimensions (each trial may differ in duration):
  * Train - (775, 3, ..., 25, 1)
  * Validation - (234, 3, ..., 25, 1)

### Config Structure
Config files configure the execution script, model architecture, optimizer settings, training state, etc. This provides separation of concern and clean abstraction from source code for the user to prototype the model on various use cases and configuration by simply editing or providing a new JSON file to the execution script.

## Installation
### Environment
Local environment uses Conda for ease and convenience. 

High Performance Computing for heavy-duty training and testing is done at Vlaams Supercomputer Centrum [(VSC)](https://www.vscentrum.be/), a Linux environment supercomputer for industry and academia in Flanders (Belgium).

#### **Local**
Create a Conda environment with all the dependencies, and clone the repository.
```shell
conda create -n rt-st-gcn --file requirements.txt
conda activate rt-st-gcn
git clone https://github.com/maximyudayev/Realtime-ST-GCN.git
```

#### **Vlaams Supercomputer Centrum (VSC)**
Build latest CUDA-enabled PyTorch with target processor toolchain or ignore this step and use scripts as-is to load VSC-provided prebuilt CUDA PyTorch 1.0.1.
> **VSC PyTorch Build Repository** [[GitHub]](https://github.com/maximyudayev/VSC-PyTorch-Build)

### Data
**Kinetics-skeleton** and **NTU RGB+D** data is preprocessed by Yan et al. (2018), while **PKU-MMD (Phase 2)** data is by Filtjens et al. (2022). Both can be obtained for reproducible apples-to-apples benchmarking of the models. The custom `Dataset` classes of this project abstract away the structure of the processed data to provide a clean uniform interface between the `DataLoader` and the `Module` with similar shape tensors.

The datasets can be downloaded by running the script below, which will download all datasets. Alternatively, only names of the desired datasets can be passed to the script.
```shell
./tools/get_data.sh
```
OR:
```shell
./tools/get_data.sh 'pku-mmd' 'kinetics'
```
You can also download the **Kinetics-skeleton** and **NTU RGB+D**, and **PKU-MMD (Phase 2)** datasets manually from Yan's [Google Drive](https://drive.google.com/open?id=103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb) and Filtjens's [Github](https://github.com/BenjaminFiltjens/MS-GCN), respectively and extract them into `./data` (local environment) or `$VSC_SCRATCH/rt_st_gcn/data` (VSC environment): high-bandwidth IO (access of the datasets) on VSC should be done from the Scratch partition (fast Infiniband interconnect), all else should be kept on the Data partition.

Otherwise, for processing raw data by yourself, or to port new dataset to the model in the format it expects, please refer to the [original author's guide](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md).

New datasets should match the data directory structure, provide job configuration and skeleton graph files, which are expected by the model and the automated scripts to setup and run without errors. Make sure to match these prerequisites and to pay attention to dataset-specific configurations (like a batch of 1 for **PKU-MMD**):
```
...
├── data/
│   ...
│   ├── new_dataset/                  (single, large, self-contained, split dataset file)
│   │   ├── actions.txt               (output classes, excl. background/null class)
│   │   ├── train_data.npy
│   │   ├── train_label.pkl
│   │   ├── val_data.npy
│   │   └── val_label.pkl
│   ...
│   ├── new_dataset_from_directory/   (directory of single trials, stored as separate files)
│   │   ├── actions.txt               (output classes, excl. background/null class)
│   │   ├── train/
│   │   │   ├── features/
│   │   │   │   ...
│   │   │   │   └── trial_N.npy       (trial names can be anything, but must match in `features` and `labels` folders)
│   │   │   └── labels/
│   │   │       ...
│   │   │       └── trial_N.csv
│   │   └── val/
│   │       ├── features/
│   │       │   ...
│   │       │   └── trial_N.npy
│   │       └── labels/
│   │           ...
│   │           └── trial_N.csv
│   ...
│   └── skeletons/
│       ...
│       └── new_skeleton.json         (skeleton graph description)
...
├── config/
│   ...
│   └── new_dataset/
│       └── config.json               (script configuration)
...
```

### Pretrained Model
We provided the pretrained model weights of our **ST-GCN** models. The model weights can be downloaded by running the script:
```shell
./tools/get_models.sh
```
You can also download the models manually from [Google Drive](https://www.youtube.com/watch?v=BBJa32lCaaY) and put them into `./pretrained_models` (local environment) or `$VSC_SCRATCH/rt_st_gcn/pretrained_models` (VSC environment), for the same reason as in the section above.

<!-- 
## Testing Pretrained Models

### Evaluation
Once datasets ready, we can start the evaluation.

To evaluate ST-GCN model pretrained on **Kinetcis-skeleton**, run
```
python main.py recognition -c config/st_gcn/kinetics-skeleton/test.yaml
```
For **cross-view** evaluation in **NTU RGB+D**, run
```
python main.py recognition -c config/st_gcn/ntu-xview/test.yaml
```
For **cross-subject** evaluation in **NTU RGB+D**, run
```
python main.py recognition -c config/st_gcn/ntu-xsub/test.yaml
``` 

Similary, the configuration file for testing baseline models can be found under the ```./config/baseline```.

To speed up evaluation by multi-gpu inference or modify batch size for reducing the memory cost, set ```--test_batch_size``` and ```--device``` like:
```
python main.py recognition -c <config file> --test_batch_size <batch size> --device <gpu0> <gpu1> ...
```

### Results
The expected **Top-1** **accuracy** of provided models are shown here:

| Model| Kinetics-<br>skeleton (%)|NTU RGB+D <br> Cross View (%) |NTU RGB+D <br> Cross Subject (%) |
| :------| :------: | :------: | :------: |
|Baseline[1]| 20.3    | 83.1     |  74.3    |
|**ST-GCN** (Ours)| **31.6**| **88.8** | **81.6** | 

[1] Kim, T. S., and Reiter, A. 2017. Interpretable 3d human action analysis with temporal convolutional networks. In BNMW CVPRW. 

## Training
To train a new ST-GCN model, run

```
python main.py recognition -c config/st_gcn/<dataset>/train.yaml [--work_dir <work folder>]
```
where the ```<dataset>``` must be ```ntu-xsub```, ```ntu-xview``` or ```kinetics-skeleton```, depending on the dataset you want to use.
The training results, including **model weights**, configurations and logging files, will be saved under the ```./work_dir``` by default or ```<work folder>``` if you appoint it.

You can modify the training parameters such as ```work_dir```, ```batch_size```, ```step```, ```base_lr``` and ```device``` in the command line or configuration files. The order of priority is:  command line > config file > default parameter. For more information, use ```main.py -h```.

Finally, custom model evaluation can be achieved by this command as we mentioned above:
```
python main.py recognition -c config/st_gcn/<dataset>/test.yaml --weights <path to model weights>
```--> 
## Model
[torchinfo](https://github.com/TylerYep/torchinfo) spits out the following model summary:

1. **Proposed Batch ST-GCN**
   1. \#parameters: 806'074
   2. \#MACs: 4.46 G (per 300 frame capture)

This checks out with manual counting:

1. **Proposed Batch ST-GCN**
   1. \#parameters: 806'074
   2. \#MACs: 14.65 M/frame -> 4.4 G (does not account for some extra multiplications with edge importance matrices and BN on residual branches of dimension-matching resnet blocks)

Models use `torch.utils.checkpoint.checkpoint` to trade compute for memory to increase the training batch size for long sequence data, which does not fit on GPUs otherwise. The checkpoints are placed at each ST-GCN layer.

## Citation
Please cite the following paper if you use this repository in your reseach.
```
@inproceedings{ ,
  title     = { },
  author    = { },
  booktitle = { },
  year      = { },
}
```

## Contact
For any questions, feel free to contact
```
Maxim Yudayev : maxim.yudayev@kuleuven.be
```

## Acknowledgements
The resources and services used in this work were provided by the VSC [(Flemish Supercomputer Center)](https://www.vscentrum.be/), funded by the Research Foundation - Flanders (FWO) and the Flemish Government.

The current ST-GCN architecture and the corresponding publication are based on and attempt to clarify and improve the **Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition**, Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018. [[Arxiv Preprint]](https://arxiv.org/abs/1801.07455). The preprocessed **Kinetics** and **NTU-RGB-D** datasets as consistent `.npy` files are provided by [Yan et. al's ST-GCN](https://github.com/yysijie/st-gcn). The preprocessed **PKU-MMD (Phase 2)** dataset as consistent `.npy` files is provided by [Filtjens et. al's MS-GCN](https://github.com/BenjaminFiltjens/MS-GCN). We thank the authors for publicly releasing their code and data.

<!-- # Multi-Stage Spatial-Temporal Convolutional Neural Network (MS-GCN)
This code implements the skeleton-based action segmentation MS-GCN model from [Automated freezing of gait assessment with
marker-based motion capture and multi-stage
spatial-temporal graph convolutional neural
networks](https://arxiv.org/abs/2103.15449) and [Skeleton-based action segmentation with multi-stage spatial-temporal graph convolutional neural networks](https://arxiv.org/abs/2202.01727), arXiv 2022 (in-review).

It was originally developed for freezing of gait (FOG) assessment on a [proprietary dataset](https://movementdisorders.onlinelibrary.wiley.com/doi/10.1002/mds.23327). Recently, we have also achieved high skeleton-based action segmentation performance on public datasets, e.g. [HuGaDB](https://arxiv.org/abs/1705.08506), [LARa version 1](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7436169/), [PKU-MMD v2](https://arxiv.org/abs/1703.07475), [TUG](https://www.nature.com/articles/s41597-020-00627-7).

## Requirements
Tested on Ubuntu 16.04 and Pytorch 1.10.1. Models were trained on a
[Nvidia Tesla K80](https://www.nvidia.com/en-gb/data-center/tesla-k80/).

The c3d data preparation script requires [Biomechanical-Toolkit](https://github.com/Biomechanical-ToolKit/BTKPython). For installation instructions, please refer to the following [issue](https://github.com/Biomechanical-ToolKit/BTKPython/issues/2).

## Datasets
* LARa: https://zenodo.org/record/3862782#.YizNT3pKjZs
* PKU-MMD: https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html
* HuGaDB: https://github.com/romanchereshnev/HuGaDB
* TUG: https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/7VF22X
* FOG: not public

## Content
* `data_prep/` -- Data preparation scripts.
* `main.py` -- Main script. I suggest working with this interactively with an IDE. Please provide the dataset and train/predict arguments, e.g. `--dataset=fog_example --action=train`.
* `batch_gen.py` -- Batch loader.
* `label_eval.py` -- Compute metrics and save prediction results.
* `model.py` -- train/predict script.
* `models/` -- Location for saving the trained models.
* `models/ms_gcn.py` -- The MS-GCN model.
* `models/net_utils/` -- Scripts to partition the graph for the various datasets. For more information about the partitioning, please refer to the section [Graph representations](https://arxiv.org/abs/2202.01727). For more information about spatial-temporal graphs, please refer to [ST-GCN](https://arxiv.org/pdf/1801.07455.pdf).
* `data/` -- Location for the processed datasets. For more information, please refer to the 'FOG' example.
* `data/signals.` -- Scripts for computing the feature representations. Used for datasets that provided spatial features per joint, e.g. FOG, TUG, and PKU-MMD v2. For more information, please refer to the section [Graph representations](https://arxiv.org/abs/2202.01727).
* `results/` -- Location for saving the results.

## Data
After processing the dataset (scripts are dataset specific), each processed dataset should be placed in the ``data`` folder. We provide an example for a motion capture dataset that is in [c3d](https://www.c3d.org/) format. For this particular example, we extract 9 joints in 3D:
* `data_prep/read_frame.py` -- Import the joints and action labels from the c3d and save both in a separate csv.
* `data_prep/gen_data/` -- Import the csv, construct the input, and save to npy for training. For more information about the input and label shape, please refer to the section [Problem statement](https://arxiv.org/abs/2202.01727).

Please refer to the example in `data/example/` for more information on how to structure the files for training/prediction.

## Pre-trained models
Pre-trained models are provided for HuGaDB, PKU-MMD, and LARa. To reproduce the results from the paper:
* The dataset should be downloaded from their respective repository.
* See the "Data" section for more information on how to prepare the datasets.
* Place the pre-trained models in ``models/``, e.g. ``models/hugadb``.
* Ensure that the correct graph representation is chosen in ``ms_gcn``.
* Comment out ``features = get_features(features)`` in model (only for lara and hugadb).
* Specify the correct sampling rate, e.g. downsampling factor of 4 for lara.
* Run main to generate the per-sample predictions with proper arguments, e.g. ``--dataset=hugadb`` ``--action=predict``.
* Run label_eval with proper arguments, e.g. ``--dataset=hugadb``.

## Acknowledgements
The MS-GCN model and code are heavily based on [ST-GCN](https://github.com/yysijie/st-gcn) and [MS-TCN](https://github.com/yabufarha/ms-tcn). We thank the authors for publicly releasing their code.

## License
[MIT](https://choosealicense.com/licenses/mit/) -->
