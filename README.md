# Realtime ST-GCN
Implementation of [ST-GCN](https://arxiv.org/abs/1801.07455) to continual realtime processing and introduction of a lightweight RT-ST-GCN for realtime embedded devices, on continual multi-action sequences of skeleton data.

Classes are decoupled for modularity and readability, with separation of concern dictated by the user's JSON configuration files or CLI arguments.

Leverages [PyTorch's Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) and our novel training trick for long different-length sequences datasets. Collectively enabling:
* accelerated distributed training and testing across multiple GPUs, without GIL (**Distributed Data Parallel**);
* consuming datasets processing data entries of which otherwise exceeds available device memory (**our trick**);
* emulate learning with large batch sizes, where other methods were previously limited to equal-length data entries and were memory-bound for batch size (**our trick**);

<!-- > **Realtime ST-GCN: Adapting for Inference at the Edge**, Maxim Yudayev, Benjamin Filtjens and Josep Balasch, TNNLS 2023. [[Arxiv Preprint]](https://arxiv.org/abs/...) -->

## Contributions
1. Formalizes application of ST-GCN and its SotA derivatives to [continuous recognition](#continuous-recognition).
2. Establishes [benchmarks](#results) on key, relevant datasets for the original unaltered ST-GCN model to compare against, and for reliable reproduction of results.
3. Proposes a distributed [training method](#training-technique) for otherwise memory-exceeding long-sequence datasets of unequal trial durations, and enables use of any, previously impossible, desired effective batch size.
4. Proposes a [lightweight model](#rt-st-gcn) optimization targeted at constrained embedded devices.

## TODO
- [x] Implement ST-GCN correctly to the paper's spec (but in RT variant), using basic differentiable tensor operators and cutting out messy Modules combination (stacked GCN + TCN).
- [x] Validate the network (tensor dimension changes) manually by chaining PyTorch operators.
- [x] Write data preparation and loading backend for out-of-core big data files.
- [x] Write a script to leverage [KU Leuven HPC](https://www.vscentrum.be/) infrastructure for the PyTorch workflow in a simple automated process that originates on the local machine.
- [x] Add support for frame buffered realtime processing.
- [ ] Add support for FIFO latency variants.
- [ ] Add support for file type datasets (equal duration trials).
- [x] Quantize the model with the 8-bit dynamic fixed-point technique.
- [x] Compare quantized and floating-point models.
- [x] Adapt training for [Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) for efficient multi-GPU training.
- [ ] Adapt BN layers to our training trick to emulate learning on larger batches.
- [ ] Adapt ST-GCN for quantization to enable PTQ and QAT.
- [ ] Benchmark quantized models.
- [ ] Add fixed trials to the repo for benchmarking and visualization.
- [ ] Clear FIFOs of the RT-ST-GCN model after each trial in the [`benchmark`](#benchmarking) routine.
- [ ] Add ST-GCN eloborate explanatory document to the repository (or link to the preprint article).
- [ ] Clarify why RT-ST-GCN can be trained as a Batch model and later just copy the learned parameters over.
- [ ] Explain the training parallelization trick.
- [ ] Propose a guideline on model type selection.
- [x] Split the processing infrastructure code from the usecase code as a standalone repository.
- [ ] Extend the infrastructure to the [Distributed RPC Framework](https://pytorch.org/docs/stable/rpc.html) and [TorchRun](https://pytorch.org/docs/stable/elastic/run.html).

## Future Directions
- [ ] Turn processing infrastructure into a microservice endpoint with interface to benchmark external models on proprietary datasets.
- [ ] Write an article on accuracy improvement after the graph construction fix.
- [ ] Do design space exploration (NAS + Knowledge Distillation) of the network parameters on the adapted network for software-hardware co-design of a hardware accelerator.
- [ ] Compare transfer learning vs. from-scratch learning for the freezing-of-gait (FOG) usecase.
- [ ] RT-ST-GCN predictions refinement with additional SotA mechanisms (transformers, input-dependent attention, different adjacency matrix modelling, squeeze-excite networks, etc.).
- [ ] Expanded loss function that minimizes logits between different frames of the same class.
- [ ] Custom loss function closer related to the [segmental F1](https://arxiv.org/abs/1611.05267) evaluation metric to help improve learning process (e.g. incorporates confidence levels, action durations, temporal shifts and overlap with the ground truth, closeness between classes in latent space).

## Directory Tree
```
root/
├── .vscode/
│   └── launch.json
├── config/
│   ├── imu_fogit_ABCD/
│   │   ├── original_local.json
│   │   ├── original_vsc.json
│   │   ├── realtime_local.json
│   │   └── realtime_vsc.json
│   └── pku-mmd/
│       ├── original_local.json
│       ├── original_vsc.json
│       ├── realtime_local.json
│       └── realtime_vsc.json
├── data/
│   ├── imu_fogit_ABCD/
│   │   ├── train/
│   │   │   ├── features/
│   │   │   └── labels/
│   │   ├── val/
│   │   │   ├── features/
│   │   │   └── labels/
│   │   ├── actions.txt
│   │   └── split.txt
│   ├── pku-mmd/
│   │   ├── train/
│   │   │   ├── features/
│   │   │   └── labels/
│   │   ├── val/
│   │   │   ├── features/
│   │   │   └── labels/
│   │   └── actions.txt
│   └── skeletons/
│       ├── imu_fogit_ABCD.json
│       ├── openpose.json
│       ├── ntu-rgb+d.json
│       └── pku-mmd.json
├── data_prep/
│   ├── dataset.py
│   └── prep.py
├── models/
│   ├── original/
│   │   └── st_gcn.py
│   ├── proposed/
│   │   ├── st_gcn.py
│   │   ├── test_folding.py
│   │   └── test_st_gcn.py
│   └── utils/
│       ├── graph.py
│       ├── test_graph.py
│       └── tgcn.py
├── pretrained_models/
│   ├── imu_fogit_ABCD/
│   │   ├── original/
│   │   └── realtime/
│   ├── pku-mmdv1/
│   │   ├── original/
│   │   └── realtime/
│   ├── pku-mmdv2/
│   │   ├── original/
│   │   └── realtime/
│   └── train-validation-curve.ods
├── tools/
│   ├── get_data.sh
│   └── get_models.sh
├── vsc/
│   ├── experiment_original_kernel.sh
│   ├── experiment_realtime_kernel.sh
│   ├── st_gcn_gpu_debug_p100.slurm
│   ├── st_gcn_gpu_train_a100.slurm
│   ├── st_gcn_gpu_train_p100.slurm
│   ├── st_gcn_gpu_train_v100.slurm
│   ├── test_realtime.sh
│   └── vsc_cheatsheet.md
├── .gitignore
├── main.py
├── processor.py
├── metrics.py
├── st_gcn_parser.py
├── visualize.py
├── README.md
├── ISSUE_TEMPLATE.md
└── LICENSE
```
### Data Structure
The project supports file and directory type datasets. The custom `Dataset` classes of this project abstract away the structure of the processed data to provide a clean uniform interface between the `DataLoader` and the `Module` with similar shape tensors.

Data and pretrained models directories are not tracked by the repository and must be downloaded from the corresponding source. Refer to the [Data section](#data).

Prepared datasets must feed `processor.py` data as a 5D tensor in the format (N-batch, C-channels, L-length, V-nodes, M-skeletons), with labels as a 1D tensor (N-batch): meaning the datasets are limited to multiple skeletons in the scene either performing the same action or being involved in a mutual activity (salsa, boxing, etc.). In a real application, each skeleton in the scene may be independent from others requiring own prediction and label. Moreover, a skeleton may perform multiple different actions while in the always-on scene: action segmentation is done on a frame-by-frame basis, rather than on the entire video capture (training data guarantees only 1 qualifying action in each capture, but requires broadcasting of labels across time and skeletons for frame-by-frame model training, to be applicable to realtime always-on classifier).

Datasets loaded from source may have different dimension ordering. Outside of PKU-MMD and IMU-FOG-IT, it is the responsibility of the user to implement data preparation function, similar to `data_prep/prep.py` for the `processor.py` code to work properly.

* IMU-FOG-IT dataset, 8 action classes, dimensions:
  * Train - (264, 6, ..., 7, 1)
  * Validation - (112, 6, ..., 7, 1)

<br>

* PKU-MMDv2, 52 action classes, cross-view dataset dimensions (each trial may differ in duration):
  * Train - (671, 3, ..., 25, 1)
  * Validation - (338, 3, ..., 25, 1)

<br>

* PKU-MMDv2, 52 action classes, cross-subject dataset dimensions (each trial may differ in duration):
  * Train - (775, 3, ..., 25, 1)
  * Validation - (234, 3, ..., 25, 1)

<br>

* PKU-MMDv1, 52 action classes, cross-view dataset dimensions (each trial may differ in duration):
  * Train - (717, 3, ..., 25, 1)
  * Validation - (359, 3, ..., 25, 1)

<br>

* PKU-MMDv1, 52 action classes, cross-subject dataset dimensions (each trial may differ in duration):
  * Train - (944, 3, ..., 25, 1)
  * Validation - (132, 3, ..., 25, 1)

### Config Structure
Config JSON files configure the execution script, model architecture, optimizer settings, training state, etc. This provides separation of concern and a clean abstraction from source code for the user to prototype the model on various use cases and model configurations by simply editing or providing a new JSON file to the execution script.

#### Parameters
* `segment`: Allows user to specify the size of chunks to chop each trial into to fit within the (GPU) memory available on user's hardware. The `processor.py` manually accumulates gradients on these chunks to produce training effect identical as if processing of the entire trial could fit in user's available (GPU) memory.

  *Note: adapt this parameter by trial and error until no OOM error is thrown, to maximize use of your available hardware.*

* `batch_size`: Manual accumulation of gradients throughout a sequentially forward passing different-duration trials in a 'minibatch', in `processor.py`, allows to emulate learning behavior expected from increasing the batch size value for an otherwise impossible case of different-duration trials (PyTorch cannot stack tensors of different lengths).

  *Note: this manual accumulation hence also permits the use of any batch size, overcoming the limitation of memory in time-series data applications, where batch size is limited by the amount of temporal data*.     

* `demo`: List of trial indices to use for generating segmentation masks to visualize model predictions.

  *Note: indices must not exceed the number of trials available in the used split. `data_prep/dataset.py` sorts filenames before constructing a `Dataset` class, hence results are reproducible across various OS and filesystems.*

* `backup`: Backup directory path to copy the results over for persistent storage (e.g., `$VSC_DATA` partition where data is not automatically cleaned up unlike `$VSC_SCRATCH` high-bandwidth partition).

  *Note: optional for local environments.*

All other JSON configuration parameters are self-explanatory and can be consulted in `config/`.

*Note: user can freely add custom uniquely identifiable parameters to the JSON files and access the values in the code from the `kwargs` argument - it is not required to adjust the `ArgumentParser`.*

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

#### **HPC**
To speed-up training, in this project we used the high-performance computing infrastructure available to us by the [Vlaams Supercomputer Centrum (VSC)](https://www.vscentrum.be/).

Create a Conda environment identical to [Local setup](#local) or leverage optimized modules compiled with toolchains for the specific VSC hardware (Intel, FOSS): to do that, launch appropriate SLURM job scripts that load PyTorch using the `module` system instead of activating the Conda environments.

### Data
**PKU-MMD** data is provided by Liu et al. (2017). **FOG-IT** proprietary data is provided by Filtjens et al. (2022) and is not public. **PKU-MMDv2** used in the project was preprocessed and made compatible with the project by Filtjens et al. (2022). **PKU-MMDv1** and **FOG-IT** data was preprocessed by us using the respective function in `data_prep/prep.py`. Any new dataset can be used with the rest of the training infrastructure, but the burden of preprocessing function implementation in `data_prep/prep.py` remains on the user (final file or directory type dataset must yield 5D tensor entries compatible with the `processor.py`).  

The non-proprietary datasets can be downloaded by running the script below to fetch all or specific datasets, respectively.
```shell
./tools/get_data.sh
```
OR:
```shell
./tools/get_data.sh 'pku-mmd' '...'
```

You can also download the datasets manually and extract them into `./data` (local environment) or `$VSC_SCRATCH/rt_st_gcn/data` (VSC environment): high-bandwidth IO (access of the datasets) on VSC should be done from the Scratch partition (fast Infiniband interconnect), all else should be kept on the Data partition. **PKU-MMDv2** from Filtjens's [Github](https://github.com/BenjaminFiltjens/MS-GCN), **PKU-MMDv1** from Liu's [project page](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html).

New datasets should match the data directory structure, provide auxiliary files (job configuration, skeleton graph, and list of actions), which are expected by the model and the automated scripts to setup and run without errors. Make sure to match these prerequisites and to pay attention to dataset-specific configurations:
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

### Pretrained Models
The ST-GCN and RT-ST-GCN models trained by us can be downloaded by running the corresponding script, specifying the dataset and model type separated by an `_`:
```shell
./tools/get_models.sh 'pku-mmd_st-gcn' 'pku-mmd_rt-st-gcn' '...'
```

Currently available models:
* PKU-MMD
* FOG-IT

You can also download the models manually from [Google Drive](https://www.youtube.com/watch?v=BBJa32lCaaY) and put them into `./pretrained_models` (local environment) or `$VSC_SCRATCH/rt_st_gcn/pretrained_models` (VSC environment), for the same reason as in the [Data section](#data).

## Functionality
The `main.py` provides multiple functionalities from one entry-point. Elaborate description of each functionality can be obtained by `python main.py --help` and `python main.py foo --help`, where `foo` is `train`, `test` or `benchmark`.

CLI arguments must be provided before specifying the configuration file (or omitting to use the default one). CLI arguments, when provided, override the configurations in the (provided) JSON configuration file.

<!-- TODO: add diagram explaining working between different classes -->

### Training
#### PKU-MMD
#### FOG-IT
Original ST-GCN takes ~15 min/epoch (4xP100 GPUs).

RT-ST-GCN takes ~1 min/epoch (1xP100 GPU).

<!-- To train a new ST-GCN model, run

```
python main.py recognition -c config/st_gcn/<dataset>/train.yaml [--work_dir <work folder>]
```
where the ```<dataset>``` must be ```ntu-xsub```, ```ntu-xview``` or ```kinetics-skeleton```, depending on the dataset you want to use.
The training results, including **model weights**, configurations and logging files, will be saved under the ```./work_dir``` by default or ```<work folder>``` if you appoint it. -->

### Testing

### Benchmarking

## Results
<!-- 
### Results
The expected **Top-1** **accuracy** of provided models are shown here:

| Model| Kinetics-<br>skeleton (%)|NTU RGB+D <br> Cross View (%) |NTU RGB+D <br> Cross Subject (%) |
| :------| :------: | :------: | :------: |
|Baseline[1]| 20.3    | 83.1     |  74.3    |
|**ST-GCN** (Ours)| **31.6**| **88.8** | **81.6** | 

[1] Kim, T. S., and Reiter, A. 2017. Interpretable 3d human action analysis with temporal convolutional networks. In BNMW CVPRW. 
--> 

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

The current ST-GCN architecture and the corresponding publication are based on and attempt to clarify and extend the **Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition**, Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018. [[Arxiv Preprint]](https://arxiv.org/abs/1801.07455). The preprocessed **PKU-MMD (Phase 2)** dataset as consistent `.npy` files is provided by [Filtjens et. al's MS-GCN](https://github.com/BenjaminFiltjens/MS-GCN). We thank the authors for publicly releasing their code and data.
