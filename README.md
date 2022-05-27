# Realtime ST-GCN
Correct implementation of ST-GCN inline with the original paper's proposed formulae, and adapted to realtime processing.

## TODO
- [x] Implement ST-GCN correctly to the paper's spec (but in RT variant), using basic differentiable tensor operators and cutting out messy Modules combination (stacked GCN + TCN).
- [ ] Validate the network (tensor dimension changes) on a realtime task and on a batch task.
- [ ] Add ST-GCN eloborate explanatory document to the repository.
- [ ] Quantize the model with 8-bit dynamic fixed point technique.
- [ ] Compare correct adapted quantized and floating-point models against the original floating-point baseline.
- [ ] Do design space exploration of the network parameters on the adapted network for software-hardware co-design of an action segmentation hardware accelerator.

> **Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition**, Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018. [[Arxiv Preprint]](https://arxiv.org/abs/1801.07455)

## Installation
### Environment
Create a Conda environment with all the dependencies, and clone the repository.
```shell
conda create -n rt-st-gcn --file requirements.txt
conda activate rt-st-gcn
git clone https://github.com/maximyudayev/Realtime-ST-GCN.git
```

### Pretrained Model
We provided the pretrained model weithts of our **ST-GCN**. The model weights can be downloaded by running the script
```shell
./tools/get_models.sh
```
You can also download the models manually from [GoogleDrive](https://www.youtube.com/watch?v=BBJa32lCaaY) and put them into ```./models```.

<!-- 
## Demo
To visualize how ST-GCN exploit local correlation and local pattern, we compute the feature vector magnitude of each node in the final spatial temporal graph, and overlay them on the original video. **Openpose** should be ready for extracting human skeletons from videos. The skeleton based action recognition results is also shwon thereon.

You can use the following commands to run the demo.

```shell
# with offline pose estimation
python main.py demo_offline [--video ${PATH_TO_VIDEO}] [--openpose ${PATH_TO_OPENPOSE}]

# with realtime pose estimation
python main.py demo [--video ${PATH_TO_VIDEO}] [--openpose ${PATH_TO_OPENPOSE}]
```

Optional arguments:

- `PATH_TO_OPENPOSE`: It is required if the Openpose Python API is not in `PYTHONPATH`.
- `PATH_TO_VIDEO`: Filename of the input video.

The realtime demo also support to load video streams from camera source by
```
python main.py demo --video camera
```

Openpose Python API is required in the above demos.

## Data Preparation

We experimented on two skeleton-based action recognition datasts: **Kinetics-skeleton** and **NTU RGB+D**.
Before training and testing, for convenience of fast data loading,
the datasets should be converted to proper file structure. 
You can download the pre-processed data from 
[GoogleDrive](https://drive.google.com/open?id=103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb)
and extract files with
``` 
cd st-gcn
unzip <path to st-gcn-processed-data.zip>
```
Otherwise, for processing raw data by yourself,
please refer to below guidances.

#### Kinetics-skeleton
[Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) is a video-based dataset for action recognition which only provide raw video clips without skeleton data. Kinetics dataset include To obatin the joint locations, we first resized all videos to the resolution of 340x256 and converted the frame rate to 30 fps.  Then, we extracted skeletons from each frame in Kinetics by [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). The extracted skeleton data we called **Kinetics-skeleton**(7.5GB) can be directly downloaded from [GoogleDrive](https://drive.google.com/open?id=1SPQ6FmFsjGg3f59uCWfdUWI-5HJM_YhZ) or [BaiduYun](https://pan.baidu.com/s/1dwKG2TLvG-R1qeIiE4MjeA#list/path=%2FShare%2FAAAI18%2Fkinetics-skeleton&parentPath=%2FShare).

After uncompressing, rebuild the database by this command:
```
python tools/kinetics_gendata.py --data_path <path to kinetics-skeleton>
```

#### NTU RGB+D
NTU RGB+D can be downloaded from [their website](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp).
Only the **3D skeletons**(5.8GB) modality is required in our experiments. After that, this command should be used to build the database for training or evaluation:
```
python tools/ntu_gendata.py --data_path <path to nturgbd+d_skeletons>
```
where the ```<path to nturgbd+d_skeletons>``` points to the 3D skeletons modality of NTU RGB+D dataset you download.


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
