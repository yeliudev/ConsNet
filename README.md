# ConsNet

[![DOI](https://badgen.net/badge/DOI/10.1145%2F3394171.3413600/blue?cache=300)](https://doi.org/10.1145/3394171.3413600)
[![arXiv](https://badgen.net/badge/arXiv/2008.06254/red?cache=300)](https://arxiv.org/abs/2008.06254)
[![PyPI](https://badgen.net/pypi/v/consnet?label=PyPI&cache=300)](https://pypi.org/project/consnet)
[![License](https://badgen.net/github/license/yeliudev/ConsNet?label=License&color=cyan&cache=300)](https://github.com/yeliudev/ConsNet/blob/main/LICENSE)

This repository maintains the official implementation of the paper **ConsNet: Learning Consistency Graph for Zero‐Shot Human‐Object Interaction Detection** by [Ye Liu](https://yeliu.me/), [Junsong Yuan](https://cse.buffalo.edu/~jsyuan/) and [Chang Wen Chen](https://www4.comp.polyu.edu.hk/~chencw/), which has been accepted by [ACM Multimedia 2020](https://2020.acmmm.org/).

<p align="center"><img src="https://raw.githubusercontent.com/yeliudev/ConsNet/main/.github/model.svg"></p>

## Installation

The ConsNet package could be installed directly from PyPI or manually from source for different uses. Please refer to the following environmental settings that we use.

- CUDA 10.2 Update 2
- CUDNN 8.0.5.39
- Python 3.9.2
- PyTorch 1.8.1
- [MMDetection](https://github.com/open-mmlab/mmdetection) 2.11.0
- [AllenNLP](https://github.com/allenai/allennlp) 2.2.0
- [NNCore](https://github.com/yeliudev/nncore) 0.2.4

### Install from PyPI

You may install ConsNet from PyPI and import it in your own project as a Python package. This library implements several useful functionalities including [Pair IoU](https://consnet.readthedocs.io/en/latest/consnet.api.bbox.html#consnet.api.bbox.pair_iou), [Pair NMS](https://consnet.readthedocs.io/en/latest/consnet.api.bbox.html#consnet.api.bbox.pair_nms) and [unified APIs for HICO-DET dataset](https://consnet.readthedocs.io/en/latest/consnet.api.data.html).

Simply run the following command to install the latest version of ConsNet.

```
pip install consnet
```

For more details about `consnet.api`, please refer to our [documentation](https://consnet.readthedocs.io/).

### Install from source

By installing ConsNet from source, you may access the full capabilities of this project, including pooling object features, constructing the consistency graph and benchmarking the ConsNet model.

1. Clone the repository from GitHub.

```
git clone https://github.com/yeliudev/ConsNet.git
cd ConsNet
```

2. Install full dependencies and the package.

```
pip install -e .[full]
```

## Getting Started

We pre-extract the visual features of all the humans and objects in the dataset and save them for training as well as testing. These features are also used to construct the consistency graph. Please refer to our paper for more details about feature extraction and data sampling.

### Build dataset and construct the consistency graph

1. Download the checkpoints of object detector and ELMo.

```shell
# Object detector checkpoints
wget https://dl.catcatdev.com/consnet/faster_rcnn_r50_fpn_3x_coco-26df6f6b.pth
wget https://dl.catcatdev.com/consnet/faster_rcnn_r50_fpn_20e_hico_det-77b91312.pth

# ELMo options and weights
wget https://dl.catcatdev.com/consnet/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
wget https://dl.catcatdev.com/consnet/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
```

2. Download [HICO-DET](http://www-personal.umich.edu/~ywchao/hico/) dataset and prepare the files in the following structure.

```
ConsNet
├── configs
├── consnet
├── tools
├── checkpoints
│   ├── faster_rcnn_r50_fpn_3x_coco-26df6f6b.pth
│   ├── faster_rcnn_r50_fpn_20e_hico_det-77b91312.pth
│   ├── elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
│   └── elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
├── data
│   └── hico_20160224_det
│       ├── anno_bbox.mat
│       └── images
│           ├── train2015
│           └── test2015
├── README.md
├── setup.py
└── ···
```

3. Convert the annotations to COCO format. The results will be saved in `data/hico_det/annotations`.

```
python tools/convert_anno.py
```

4. Build dataset and construct the consistency graph. The results will be saved in `data/hico_det`.

```
python tools/build_dataset.py --checkpoint <path-to-checkpoint>
```

### Train a model

Run the following command to train a model using specified configs.

```
python tools/launch.py --config <path-to-config>
```

### Test a model and evaluate results

Run the following command to test a model and evaluate results.

```
python tools/launch.py --config <path-to-config> --checkpoint <path-to-checkpoint> --eval
```

## Model Zoo

We provide multiple HICO-DET pre-trained models here. All the models are trained using a single NVIDIA Tesla V100-SXM2 GPU and are evaluated under the `default` metric of HICO-DET dataset.

<table>
  <tr>
    <th rowspan="2">Detector</th>
    <th rowspan="2">Model</th>
    <th rowspan="2">Type</th>
    <th colspan="5">Performance (mAP)</th>
    <th rowspan="2">Download</th>
  </tr>
  <tr>
    <th>Full</th>
    <th>Rare</th>
    <th>Non-Rare</th>
    <th>Seen</th>
    <th>Unseen</th>
  </tr>
  <tr>
    <td align="center" rowspan="5">
      <a href="https://dl.catcatdev.com/consnet/faster_rcnn_r50_fpn_3x_coco-26df6f6b.pth">COCO</a>
    </td>
    <td align="center">
      <a href="https://github.com/yeliudev/ConsNet/blob/main/configs/consnet_uc_5e_hico_det.py">ConsNet</a>
    </td>
    <td align="center">UC</td>
    <td align="center">19.78</td>
    <td align="center">14.43</td>
    <td align="center">21.37</td>
    <td align="center">20.69</td>
    <td align="center">16.13</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/consnet/consnet_uc_5e_hico_det-3a355824.pth">model</a> |
      <a href="https://dl.catcatdev.com/consnet/consnet_uc_5e_hico_det.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/ConsNet/blob/main/configs/consnet_uo_5e_hico_det.py">ConsNet</a>
    </td>
    <td align="center">UO</td>
    <td align="center">20.71</td>
    <td align="center">16.81</td>
    <td align="center">21.87</td>
    <td align="center">20.99</td>
    <td align="center">19.27</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/consnet/consnet_uo_5e_hico_det-21652552.pth">model</a> |
      <a href="https://dl.catcatdev.com/consnet/consnet_uo_5e_hico_det.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/ConsNet/blob/main/configs/consnet_ua_5e_hico_det.py">ConsNet</a>
    </td>
    <td align="center">UA</td>
    <td align="center">19.04</td>
    <td align="center">14.54</td>
    <td align="center">20.38</td>
    <td align="center">20.02</td>
    <td align="center">14.12</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/consnet/consnet_ua_5e_hico_det-492bab60.pth">model</a> |
      <a href="https://dl.catcatdev.com/consnet/consnet_ua_5e_hico_det.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/ConsNet/blob/main/configs/consnet_gt_5e_hico_det.py">ConsNet</a>
    </td>
    <td align="center">GT</td>
    <td align="center">53.04</td>
    <td align="center">38.79</td>
    <td align="center">57.3</td>
    <td align="center">—</td>
    <td align="center">—</td>
    <td align="center" rowspan="2">
      <a href="https://dl.catcatdev.com/consnet/consnet_5e_hico_det-684a879d.pth">model</a> |
      <a href="https://dl.catcatdev.com/consnet/consnet_5e_hico_det.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/ConsNet/blob/main/configs/consnet_5e_hico_det.py">ConsNet</a>
    </td>
    <td align="center">—</td>
    <td align="center">22.15</td>
    <td align="center">17.55</td>
    <td align="center">23.52</td>
    <td align="center">—</td>
    <td align="center">—</td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://dl.catcatdev.com/consnet/faster_rcnn_r50_fpn_20e_hico_det-77b91312.pth">HICO-DET</a>
    </td>
    <td align="center">
      <a href="https://github.com/yeliudev/ConsNet/blob/main/configs/consnet_5e_hico_det.py">ConsNet-F</a>
    </td>
    <td align="center">—</td>
    <td align="center">25.94</td>
    <td align="center">19.35</td>
    <td align="center">27.91</td>
    <td align="center">—</td>
    <td align="center">—</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/consnet/consnet_f_5e_hico_det-44c8412c.pth">model</a> |
      <a href="https://dl.catcatdev.com/consnet/consnet_f_5e_hico_det.json">metrics</a>
    </td>
  </tr>
</table>

Note that: Type `UC`, `UO`, `UA` and `GT` represent unseen action-object combination, unseen object, unseen action and ground truth scenarios respectively.

## Customization

Thanks to the modulized implementation based on [NNCore](https://github.com/yeliudev/nncore), this project is highly customizable with a number of replaceable modules. You may play with the hyperparameters in `configs` or construct your own HOI detection pipeline by replacing the dataset, detector, embedder, etc. Please check the [documentation](https://nncore.readthedocs.io/) of NNCore for more details about customizing the engine and modules.

## Citation

If you find this project useful for your research, please kindly cite our paper.

```
@inproceedings{liu2020consnet,
  title={ConsNet: Learning Consistency Graph for Zero-Shot Human-Object Interaction Detection},
  author={Liu, Ye and Yuan, Junsong and Chen, Chang Wen},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={4235--4243},
  year={2020}
}
```
