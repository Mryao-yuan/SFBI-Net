# SFBI-Net

> This work presents at [TGRS2024](https://ieeexplore.ieee.org/document/10589371)

Recently, transformers have been widely explored in remote sensing image change detection and achieved remarkable performance. However, most existing transformer-based change detection methods overlook exploring the spatiotemporal relationships between bitemporal images at the features within the same layer, which is crucial for learning discriminative features to perceive changes. Moreover, no explicit spatial constraint has been imposed on the final fused bitemporal features, leading to reduced detection performance on small targets. To address these issues, we propose a spatial focused bitemporal interactive network (SFBI-Net) for remote sensing image change detection. Specifically, a bitemporal spatiotemporal interactive (BSI) module is proposed, which performs global interactions on bitemporal features at the same network layer and supplements local information to obtain spatiotemporal relationships of bitemporal features for discriminative representation. Furthermore, a spatial focus diversity loss (SFD-Loss) is developed to maximize bitemporal features in the spatial dimension and further enhance the feature representation of change areas, especially small target areas. The experimental results on challenging benchmark datasets demonstrate the superiority of our SFBI-Net. 

![](./SFBI-Net/image/Networks.png)

## Contents

- [SFBI-Net](#sfbi-net)
  - [Contents](#contents)
    - [Depencies](#depencies)
    - [Filetree](#filetree)
    - [Quick start](#quick-start)
      - [Installation](#installation)
      - [Train](#train)
      - [Test](#test)
    - [Qualitative Results](#qualitative-results)
      - [results on LEVIR-CD](#results-on-levir-cd)
      - [results on CLCD](#results-on-clcd)
      - [results on EGY](#results-on-egy)
    - [Copyright](#copyright)
    - [Thanks](#thanks)

### Depencies
1. Pytorch 2.0.0
2. Python 3.10.12
3. CUDA 11.7
4. Ubuntu 18.04

### Filetree

```
SFBI-Net
├─ README.md
├─ data_config.py
├─ datasets
│  ├─ CD_dataset.py
│  └─ data_utils.py
├─ eval.py
├─ image
│     ├─ Network.png
│     ├─ clcd.png
│     ├─ egy.png
│     └─ levir-cd.png
├─ misc
│  ├─ logger_tool.py
│  └─ metric_tool.py
├─ models
│  ├─ SFBI-Net.py
│  ├─ __init__.py
│  ├─ _utils.py
│  ├─ basic_model.py
│  ├─ evaluator.py
│  ├─ help_funcs.py
│  ├─ losses.py
│  ├─ networks.py
│  ├─ resnet.py
│  └─ trainer.py
├─ output
│  ├─ checkpoints
│  └─ vis
├─ requirements.txt
├─ script
│  ├─ eval_SFBI-Net.sh
│  └─ run_SFBI-Net.sh
├─ train.py
└─ utils.py

```

### Quick start

#### Installation

clone this repo:

```sh
git clone https://github.com/Mryao-yuan/SFBI-Net.git
cd SFBI-Net
```

#### Train

```sh
sh script/run_SFBI-Net.sh
```

#### Test

```sh
sh script/eval_SFBI-Net.sh
```

### Qualitative Results

#### results on [LEVIR-CD](https://www.mdpi.com/2072-4292/12/10/1662/pdf)
![](./SFBI-Net/image/LEVIR.png)

#### results on [CLCD](https://ieeexplore.ieee.org/abstract/document/10145434)
![](./SFBI-Net/image/CLCD.png)

#### results on [EGY](https://ieeexplore.ieee.org/iel7/4609443/4609444/09780164.pdf)
![](./SFBI-Net/image/EGY.png)

### Copyright

The project has been licensed by Apache-2.0. Please refer to for details. [LICENSE.txt](./LICENSE)

### Thanks

* [Pytorch-Grad-Cam](https://github.com/jacobgil/pytorch-grad-cam)
* [BIT](https://github.com/justchenhao/BIT_CD)
* [ChangeFormer](https://github.com/wgcban/ChangeFormer)

(Our SFBI-Net is implemented on the code provided in this repository)
