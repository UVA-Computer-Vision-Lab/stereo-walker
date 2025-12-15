# StereoWalker: Empowering Dynamic Urban Navigation with Stereo and Mid-Level Vision


This repository contains official implementation of the code for the paper [Empowering dynamic urban navigation with stereo and mid-level vision](https://arxiv.org/abs/2512.10956).

Authors: [Wentao Zhou](https://smirkkkk.github.io/), [Xuweiyi Chen](https://xuweiyichen.github.io/), [Vignesh Rajagopal](https://cral-uva.github.io/people/vignesh_rajagopal/index.html), [Jeffrey Chen
](https://jeffchen888.github.io/), [Rohan Chandra](https://cral-uva.github.io/people/rohan_chandra/index.html), [Zezhou Cheng](https://sites.google.com/site/zezhoucheng/)

If you find this code useful, please consider citing:

```
@article{zhou2025empowering,
  title={Empowering Dynamic Urban Navigation with Stereo and Mid-Level Vision},
  author={Zhou, Wentao and Chen, Xuweiyi and Rajagopal, Vignesh and Chen, Jeffrey and Chandra, Rohan and Cheng, Zezhou},
  journal={arXiv preprint arXiv:2512.10956},
  year={2025}
}
```

## Installation

The project is tested with Python 3.11, PyTorch 2.5.0, and CUDA 12.1. Install dependencies with:

```bash
conda env create -f environment.yml
conda activate stereowalker
```

## Inference

Pretrained StereoWalker weights: [Download here](https://github.com/UVA-Computer-Vision-Lab/stereo-walker/releases/tag/v0)

```bash
python test.py --config config/teleop_eval.yaml --checkpoint [path to ckpt]
```

## Acknowledgments

We greatly appreciate CityWalker for open-source its code.
