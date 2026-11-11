[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/activating-more-pixels-in-image-super/image-super-resolution-on-set5-4x-upscaling)](https://paperswithcode.com/sota/image-super-resolution-on-set5-4x-upscaling?p=activating-more-pixels-in-image-super)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/activating-more-pixels-in-image-super/image-super-resolution-on-urban100-4x)](https://paperswithcode.com/sota/image-super-resolution-on-urban100-4x?p=activating-more-pixels-in-image-super)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/activating-more-pixels-in-image-super/image-super-resolution-on-set14-4x-upscaling)](https://paperswithcode.com/sota/image-super-resolution-on-set14-4x-upscaling?p=activating-more-pixels-in-image-super)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/activating-more-pixels-in-image-super/image-super-resolution-on-manga109-4x)](https://paperswithcode.com/sota/image-super-resolution-on-manga109-4x?p=activating-more-pixels-in-image-super)

# メモ（使い方）

## 注意: Real-ESRGAN とは別の Python 環境を使う
- HAT は `basicsr==1.3.4.9` 固定、Real-ESRGAN は `basicsr>=1.4.2` が必要。
  同じ環境に入れるとどちらかが壊れるので、HAT 専用の環境を作る。
- 古い basicsr は新しい Python では入らないことが多いので、Python 3.10 くらいを使う。
- 実行するコマンドはすべて HAT フォルダ（この README がある場所）で行う。

## 1. 環境を作る（初回だけ）
```
# まだ無ければ: git clone https://github.com/haru452/HAT.git ; cd HAT
py -3.10 -m venv .venv-hat
.venv-hat\Scripts\activate
# PyTorch を先に入れる。cu121 の部分は自分の GPU / CUDA に合わせる
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python setup.py develop
```
- 2回目からは `.venv-hat\Scripts\activate` だけでよい。
- `No module named 'torchvision.transforms.functional_tensor'` と出たら、
  basicsr の `data/degradations.py` の該当 import を
  `from torchvision.transforms.functional import rgb_to_grayscale` に書き換える。

## 2. データを用意する
| 置き場所 | 中身 |
|---|---|
| `datasets/GT` | 学習用の高解像度画像 |
| `datasets/LR` | 学習用の低解像度画像（下のスクリプトで作る） |
| `datasets/val/GT`, `datasets/val/LR` | 検証用（学習に使わない画像を 50〜100 枚ほど） |
| `experiments/pretrained_models/HAT-L_SRx2_ImageNet-pretrain.pth` | 学習済みの重み（下の Google Drive から） |

LR 画像の作り方（GT を 1/2 に縮小。GT と同じファイル名で出力）:
```
python options/make_lr.py
python options/make_lr.py --gt datasets/val/GT --lr datasets/val/LR
```
- 縦横が 2 の倍数でない画像は飛ばされる。`--crop-gt` を付けると GT の端を切ってそろえる（GT を上書きする）。
- GT 画像の短辺が 128 未満だと学習でエラーになる。

## 3. 学習する（ファインチューニング x2）
```
python hat/train.py -opt options/train/train_HAT-L_SRx2_finetune.yml
```
- 設定は `options/train/train_HAT-L_SRx2_finetune.yml`。データや重みのパスを変えるときはここを直す。
- 結果（ログ・重み）は `experiments/train_HAT-L_SRx2_finetune/` に保存される。
- GPU のメモリが足りないときは yml の `batch_size_per_gpu` を下げる。


# HAT [![Replicate](https://replicate.com/cjwbw/hat/badge)](https://replicate.com/cjwbw/hat)

### Activating More Pixels in Image Super-Resolution Transformer [[Paper Link]](https://arxiv.org/abs/2205.04437)
[Xiangyu Chen](https://chxy95.github.io/), [Xintao Wang](https://xinntao.github.io/), [Jiantao Zhou](https://www.fst.um.edu.mo/personal/jtzhou/), [Yu Qiao](https://scholar.google.com.hk/citations?user=gFtI-8QAAAAJ) and [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=zh-CN)

### HAT: Hybrid Attention Transformer for Image Restoration [[Paper Link]](https://arxiv.org/abs/2309.05239)
[Xiangyu Chen](https://chxy95.github.io/), [Xintao Wang](https://xinntao.github.io/), [Wenlong Zhang](https://wenlongzhang0517.github.io/), [Xiangtao Kong](https://xiangtaokong.github.io/), [Jiantao Zhou](https://www.fst.um.edu.mo/personal/jtzhou/), [Yu Qiao](https://scholar.google.com.hk/citations?user=gFtI-8QAAAAJ) and [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=zh-CN)

## Updates
- ✅ 2022-05-09: Release the first version of the paper at Arxiv.
- ✅ 2022-05-20: Release the codes, models and results of HAT.
- ✅ 2022-08-29: Add a Replicate demo for SRx4.
- ✅ 2022-09-25: Add the tile mode for inference with limited GPU memory.
- ✅ 2022-11-24: Upload a GAN-based HAT model for **Real-World SR** (Real_HAT_GAN_SRx4.pth). 
- ✅ 2023-03-19: Update paper to CVPR version. Small HAT models are added.
- ✅ 2023-04-05: Upload the HAT-S codes, models and results. 
- ✅ 2023-08-01: Upload another GAN model for sharper results (Real_HAT_GAN_SRx4_sharper.pth). 
- ✅ 2023-08-01: Upload the training configs for the **Real-World GAN-based model**.
- ✅ 2023-09-11: Release the extended version of the paper at [Arxiv](https://arxiv.org/abs/2309.05239).
- **(To do)** Add the tile mode for Replicate demo. 
- **(To do)** Update the Replicate demo for Real-World SR. 
- **(To do)** Add HAT models for Multiple Image Restoration tasks. 

## Overview
<img src="https://raw.githubusercontent.com/chxy95/HAT/master/figures/Performance_comparison.png" width="600"/>

**Benchmark results on SRx4 without ImageNet pretraining. Mulit-Adds are calculated for a 64x64 input.**
| Model | Params(M) | Multi-Adds(G) | Set5 | Set14 | BSD100 | Urban100 | Manga109 |
|-------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| [SwinIR](https://github.com/JingyunLiang/SwinIR) |   11.9    | 53.6 | 32.92 | 29.09 | 27.92 | 27.45 | 32.03 |
| HAT-S |   9.6    | 54.9 | 32.92 | 29.15 | 27.97 | 27.87 | 32.35 |
| HAT |   20.8    | 102.4 | 33.04 | 29.23 | 28.00 | 27.97 | 32.48 |

## Real-World SR Results
**Note that:**
- The default settings in the training configs (almost the same as Real-ESRGAN) are for training **Real_HAT_GAN_SRx4_sharper**.
- **Real_HAT_GAN_SRx4** is trained using similar settings without USM the ground truth.
- **Real_HAT_GAN_SRx4** would have better fidelity.
- **Real_HAT_GAN_SRx4_sharper** would have better perceptual quality.

**Results produced by** Real_HAT_GAN_SRx4_sharper.pth.

<img src="https://raw.githubusercontent.com/chxy95/HAT/master/figures/Visual_Results.png" width="800"/>

**Comparison with the state-of-the-art Real-SR methods.**

<img src="https://raw.githubusercontent.com/chxy95/HAT/master/figures/Comparison.png" width="800"/>

## Citations
#### BibTeX

    @InProceedings{chen2023activating,
        author    = {Chen, Xiangyu and Wang, Xintao and Zhou, Jiantao and Qiao, Yu and Dong, Chao},
        title     = {Activating More Pixels in Image Super-Resolution Transformer},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2023},
        pages     = {22367-22377}
    }

    @article{chen2023hat,
      title={HAT: Hybrid Attention Transformer for Image Restoration},
      author={Chen, Xiangyu and Wang, Xintao and Zhang, Wenlong and Kong, Xiangtao and Qiao, Yu and Zhou, Jiantao and Dong, Chao},
      journal={arXiv preprint arXiv:2309.05239},
      year={2023}
    }

## Environment
- [PyTorch >= 1.7](https://pytorch.org/) **(Recommend **NOT** using torch 1.8!!! It would cause abnormal performance.)**
- [BasicSR == 1.3.4.9](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 
### Installation
Install Pytorch first.
Then,
```
pip install -r requirements.txt
python setup.py develop
```

## How To Test

Without implementing the codes, [chaiNNer](https://github.com/chaiNNer-org/chaiNNer) is a nice tool to run our models.

Otherwise, 
- Refer to `./options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.  
- The pretrained models are available at
[Google Drive](https://drive.google.com/drive/folders/1HpmReFfoUqUbnAOQ7rvOeNU3uf_m69w0?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1u2r4Lc2_EEeQqra2-w85Xg) (access code: qyrl).  
- Then run the following codes (taking `HAT_SRx4_ImageNet-pretrain.pth` as an example):
```
python hat/test.py -opt options/test/HAT_SRx4_ImageNet-pretrain.yml
```
The testing results will be saved in the `./results` folder.  

- Refer to `./options/test/HAT_SRx4_ImageNet-LR.yml` for **inference** without the ground truth image.

**Note that the tile mode is also provided for limited GPU memory when testing. You can modify the specific settings of the tile mode in your custom testing option by referring to `./options/test/HAT_tile_example.yml`.**

## How To Train
- Refer to `./options/train` for the configuration file of the model to train.
- Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md). ImageNet dataset can be downloaded at the [official website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
- The training command is like
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 hat/train.py -opt options/train/train_HAT_SRx2_from_scratch.yml --launcher pytorch
```
- Note that the default batch size per gpu is 4, which will cost about 20G memory for each GPU.  

The training logs and weights will be saved in the `./experiments` folder.

## Results
The inference results on benchmark datasets are available at
[Google Drive](https://drive.google.com/drive/folders/1t2RdesqRVN7L6vCptneNRcpwZAo-Ub3L?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1CQtLpty-KyZuqcSznHT_Zw) (access code: 63p5).


## Contact
If you have any question, please email chxy95@gmail.com or join in the [Wechat group of BasicSR](https://github.com/XPixelGroup/BasicSR#-contact) to discuss with the authors.
