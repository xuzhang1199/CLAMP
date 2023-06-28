# CLAMP

By Xu Zhang, Wen Wang, Zhe Chen, Yufei Xu, Jing Zhang, and Dacheng Tao

This repository is an official implementation of CLAMP in the paper [CLAMP: Prompt-based Contrastive Learning for Connecting Language and Animal Pose](https://arxiv.org/abs/2206.11752), which is accepted to CVPR 2023.


## Main Results

Models can be downloaded from [Google Drive](https://drive.google.com/file/d/1ep5WExRnN51n1w-0mg7Lxv-1vJahcECM/view?usp=sharing)

## Installation

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.6
  
* PyTorch>=1.5.0, torchvision>=0.6.0 (following instructions [here](https://pytorch.org/))

* mmcv
    ```bash
    cd mmcv
    pip install -r requirements.txt
    pip install -v -e .
    ```

* mmpose
    ```bash
    cd ..
    pip install -r requirements.txt
    pip install -v -e .
    ```

## Usage

### Dataset preparation

Please download the dataset from [AP-10K](https://github.com/AlexTheBad/AP-10K).

### CLIP-pretrained models

Please download CLIP pretrained models from [CLIP](https://github.com/openai/CLIP).

### Training

#### Training CLAMP on AP-10K

```bash
bash tools/dist_train.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py 4 "0,1,2,3"
```

### Evaluation

You can get the pretrained model (the link is in "Main Results" session), then run following command to evaluate it on the validation set:

```bash
bash tools/dist_test.sh configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/CLAMP_ViTB_ap10k_256x256.py work_dirs/CLAMP_ViTB_ap10k_256x256/epoch_210.pth 4 "0,1,2,3"
```

## Acknowledgement 

This project is based on [mmpose](https://github.com/open-mmlab/mmpose), [AP-10K](https://github.com/AlexTheBad/AP-10K), [CLIP](https://github.com/openai/CLIP), and [DenseCLIP](https://github.com/raoyongming/DenseCLIP). Thanks for their wonderful works. See [LICENSE](./LICENSE) for more details. 


## Citing CLAMP
If you find CLAMP useful in your research, please consider citing:
```bibtex
@inproceedings{zhang2023clamp,
  title={CLAMP: Prompt-Based Contrastive Learning for Connecting Language and Animal Pose},
  author={Zhang, Xu and Wang, Wen and Chen, Zhe and Xu, Yufei and Zhang, Jing and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23272--23281},
  year={2023}
}
```
