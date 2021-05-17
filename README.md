# Seeking the Shape of Sound

An implement of the CVPR 2021 paper: [**Seeking the Shape of Sound: An Adaptive Framework for Learning Voice-Face Association**](https://arxiv.org/abs/2103.07293)
![image](fig_overview.jpg)

## Environments
* **Ubuntu** 16.04
* **CUDA** 10.2
* **Python** 3.7.3
* **Pytorch** 1.4.0

<!-- This code is implemented with Pytorch (tested on 1.4.0).  -->
See [requirement.txt](https://github.com/KID-7391/seeking-the-shape-of-sound/blob/main/requirement.txt).

## Data preparation
Download [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html), [VGGFace](https://drive.google.com/file/d/1qmxGwW5_lNQbTqwW81yPObJ-S-n3rpXp/view) and unzip them to ./data.

Limited by file size, only part of the query lists is included in `./data`. Other lists used in the article can be downloaded from [Google drive](https://drive.google.com/file/d/1uw1pPLmhk95cSVhyF8fJzaHssg12xLG-/view?usp=sharing) or [Baidu drive](https://pan.baidu.com/s/1cBYLt9aLP13psuW2jkN7NA) (passwd: rfri).

## Training
<!-- The training process consists of three steps: -->
<!-- 1. Train the model and update identity weights: -->
1. Download pretrained models for backbones into `./pretrained_models`.

Google drive:

[SE-ResNet-50](https://drive.google.com/file/d/1z3iTyfEvfLeLuMvpgyazj9rxeRpKmyua/view?usp=sharing)

[Thin-ResNet-34](https://drive.google.com/file/d/1aC_TpAIBKm2vaMWw8DKGhUsqgb8nf4aA/view?usp=sharing)

Baidu drive:

[SE-ResNet-50](https://pan.baidu.com/s/1HY1LH6pQfQA7-10-xIpmqA) (passwd: jy55)

[Thin-ResNet-34](https://pan.baidu.com/s/1_MOOzx6oXKe4ItCvbtI-2A) (passwd: tc6i)

2. Train the model and update identity weights:
```shell
python3 train.py config/train_reweight.yaml
```
3. Extract identity weights from saved model file:
```shell
python3 extract_id_weight.py config/extract_id_weight.yaml
```
The 
4. Retrain the final model:
```shell
python3 train.py config/train_main.yaml
```

The model and log are saved in `save/vox1_train/Voice2Face/main` by default.

## Evaluation
1. Download the pretrained model from [Google drive](https://drive.google.com/file/d/1ZCPMk_0kKz8YO37ciAVJRDnmnTCqNhoG/view?usp=sharing) or [Baidu drive](https://pan.baidu.com/s/1ugbyM1AwmMUxDl3apmgtFg) (passwd: 4vyf).
2. Modify configures in `config/train_main.yaml`: change `resume\_eval` to the path where the model is saved.
3. Run
```shell
python3 eval.py config/train_main.yaml
```

Expected results (%):
|  |  1:2 Matching (U) | 1:2 Matching (G) |  Verification (U) |  Verification (G) |  Retrieval |
| ------------- | ------------- | ------------- | ------------- |  ------------- | ------------- |
| Voice-to-Face | 87.2 | 77.7 | 87.2 | 77.5 | 5.5 |
| Face-to-Voice | 86.5 | 75.3 | 87.0 | 76.1 | 5.8 |

The results might slightly differ from the above due to random factors in the training process.

## References
If this code is helpful to you, please consider citing our paper:
```
@inproceedings{wen2021seeking,
  title={Seeking the shape of sound: An adaptive framework for learning voice-face association},
  author={Wen, Peisong and Xu, Qianqian and Jiang, Yangbangyan and Yang, Zhiyong and He, Yuan and Huang, Qingming},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```