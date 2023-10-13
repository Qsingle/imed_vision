# Some codes for our experiments, you can use and change it by youself

## Suport Method

+ DeepLabV3: [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

+ DeepLabV3 Plus: [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611v1)

+ DANet: [Dual Attention Network for Scene Segmentation](https://arxiv.org/pdf/1809.02983.pdf)

+ EncNet: [Context Encoding for Semantic Segmentation](https://arxiv.org/pdf/1803.08904.pdf)

+ ESPNet V2: [ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network](https://arxiv.org/pdf/1811.11431.pdf)

+ UNet: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

+ PFSeg: [Patch-Free 3D Medical Image Segmentation Driven by Super-Resolution Technique and Self-Supervised Guidance](https://link.springer.com/chapter/10.1007/978-3-030-87193-2_13)

+ PSPNet: [Pyramid Scene Parsing Network](https://arxiv.org/pdf/1612.01105.pdf)

+ STDCNet: [Rethinking BiSeNet For Real-time Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Rethinking_BiSeNet_for_Real-Time_Semantic_Segmentation_CVPR_2021_paper.pdf)

+ SCS-Net:  [SCS-Net: A Scale and Context Sensitive Network for Retinal Vessel Segmentation](https://www.sciencedirect.com/science/article/pii/S1361841521000712#)

+ SA-UNet: [SA-UNet: Spatial Attention U-Net for Retinal Vessel Segmentation](https://arxiv.org/abs/2004.03696v3)

+ BiSeNetV2: [BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation](https://arxiv.org/pdf/2004.02147.pdf)

+ D2SF: [Rethinking Dual-Stream Super-Resolution Semantic Learning in Medical Image Segmentation](https://ieeexplore.ieee.org/document/10274145) **Official**

+ SuperVessel: [SuperVessel: Segmenting High-resolution Vessel from Low-resolution Retinal Image](https://arxiv.org/abs/2207.13882) **Official**

+ SS-MAF [Hard Exudate Segmentation Supplemented by Super-Resolution with Multi-scale Attention Fusion Module](https://arxiv.org/pdf/2211.09404.pdf) **Offficial**

+ [Learnable Ophthalmology SAM](https://arxiv.org/abs/2304.13425)

+ DPT: [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413)

  

## Usage

+ Clone or download this repository to your computer

  ```shell
  git clone https://github.com/Qsingle/imed_vision.git
  ```

  

+ Follow the samples at the `configs` directory to set the parameters used to train.

+ Run the training script (e.g. [vessel_segmentation_train.py](./vessel_segmentation_train.py)).

  ```shel
  python  vessel_segmentation_train.py --json_path ./configs/iostar_vessel_segmentation.json
  ```

  

## TODO

- [x] Update the structure for this repository
- [ ] Add the scripts for classification model train 



## Acknowledgment

[Segment-Anything](https://github.com/facebookresearch/segment-anything)

[PFSeg](https://github.com/Dootmaan/PFSeg)



## Cititions

```bibtex
@inproceedings{hu2022supervessel,
  title={Supervessel: Segmenting high-resolution vessel from low-resolution retinal image},
  author={Hu, Yan and Qiu, Zhongxi and Zeng, Dan and Jiang, Li and Lin, Chen and Liu, Jiang},
  booktitle={Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  pages={178--190},
  year={2022},
  organization={Springer Nature Switzerland Cham}
}

@inproceedings{zhang2022hard,
  title={Hard Exudate Segmentation Supplemented by Super-Resolution with Multi-scale Attention Fusion Module},
  author={Zhang, Jiayi and Chen, Xiaoshan and Qiu, Zhongxi and Yang, Mingming and Hu, Yan and Liu, Jiang},
  booktitle={2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={1375--1380},
  year={2022},
  organization={IEEE}
}

@article{qiu2023learnable,
  title={Learnable ophthalmology sam},
  author={Qiu, Zhongxi and Hu, Yan and Li, Heng and Liu, Jiang},
  journal={arXiv preprint arXiv:2304.13425},
  year={2023}
}

@article{qiu2023rethinking,
  title={Rethinking Dual-Stream Super-Resolution Semantic Learning in Medical Image Segmentation},
  author={Qiu, Zhongxi and Hu, Yan and Chen, Xiaoshan and Zeng, Dan and Hu, Qingyong and Liu, Jiang},
  journal={IEEE Transactions on Pattern Analysis \& Machine Intelligence},
  number={01},
  pages={1--14},
  year={2023},
  publisher={IEEE Computer Society}
}
```

