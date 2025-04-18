# APPNet（code for An Adaptive Post-Processing Network With Global-Local Aggregation For semantic segmentation）

## Introduction
  In this paper, we propose an adaptive post-processing network (APPNet) for semantic segmentation based on the predictions of current methods in the global image and local image patches. The key point of APPNet is the global-local aggregation module, which models the context between global
predictions and local predictions to generate accurate pixel-wise representation. Furthermore, we develop an adaptive points
replacement module to compensate for the lack of fine detail in global prediction and the overconfidence in local predictions.


<p align="center">
  <img src="https://github.com/zhu-gl-ux/APPNet/blob/master/image/pipeline.png" />
</p>



## Results
Our method can be readily integrated into existing segmentation methods (i.e., ConvNeXt, HRNet, ViT-Adapter) with little memory and without extra modification in current models. We empirically demonstrate our method brings performance improvements across diverse datasets (i.e., Cityscapes, ADE20K, PASCAL-Context, COCO-Stuff).

<p align="center">
<img src="https://github.com/zhu-gl-ux/APPNet/blob/master/image/results.png" width="600" />
</p>

## Visualization
<img src="https://github.com/zhu-gl-ux/APPNet/blob/master/image/cityscapes.png" />


## Acknowledgment
Our code is based on [MagNet](https://github.com/VinAIResearch/MagNet/tree/main)
## Reference
[1] Deep High-Resolution Representation Learning for Visual Recognition. Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao. Accepted by TPAMI. [download](https://arxiv.org/pdf/1908.07919.pdf) 

[2] Object-Contextual Representations for Semantic Segmentation. Yuhui Yuan, Xilin Chen, Jingdong Wang. Accepted by ECCV. [download](https://arxiv.org/pdf/1909.11065.pdf)

[3] A ConvNet for the 2020s. Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie. Accepted by CVPR. [download](https://arxiv.org/pdf/2201.03545.pdf)

[4] Vision Transformer Adapter for Dense Predictions. Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, Yu Qiao. Accepted by ICLR. [download](https://arxiv.org/pdf/2205.08534.pdf)
