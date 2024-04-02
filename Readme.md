# ADs: Active Data-sharing for Data Quality Assurance in Advanced Manufacturing Systems
<hr>

Implementation of our paper, [Active Data-sharing for Data Quality Assurance in Advanced Manufacturing Systems](http://arxiv.org/abs/2404.00572).

## 1. Abstract
Machine learning (ML) methods are widely used in manufacturing applications, which usually require a large amount of training data. However, data collection needs extensive costs and time investments in the manufacturing system, and data scarcity commonly exists. With the development of the industrial internet of things (IIoT), data-sharing is widely enabled among multiple machines with similar functionality to augment the dataset for building ML models. Despite the machines being designed similarly, the distribution mismatch inevitably exists in their data due to different working conditions, process parameters, measurement noise, etc. However, the effective application of ML methods is built upon the assumption that the training and testing data are sampled from the same distribution. Thus, an intelligent data-sharing framework is needed to ensure the quality of the shared data such that only beneficial information is shared to improve the performance of ML methods. In this work, we propose an Active Data-sharing (ADs) framework to ensure the quality of the shared data among multiple machines. It is designed as a self-supervised learning framework by integrating the architecture of contrastive learning (CL) and active learning (AL). A novel acquisition function is developed for active learning by integrating the information measure for benefiting downstream tasks and the similarity score for data quality assurance. To validate the effectiveness of the proposed ADs framework, we collected real-world *in-situ* monitoring data from three 3D printers, two of which share identical specifications, while the other one is different. The results demonstrated that our ADs framework could intelligently share monitoring data between identical machines while eliminating the data points from the different machines when training ML methods. With a high-quality augmented dataset generated by our proposed framework, the ML methods can achieve a better performance of accuracy 95.78\% when utilizing 26\% labeled data. This represents an improvement of 1.41\% compared with benchmark methods which used 100\% labeled data.

## 2. Citation
If you find our work useful in your research, please consider citing:

```
@misc{zhao2024ads,
      title={ADs: Active Data-sharing for Data Quality Assurance in Advanced Manufacturing Systems}, 
      author={Yue Zhao and Yuxuan Li and Chenang Liu and Yinan Wang},
      year={2024},
      eprint={2404.00572},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## 3. Requirements
<hr>

### Environments

 Currently required following package

* Cuda==11.8.0
* python==3.10.0
* Tensorlfow == 2.14.0
* torch == 2.1.0
* scikit-learn==1.3.1
* pandas==2.1.1

<hr>

## 4. Data Augmentation Model

Check for Data Augmentation model Training 

`cd WKA && python WKATraining.py`

<hr>


## 5. Training

### Supervides
`CUDA_VISIBLE_DEVICES=id python Main.py --method Supervised --supervisedepochs 200`
### Random Pick S
`CUDA_VISIBLE_DEVICES=id python Main.py --method RandomS --supervisedepochs 200`
### Random Pick S+L
`CUDA_VISIBLE_DEVICES=id python Main.py --method RandomSL --supervisedepochs 200`
### ADS wthout CL
`CUDA_VISIBLE_DEVICES=id python Main.py --method ADSNonCL  --initdata 20 --cycle 10 --totalpoints 400 --contrastiveEpoches 20 --uncertainityEpoches 200`
### ADS
`CUDA_VISIBLE_DEVICES=id python Main.py --method ADS  --initdata 20 --cycle 10 --totalpoints 400 --contrastiveEpoches 20 --uncertainityEpoches 200`

<hr>

# 6. All Arguments

| option | optionName               | Description                                                                                                              |
|--------|--------------------------|--------------------------------------------------------------------------------------------------------------------------|
| -id    | --initdata               | you can use -id or --initdata for specify the inital data percentage                                                     |
| -cy    | --cycle                  | you can use -cpy or --cycle for specify  the number of cycle you want to run                                             |
| -tp    | --totalpoints            | you can use -tp for --totalpoints for specify the total number of points to select it will calculate the number of cycle |
| -ce    | --contrastiveEpoches     | you can also specify the contrastive epoches with -ce                                                                    |
| -ue    | --uncertainityEpoches    | you can also specify the uncertainity epoches with -ue                                                                   |
| -se    | --supervisedepochs    | you can also specify the Supervise epoches with -se for epoches for Random  and Validation                           |


