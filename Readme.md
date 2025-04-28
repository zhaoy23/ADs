# ADs: Active Data-selection for Data Quality Assurance in Data-sharing over Advanced Manufacturing Systems

<hr>

Implementation of our paper, [Active Data-sharing for Data Quality Assurance in Advanced Manufacturing Systems](http://arxiv.org/abs/2404.00572).

## 1. Abstract
Machine learning (ML) methods are widely used in manufacturing applications, which usually require a large amount of training data. However, data collection needs extensive costs and time investments in the manufacturing system, and data scarcity commonly exists. With the development of the industrial internet of things (IIoT), data-sharing is widely enabled among multiple machines with similar functionality to augment the dataset for building ML models. Despite the machines being designed similarly, the distribution mismatch inevitably exists in their data due to different working conditions, process parameters, measurement noise, etc. However, the effective application of ML methods is built upon the assumption that the training and testing data are sampled from the same distribution. 

In this work, we propose an Active Data-selection (ADs) framework to ensure the quality of the shared data among multiple machines. A novel acquisition function is developed for data selection, which integrates the information measure for benefiting downstream tasks and the similarity score for data quality assurance. The simulation study and case study are designed and conducted to validate the effectiveness of the proposed ADs framework. In the simulation study, data samples from different distributions are generated. In the case study, we collected real-world *in-situ* monitoring data. The results demonstrated that our ADs framework could intelligently share data samples from the target distribution while eliminating the out-of-distribution data samples when training ML methods. Our proposed framework can improve the performance of ML models by selecting a high-quality subset of the full dataset compared with the classical AL method. In addition, the model performance can be comparable to or even better than training with the fully labeled data.

## 2. Citation
If you find our work useful in your research, please consider citing:
## 1. Requirements
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


## Train Data Augmentation Model

Check for Data Augmentation model Training 

`cd WKA && python WKATraining.py`

For simulation Augmentation Run

`cd WKA && python WKATrainingDist.py`

<hr>


## Training

## Dataset 
### Supervides
`CUDA_VISIBLE_DEVICES=id python Main.py --method Supervised --supervisedepochs 200`
### Random Pick S
`CUDA_VISIBLE_DEVICES=id python Main.py --method RandomS --supervisedepochs 200`
### Random Pick S+L
`CUDA_VISIBLE_DEVICES=id python Main.py --method RandomSL_S --supervisedepochs 200`
### ADS wthout CL
`CUDA_VISIBLE_DEVICES=id python ActiveLearninig.py --disableContrastive --initdata 5 --cycle 10 --totalpoints 800 --contrastiveEpoches 20 --uncertainityEpoches 200`
### ADS
`CUDA_VISIBLE_DEVICES=id python ActiveLearninig.py --initdata 5 --cycle 10 --totalpoints 800 --contrastiveEpoches 20 --uncertainityEpoches 200`

## Simulation Experiment 
### ADS wthout CL
`CUDA_VISIBLE_DEVICES=id python ActiveLearninig.py --DatasetName Random --DataIndex 3 --disableContrastive --initdata 20 --cycle 10 --totalpoints 400 --contrastiveEpoches 20 --uncertainityEpoches 200`
### ADS
`CUDA_VISIBLE_DEVICES=id python ActiveLearninig.py --DatasetName Random --DataIndex 3 --initdata 20 --cycle 10 --totalpoints 400 --contrastiveEpoches 20 --uncertainityEpoches 200`



<hr>

# All Arguments

| option | optionName               | Description                                                                                                              |
|--------|--------------------------|--------------------------------------------------------------------------------------------------------------------------|
| -id    | --initdata               | Use -id or --initdata to specify the initial data percentage.                                                            |
| -cy    | --cycle                  | Use -cy or --cycle to specify the number of cycles.                                                                      |
| -tp    | --totalpoints            | Use -tp or --totalpoints to specify the total number of points to select.                                                |
| -ce    | --contrastiveEpoches     | Use -ce to specify the number of contrastive epochs.                                                                     |
| -ue    | --uncertainityEpoches    | Use -ue to specify the number of uncertainty epochs.                                                                     |
| -se    | --supervisedepochs       | Use -se to specify the number of supervised epochs.                                                                      |
| -dn    | --DatasetName            | Use -dn to select either simulation data or real-world data.                                                             |
| -di    | --DataIndex              | Use -di to select the underlying distribution of the data points.                                                        |

 
