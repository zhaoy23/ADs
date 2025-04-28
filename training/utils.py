from training.scheduler import GradualWarmupScheduler
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from training.contrastive_loss import NT_xentSimilarityLoss,CosineSimilaityLoss,euclideanDistanceLoss,PairwiseDistanceLoss



def getLoss(lossname,chunks=3):
    # choose loss
    if lossname == "NTXENT":
        return NT_xentSimilarityLoss
    elif lossname == "Cosine":
        return CosineSimilaityLoss(chunk=chunks)
    elif lossname == "Euclidean":
        return euclideanDistanceLoss(chunk=chunks)
    elif lossname == "PairWise":
        return PairwiseDistanceLoss(chunk=chunks)
    else:
        raise NotImplementedError()

def getOptimizer(model,optimizername,lr=1e-1,addschedular=True):
    # get optimizer
    if optimizername == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
        lr_decay_gamma = 0.1
    elif optimizername == 'adam':
        # optimizer = optim.Adam(model.parameters(), lr=lr, betas=(.9, .999), weight_decay=1e-6)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_decay_gamma = 0.3
    else:
        raise NotImplementedError()
    scheduler_warmup=None
    if addschedular:
        # define schdular and warmup schedular
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 70)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10.0, total_epoch=10,
                                                  after_scheduler=scheduler)

    return optimizer,scheduler_warmup

