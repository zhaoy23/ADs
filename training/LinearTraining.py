import time,torch
from utils.utils import AverageMeter

def train(epoch, model, criterion, optimizer,  loader,device, logger=None,scheduler=None):
    # This function train the model for one epoch
    log_=print if logger is None else logger.log
    # define some average metters for get eveage1
    batch_time = AverageMeter()
    losses = dict()
    losses['LR'] = AverageMeter()
    losses['linear'] = AverageMeter()
    for n, (images, labels,index) in enumerate(loader):
        model.train()
        check = time.time()
        ### SimCLR loss ###
        batch_size = images.size(0)
        images = images.to(device)
        labels = labels.to(device).float()
        prediction = model(images,sigmoid=True)
        loss = criterion(torch.squeeze(prediction,dim=1), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']
        batch_time.update(time.time() - check)

        ### Log losses ###
        losses['linear'].update(loss,batch_size)
        losses['LR'].update(lr,batch_size)


    log_('[DONE] [Epoch %d] [Time %.3f] [LossLinear %f] ' %
         (epoch,batch_time.average,  losses['linear'].average))
    if logger is not None:
        logger.scalar_summary('train/Learning Rate', losses['LR'].average, epoch)
        logger.scalar_summary('train/Loss_Linear', losses['linear'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)
    return losses['linear'].average

