import os,time,torch,numpy as np
from training.utils import getLoss
import torch.utils.data as data
from utils.utils import save_checkpoint,Logger,AverageMeter, normalize
from sklearn.metrics import accuracy_score

def trainContrastive(epochs,cycle, model,  optimizer, scheduler,lossname, loader, logger=None,modelbasepath="logs/models/",device=torch.device("cpu"),savemodel=False):
    sim_lambda,dis_lambda = 1.0,1.0
    if logger is None:log_ = print
    else: log_ = logger.log
    print("Contrastive Class Counts", np.unique(torch.cat([l for _,_,_,_,l,_,_ in loader]), return_counts=True))
    lossmodule=getLoss(lossname,chunks=3)
    averageloss=[]
    for epoch in range(0, 101):
        logger.log_dirname(f"Epoch {epoch}")
        model.train()
        #  train model for epoch
        # define some average metters for get eveage1
        batch_time,data_time = AverageMeter(), AverageMeter()
        losses = {d:AverageMeter() for d in ["loss","linear"]}
        check = time.time()
        for n, (anchor,postive,negative,labels,sllabel,nalabel,index) in enumerate(loader):
            model.train()
            data_time.update(time.time() - check)
            check = time.time()
            ### SimCLR loss ###
            batch_size = anchor.size(0)
            combineData = torch.cat([anchor.to(device),postive.to(device),negative.to(device)])
            # training for Similar and dissimilar
            # get the simclr attributes
            _, outputs_aux = model(combineData, simclr=True, penultimate=True)
            simclr = normalize(outputs_aux['simclr'])  # normalize
            loss = lossmodule(simclr,sim_lambda=sim_lambda, dis_lambda=dis_lambda)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch - 1 + n / len(loader))
            batch_time.update(time.time() - check)
            ### Log losses ###
            losses['loss'].update(loss, batch_size)
            check = time.time()
        log_('[DONE] [Epoch %d] [Time %.3f] [Data %.3f] [LossC %f] ' %
             (epoch,batch_time.average, data_time.average,
              losses['loss'].average))
        averageloss.append(losses['loss'].average.detach().cpu().numpy())
        if savemodel and (epoch % 2 == 0 or epoch == epochs):
            save_states = model.state_dict()
            os.makedirs(modelbasepath, exist_ok=True)
            modelinfo="Contrastive_"+lossname+"_"+str(cycle)
            save_checkpoint(epoch, save_states, optimizer.state_dict(), modelbasepath, modelinfo)
            print("MODEL SAVED AT ", modelbasepath + 'last_' +modelinfo +"_"+str(epoch)+ '.model')
        if epoch == epochs:
            break
        print("---------------"+str(cycle)+" training distinctive contrast model ending-----------------------")
    return np.mean(averageloss)


def trainUncertanity(epochs,cycle, model, criterion, optimizer,  loader,device,saveStep=2, logger=None,modelbasepath="logs/models/",scheduler=None,savemodel=False):
    # This function train the model for one epoch
    log_=print if logger is None else logger.log
    # define some average metters for get eveage1
    trainloss = []
    print("Uncertainity Class Counts", np.unique(torch.cat([l for _,_,_,_,_,l, _ in loader]), return_counts=True))
    model.train()
    for epoch in range(0, epochs):
        # logger.log_dirname(f"Epoch {epoch}")
        #  train model for epoch
        batch_time = AverageMeter()
        losses={'LR':AverageMeter(),"linear":AverageMeter()}
        for n, (anchor,postive,negative, labels,salabel,nalabel,index) in enumerate(loader):
            check = time.time()
            ### SimCLR loss ###
            batch_size = anchor.size(0)
            anchor = anchor.to(device)
            labels = nalabel.to(device)
            optimizer.zero_grad()
            predictions = model(anchor)
            # print(prediction.shape,labels.shape,prediction.dtype,labels.dtype,prediction.device,labels.device)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch - 1 + n / len(loader))
            lr = optimizer.param_groups[0]['lr']
            batch_time.update(time.time() - check)

            ### Log losses ###
            # losses['cls'].update(0, batch_size)
            losses['linear'].update(loss,batch_size)
            losses['LR'].update(lr,batch_size)
            check = time.time()

        log_('[DONE] [Cycle %d] [Epoch %d] [Time %.3f] [LossLinear %f] ' %
             (cycle,epoch,batch_time.average,  losses['linear'].average))
        trainloss.append(losses['linear'].average.detach().cpu().numpy())
        if savemodel and (epoch % saveStep == 0 or epoch == epochs - 1):
            save_states = model.state_dict()
            os.makedirs(modelbasepath, exist_ok=True)
            modelinfo="Linear_" + str(cycle)
            save_checkpoint(epoch, save_states, optimizer.state_dict(), modelbasepath, modelinfo)
            print("MODEL SAVED AT **", modelbasepath + 'last_' + modelinfo+ "_" + str(epoch)+ '.model')
    print("---------------"+str(cycle)+" Uncertaininty training Complete-----------------------")
    return  np.mean(trainloss)

def DivideClasses(labels):
    SL_Labels=torch.tensor(labels)
    SL_Labels[SL_Labels==1]=0
    SL_Labels[SL_Labels==2]=1
    SL_Labels[SL_Labels==3]=1
    NA_Labels=torch.tensor(labels)
    NA_Labels[NA_Labels==2]=0
    NA_Labels[NA_Labels==3]=1
    return SL_Labels,NA_Labels

def EvaluateUncertainModel(model, trainloader, testLoader,criterion, device, logger, cycle=0, epoch=0):
    # This function Evaluate the model on train and test data
    model.eval()
    trainloss,testloss=[],[]
    train_labels,train_predictions,test_labels,test_predicitons=[],[],[],[]
    with torch.no_grad():
        for data,_,_, labels, index in trainloader:
            output=model(data.to(device)).cpu()
            train_labels.append(labels)
            train_predictions.append(output)
            trainloss.append(criterion(output.to(device),labels.to(device)).cpu())
        for data,_,_, labels, index in testLoader:
            output=model(data.to(device)).cpu()
            testloss.append(criterion(output.to(device),labels.to(device)).cpu())
            test_labels.append(labels)
            test_predicitons.append(output)
    train_labels,train_predictions=torch.cat(train_labels),torch.cat(train_predictions)
    test_labels, test_predicitons=torch.cat(test_labels),torch.cat(test_predicitons)
    if train_labels.dim()>1: train_labels=torch.argmax(train_labels,dim=1)
    if test_labels.dim()>1: test_labels=torch.argmax(test_labels,dim=1)
    if train_predictions.shape[1] > 1: _, train_predictions = torch.max(train_predictions, dim=1)
    else:train_predictions=(torch.squeeze(train_predictions)>.5).int()
    if test_predicitons.shape[1] > 1: _, test_predicitons = torch.max(test_predicitons, dim=1)
    else:test_predicitons=(torch.squeeze(test_predicitons)>.5).int()
    # S1 vs L
    SL_TrainingLabels,NA_TrainingLabels=DivideClasses(train_labels)
    SL_TrainingPrediction,NA_TrainingPrediction=DivideClasses(train_predictions)
    SL_TestLabels,NA_TestLabels=DivideClasses(test_labels)
    SL_TestPrediction,NA_TestPrediction=DivideClasses(test_predicitons)

    train_accuracy,test_accuacy=accuracy_score(train_labels,train_predictions),accuracy_score(test_labels,test_predicitons)
    sl_train_accuracy,sl_test_accuacy=accuracy_score(SL_TrainingLabels,SL_TrainingPrediction),accuracy_score(SL_TestLabels,SL_TestPrediction)
    na_train_accuracy, na_test_accuacy = accuracy_score(NA_TrainingLabels, NA_TrainingPrediction), accuracy_score(NA_TestLabels, NA_TestPrediction)
    logger.log("Cycle "+str(cycle)+" Epoch "+str(epoch)+" Training Accuracy: "+str(train_accuracy)+" Test Accuracy: "+str(test_accuacy))
    return train_accuracy,test_accuacy,sl_train_accuracy,sl_test_accuacy,na_train_accuracy, na_test_accuacy,torch.mean(torch.Tensor(trainloss)),torch.mean(torch.Tensor(testloss))

def TrainModel(contrastiveModel, contrastiveoptimizer, scheduler_warmup,lossname,uncertainitymodel,uncertainityoptimizer,criterion,uncertainty_train_loader,contrastive_train_loder,conepochs,uncertainepochs,cycle,logger,device,modelbasepath="logs/models/"):
    # Run experiments
    contrastiveTrainLoss, uncertainityTrainloss=None,None
    if contrastiveModel:
        contrastiveTrainLoss=trainContrastive(conepochs,cycle, contrastiveModel, contrastiveoptimizer, scheduler_warmup, lossname, contrastive_train_loder, logger=logger,device=device,modelbasepath=modelbasepath)
    if uncertainitymodel:
        uncertainityTrainloss=trainUncertanity(uncertainepochs, cycle, uncertainitymodel, criterion, uncertainityoptimizer, uncertainty_train_loader, device, saveStep=100, logger=logger,modelbasepath=modelbasepath)
    # print("contrastiveTrainLoss",contrastiveTrainLoss,"uncertainityTrainloss",uncertainityTrainloss)
    return contrastiveTrainLoss,uncertainityTrainloss

def UncertainPrediciton(model,dataloader,device):
    # def UncertainPrediciton(model, Data_set, indexixes, device):
    # dataloader = data.DataLoader(Data_set, batch_size=32, sampler=indexixes)
    model.eval()
    outputs,alllabels,naLabels=[],[],[]
    with torch.no_grad():
        for datasampel,_,_, labels,sllabels,nalabels, index in dataloader:
            outputs.append(model(datasampel.to(device)).cpu())
            alllabels.append(labels)
            naLabels.append(nalabels)
    return torch.max(torch.cat(outputs), dim=1)[1].numpy(),np.concatenate(naLabels),np.concatenate(alllabels)


def EvaluateUncertainModel(model, train_loader,test_loader,device):
    train_prediction,train_labels,_=UncertainPrediciton(model, train_loader, device)
    test_prediction,test_labels,_=UncertainPrediciton(model, test_loader, device)
    train_accuracy=accuracy_score(train_labels,train_prediction)
    test_accuracy=accuracy_score(test_labels,test_prediction)
    return train_accuracy,test_accuracy

