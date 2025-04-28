import numpy as np
import torch
from torch.nn.functional import pairwise_distance

import torch.distributed as dist
import diffdist.functional as distops

# This File have all custom losses
def get_similarity_matrix(outputs, chunk=2, multi_gpu=False):
    '''
        Compute similarity matrix
        - outputs: (B', d) tensor for B' = B * chunk
        - sim_matrix: (B', B') tensor
    '''

    if multi_gpu:
        outputs_gathered = []
        for out in outputs.chunk(chunk):
            gather_t = [torch.empty_like(out) for _ in range(dist.get_world_size())]
            gather_t = torch.cat(distops.all_gather(gather_t, out))
            outputs_gathered.append(gather_t)
        outputs = torch.cat(outputs_gathered)
        # 将在多个gpu上运行的feature集合到一起

    sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')
    # 矩阵相乘----计算内积
    # 余弦相似度 = (z1·z2 )/ (\\z1\\*\\z2\\) 已经归一化，范数值为1
    # = z1·z2 （矩阵相乘，具体到向量为点乘，内积）
    #取值范围[-1,1]

    return sim_matrix


def NT_xent(sim_matrix, temperature=0.5, chunk=2, eps=1e-8):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device
    B = sim_matrix.size(0) // chunk  # B = B' / chunk
    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal
    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix
    print(sim_matrix[:B, B:].diag().shape)
    loss = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)

    return loss


def NT_xentSimilarityLoss(simclr):
    sim_matrix = get_similarity_matrix(simclr)
    loss_sim = NT_xent(sim_matrix, temperature=0.5)
    return loss_sim


def CosineSimilarty(d1,d2):
    d1d2=torch.sum(torch.mul(d1,d2),dim=1)
    d1sqrt=torch.sqrt(torch.sum(d1**2,dim=1))
    d2sqrt=torch.sqrt(torch.sum(d2**2,dim=1))
    return torch.mean(torch.abs(d1d2/(d1sqrt*d2sqrt)))


def CosineSimilaityLoss(chunk=3):
    if chunk==2:
        def Loss(simclr):
            d1,d2=simclr.chunk(2)
            return CosineSimilarty(d1,d2)
    if chunk==3:
        def Loss(simclr,maxSim=100,sim_lambda=1,dis_lambda=1):
            assert simclr.size(0) %3==0,"The Loss is Invalid"
            anchor,postive,negative=simclr.chunk(3)
            poss_simm = CosineSimilarty(anchor,postive)
            neg_simm = CosineSimilarty(anchor,negative)
            # loss = max(0,maxSim-poss_simm) * sim_lambda + neg_simm * dis_lambda
            loss = max(0,((neg_simm *dis_lambda )-(poss_simm * sim_lambda))+maxSim )
            return loss,poss_simm,neg_simm
    return Loss

def euclideanDistance(d1,d2):
    return torch.mean(torch.sqrt(torch.sum((d1-d2)**2+1e-8,dim=1)))

def euclideanDistanceLoss(chunk=3):
    if chunk == 2:
        def Loss(simclr):
            d1,d2=simclr.chunk(2)
            return euclideanDistance(d1, d2)
    if chunk == 3:
        def Loss(simclr, maxdiss=100, sim_lambda=10, dis_lambda=10):
            assert simclr.size(0) % 3 == 0, "The Loss is Invalid"
            anchor,postive,negative=simclr.chunk(3)
            pos_diss = euclideanDistance(anchor, postive)
            negative_diss = euclideanDistance(anchor, negative)
            # Optimizer Work to decrease the loss so we have minus the negative_diss from maxSim so it can be increased by optimizer
            loss = pos_diss * sim_lambda + max(0,maxdiss-negative_diss) * dis_lambda
            # loss = max(0,((pos_diss * sim_lambda)-(negative_diss * dis_lambda))+maxdiss)
            # print(loss.detach().cpu().numpy(),pos_diss.detach().cpu().numpy(),negative_diss.detach().cpu().numpy(),(pos_diss-negative_diss).detach().cpu().numpy())
            return loss,pos_diss,negative_diss
    return Loss


def PairwiseDistanceLoss(chunk=3):
    if chunk == 2:
        def Loss(simclr):
            d1,d2=simclr.chunk(2)
            return torch.mean(pairwise_distance()(d1, d2))
    if chunk == 3:
        def Loss(simclr, maxdiss=100, sim_lambda=1, dis_lambda=1):
            assert simclr.size(0) % 3 == 0, "The Loss is Invalid"
            anchor,postive,negative=simclr.chunk(3)
            pos_diss = torch.sum(pairwise_distance(anchor, postive))
            negative_diss = torch.sum(pairwise_distance(anchor, negative))
            # Optimizer Work to decrease the loss so we have minus the negative_diss from maxSim so it can be increased by optimizer
            # loss = max(0,((pos_diss * sim_lambda)-(negative_diss * dis_lambda))+maxdiss)
            loss = pos_diss * sim_lambda + max(0, maxdiss - negative_diss) * dis_lambda
            return loss,pos_diss,negative_diss
    return Loss
