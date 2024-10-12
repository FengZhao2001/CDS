import torch
import torch.nn.functional as F
import random
import numpy as np
def Class_separation(max_probs_feature,pred_joined_unlabeledT, joined_pseudolabelsT,features_joined_unlabeledA,num_classes,memory,memory1):
    features_joined_unlabeledA = F.normalize(features_joined_unlabeledA, dim=1)  # N, 256
    '''挑选高置信度的老师网络特征'''
    BETA = 0.85
    T_MAX=max_probs_feature > BETA
    features_joined_unlabeledT_BESTBETA=pred_joined_unlabeledT[T_MAX,:]
    joined_pseudolabelsT=joined_pseudolabelsT[T_MAX]
    loss = 0
    delta=20;
    tor=0.5;
    P_ALL_CLASS = torch.empty(features_joined_unlabeledA.size(0), num_classes)
    for c in range(num_classes):
        '''计算类原型'''
        mask_c = joined_pseudolabelsT == c
        features_c = features_joined_unlabeledT_BESTBETA[mask_c, :]
        mask_c_num = mask_c.long()
        NUM_SELECTION = torch.sum(mask_c_num, dim=0)
        features_c = F.normalize(features_c, dim=1)  # M, 256
        TA = torch.sum(features_c, dim=0)
        '''Z_MEAN=类原型'''
        if NUM_SELECTION==0:
            Z_MEAN=TA
        else:
            Z_MEAN = TA / NUM_SELECTION
        '''根据类原型通过相似度获得学生网络的预测向量'''
        P = torch.exp((-delta * (1 - torch.cosine_similarity(features_joined_unlabeledA, Z_MEAN, dim=1))))
        P_ALL_CLASS[:, c] = P
    '''预测向量标准化'''
    P_ALL_CLASS = F.normalize(P_ALL_CLASS, dim=1)

    P_ALL_CLASS1 = P_ALL_CLASS
    features_joined_unlabeledA1 = features_joined_unlabeledA
    '''#挑选低阈值的预测向量
    [valueA, indicesA] = torch.max(P_ALL_CLASS, dim=1)
    GAA = 0.6
    T_MAXA = valueA < GAA
    P_ALL_CLASS1 = P_ALL_CLASS[T_MAXA, :]
    features_joined_unlabeledA1=features_joined_unlabeledA[T_MAXA,:]'''

    '''根据预测向量获得伪标签'''
    softmax_u_w1 = torch.softmax(P_ALL_CLASS1, dim=1)
    max_probs1p, pseudo_labelp = torch.max(softmax_u_w1, dim=1)  # Get pseudolabels
    for c in range(num_classes):
        mask_cp = pseudo_labelp == c
        features_joined_unlabeledA11=features_joined_unlabeledA1[mask_cp,:]
        memory_c = memory[c]
        memory_c1 = memory1[c]
        if memory_c is not None and memory_c.shape[0] > 1 and memory_c1.shape[0] > 1 and memory_c1 is not None and features_joined_unlabeledA11.shape[0] > 1:
            '''获取正负记忆库的第c类样本'''
            memory_c = torch.from_numpy(memory_c).cuda()
            memory_c = F.normalize(memory_c, dim=1)
            memory_c1 = torch.from_numpy(memory_c1).cuda()
            memory_c1 = F.normalize(memory_c1, dim=1)
            features_joined_unlabeledA11 = F.normalize(features_joined_unlabeledA11, dim=1)
            '''计算所有特征与所有正样本相似性'''
            similarities_positive = torch.mm(features_joined_unlabeledA11, memory_c.transpose(1, 0))
            distances_positive = 1 - similarities_positive
            '''计算所有特征与所有负样本相似性'''
            similarities_negitive = torch.mm(features_joined_unlabeledA11, memory_c1.transpose(1, 0))
            distances_negitive = 1 - similarities_negitive
            '''计算对比损失'''
            loss_1 = -torch.log(torch.exp(distances_positive / tor) / (
                        torch.exp(distances_positive / tor) + torch.sum(distances_negitive)))
            '''平均所有的正样本和特征数量'''
            loss = loss + torch.sum(loss_1) / (loss_1.size(0) * loss_1.size(1))

            '''对比学习'''
            '''features_joined_unlabeledA11_MEAN=torch.sum(features_joined_unlabeledA11,dim=0)/features_joined_unlabeledA11.size(0)
            EXP_negition=torch.exp(torch.cosine_similarity(features_joined_unlabeledA11_MEAN, memory_c1, dim=1)/tor)
            lce = -torch.log(
                torch.exp(torch.cosine_similarity(features_joined_unlabeledA11_MEAN, memory_c, dim=1) / tor) / (
                        torch.exp(torch.cosine_similarity(features_joined_unlabeledA11_MEAN, memory_c,
                                                          dim=1) / tor) + torch.sum(EXP_negition)))
            lce=torch.sum(lce)
            loss = loss + lce/memory_c1.size(0)'''
            '''for i1 in range(features_joined_unlabeledA11.size(0)):
                EXP_negition=torch.exp(torch.cosine_similarity(features_joined_unlabeledA11[i1,:], memory_c1, dim=1)/tor)
                lce = -torch.log(
                    torch.exp(torch.cosine_similarity(features_joined_unlabeledA11[i1, :], memory_c, dim=1) / tor) / (
                                torch.exp(torch.cosine_similarity(features_joined_unlabeledA11[i1, :], memory_c,
                                                                  dim=1) / tor) + torch.sum(EXP_negition)))
                lce=torch.sum(lce)
                loss = loss + lce/memory_c.size(0)'''
    return loss / (num_classes)
def HIGH_LEVEL_CL(pred_joined_unlabeledA_high_level,pseudo_label1,positives,pseudo_label2,memory,num_classes):
    loss = 0
    tor = 0.5;
    for c in range(num_classes):
        mask_c1 = pseudo_label1 == c
        mask_c2 = pseudo_label2 == c
        features_c1 = pred_joined_unlabeledA_high_level[mask_c1, :]
        features_c2 = positives[mask_c2, :]
        memory_c = memory[c]  # N, 256

        if memory_c is not None and features_c1.shape[0] > 1 and memory_c.shape[0] > 1 and features_c2.shape[0] > 1:
            memory_c = torch.from_numpy(memory_c).cuda()
            memory_c = F.normalize(memory_c, dim=1)  # N, 256
            '''if memory_c.size(0) > 100:
                list = range(1, memory_c.size(0))
                py = random.sample(list, 30)
                memory_c = memory_c[py, :]'''
            features_c1 = F.normalize(features_c1, dim=1)
            features_c2 = F.normalize(features_c2, dim=1)
            '''计算所有特征与所有正样本相似性'''
            similarities_positive = torch.mm(features_c1, features_c2.transpose(1, 0))
            distances_positive = 1 - similarities_positive
            '''计算所有特征与所有负样本相似性'''
            similarities_negitive = torch.mm(features_c1, memory_c.transpose(1, 0))
            distances_negitive = 1 - similarities_negitive
            '''计算对比损失'''
            loss_1 = -torch.log(torch.exp(distances_positive / tor) / (
                    torch.exp(distances_positive / tor) + torch.sum(distances_negitive)))
            '''平均所有的正样本和特征数量'''
            loss = loss + torch.sum(loss_1) / (loss_1.size(0) * loss_1.size(1))

            '''对比学习'''
            '''for i1 in range(features_c1.size(0)):
                EXP_negition = torch.exp(torch.cosine_similarity(features_c1[i1, :], memory_c, dim=1) / tor)
                lce = -torch.log(
                    torch.exp(torch.cosine_similarity(features_c1[i1, :], features_c2, dim=1) / tor) / (
                            torch.exp(torch.cosine_similarity(features_c1[i1, :], features_c2,
                                                              dim=1) / tor) + torch.sum(EXP_negition)))
                lce = torch.sum(lce)
                loss = loss + lce / memory_c.size(0)'''
    return loss / (num_classes)

def HIGH_LEVEL_CL_MENEYBAKK(pred_joined_unlabeled_high_level,pseudo_label,memory1,memory2,num_classes):
    loss = 0
    tor=0.5;
    for c in range(num_classes):
        mask_c = pseudo_label == c
        features_c = pred_joined_unlabeled_high_level[mask_c, :]
        memory_c1 = memory1[c]  # N, 256
        memory_c2 = memory2[c]  # N, 256
        if memory_c1 is not None and features_c.shape[0] > 1 and memory_c1.shape[0] > 1 and memory_c2.shape[0] > 1:
            memory_c1 = torch.from_numpy(memory_c1).cuda()
            memory_c2 = torch.from_numpy(memory_c2).cuda()
            memory_c1 = F.normalize(memory_c1, dim=1)  # N, 256
            memory_c2 = F.normalize(memory_c2, dim=1)  # N, 256
            features_c = F.normalize(features_c, dim=1)
            '''计算所有特征与所有正样本相似性'''
            similarities_positive = torch.mm(features_c, memory_c1.transpose(1, 0))
            distances_positive = 1 - similarities_positive
            '''计算所有特征与所有负样本相似性'''
            similarities_negitive = torch.mm(features_c, memory_c2.transpose(1, 0))
            distances_negitive = 1 - similarities_negitive
            '''计算对比损失'''
            loss_1 = -torch.log(torch.exp(distances_positive / tor) / (
                    torch.exp(distances_positive / tor) + torch.sum(distances_negitive)))
            '''平均所有的正样本和特征数量'''
            loss = loss + torch.sum(loss_1) / (loss_1.size(0) * loss_1.size(1))
    return loss / (num_classes)
