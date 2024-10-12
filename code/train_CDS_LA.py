import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from utils.feature_memory import *
from utils.sereparent import *
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='CDS_unlabel', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--model', type=str,  default='vnet_supervisedonly_dp', help='model_name')
parser.add_argument('--dropout_rate', type=float,  default=0.3)

args = parser.parse_args()
with open(args.root_path + '/../test.list', 'r') as f:
    image_list = f.readlines()
image_list = [args.root_path + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
test_save_path = "../model/prediction/"+args.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def update_ema_variables(model, model1,ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step*10 + 1), alpha)
    for ema_param, param,param1 in zip(ema_model.parameters(), model.parameters(),model1.parameters()):
        ema_param.data.mul_(alpha).add_(((param.data+param1.data)/2)*(1 - alpha))
num_classes = 2
patch_size = (112, 112, 80)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False, has_dropout=False):
        # Network definition
        if has_dropout:
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True,
                       dropout_rate=args.dropout_rate)
        else:
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model(has_dropout=True)  # student model
    ema_model = create_model(ema=True, has_dropout=True)  # teacher model
    model1 = model

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    db_test = LAHeart(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           CenterCrop(patch_size),
                           ToTensor()
                       ]))
    labeled_idxs = list(range(16))
    unlabeled_idxs = list(range(16, 80))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    model1.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    feature_memory = FeatureMemory(num_samples=16, memory_per_class=256, feature_size=256,
                                   n_classes=num_classes)

    feature_memory1 = FeatureMemory(num_samples=16, memory_per_class=256,
                                    feature_size=256,
                                    n_classes=num_classes)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    RAMP_UP_ITERS = 1000
    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model.train()
    model1.train()
    best_performance=0
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            loss = 0
            '''导入数据'''
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            '''选出无标签数据'''
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            ema_inputs=unlabeled_volume_batch
            '''学生有标签数据的数据增强'''
            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            student_input=volume_batch +noise
            '''学生A和学生B网络获取未标签数据和标签数据的预测向量和特征'''
            outputs_A, features_A = model(student_input)
            outputs_B, features_B = model1(student_input)
            '''老师网络获取未标签数据的预测向量和特征，以及伪标签'''
            with torch.no_grad():
                ema_output,features_unlabeled_teacher = ema_model(ema_inputs)
            softmax_u_w = torch.softmax(ema_output, dim=1)
            max_probs, pseudo_label_teacher = torch.max(softmax_u_w, dim=1)
            '''学生A和学生B的监督损失，[:labeled_bs]代表有标签数据'''
            loss_seg_A = F.cross_entropy(outputs_A[:labeled_bs], label_batch[:labeled_bs])
            loss_seg_B = F.cross_entropy(outputs_B[:labeled_bs], label_batch[:labeled_bs])

            '''学生A和学生B的竞争标签模块，[labeled_bs:]代表无标签数据'''
            outputs_A_norm = F.softmax(outputs_A[labeled_bs:], dim=1)
            outputs_B_norm = F.softmax(outputs_B[labeled_bs:], dim=1)
            [value1, indices1] = torch.max(outputs_A_norm, dim=1)
            [value2, indices2] = torch.max(outputs_B_norm, dim=1)
            '''学生A和学生B的预测向量更高的掩码'''
            COM_unlabeledA = torch.mul((value1 > value2).long().unsqueeze(1), outputs_A[labeled_bs:])
            COM_unlabeledB = torch.mul((value1 < value2).long().unsqueeze(1), outputs_B[labeled_bs:])
            '''学生A和学生B的竞争伪标签损失'''
            loss_Pseudo_seg_A = F.cross_entropy(COM_unlabeledA, pseudo_label_teacher)
            loss_Pseudo_seg_B = F.cross_entropy(COM_unlabeledB, pseudo_label_teacher)

            loss = loss + loss_seg_A+loss_seg_B+loss_Pseudo_seg_A+loss_Pseudo_seg_B

            '''在开始进行对比之前，构建100迭代的正记忆库和负记忆库'''
            if iter_num > RAMP_UP_ITERS:
                with torch.no_grad():
                    ema_output_label, labeled_features_ema = ema_model(volume_batch)
                '''记录老师网络中有标签数据中概率最大的类别'''
                probability_prediction_ema, label_prediction_ema = torch.max(torch.softmax(ema_output_label, dim=1), dim=1)
                '''有标签数据的标签：2 x 112 x 112 x 80 ---> 2 x 7 x 7 x 5'''
                labels_feature = nn.functional.interpolate(label_batch.float().unsqueeze(1), size=(
                labeled_features_ema.shape[2], labeled_features_ema.shape[3],labeled_features_ema.shape[4]),
                                                        mode='nearest').squeeze(1)
                '''概率最大的类别的掩码：2 x 112 x 112 x 80 ---> 2 x 7 x 7 x 5'''
                probability_prediction_feature = nn.functional.interpolate(probability_prediction_ema.float().unsqueeze(1),
                                                                        size=(labeled_features_ema.shape[2],
                                                                              labeled_features_ema.shape[3],
                                                                              labeled_features_ema.shape[4]),
                                                                        mode='nearest').squeeze(1)
                label_prediction_ema_feature = nn.functional.interpolate(
                    label_prediction_ema.float().unsqueeze(1),
                    size=(labeled_features_ema.shape[2],
                          labeled_features_ema.shape[3],
                          labeled_features_ema.shape[4]),
                    mode='nearest').squeeze(1)
                label_true_feature = nn.functional.interpolate(
                    label_batch.float().unsqueeze(1),
                    size=(labeled_features_ema.shape[2],
                          labeled_features_ema.shape[3],
                          labeled_features_ema.shape[4]),
                    mode='nearest').squeeze(1)
                '''概率>0.8的掩码，并构建特征向量和伪标签'''
                mask_prediction_correctly_T = ((label_prediction_ema_feature == label_true_feature).float()*(probability_prediction_feature > 0.85).float()).bool()
                labeled_features_correct = labeled_features_ema.permute(0, 2, 3, 4, 1)
                labels_down_correct_T = labels_feature[mask_prediction_correctly_T]
                labeled_features_correct_T = labeled_features_correct[mask_prediction_correctly_T, ...]
                if labeled_features_correct_T.size(0)>1:
                    '''获得老师网络的正样本投影头'''
                    labeled_features_correct_T = ema_model.projection_head(labeled_features_correct_T)
                    '''构建正样本记忆库'''
                    feature_memory.add_features_from_sample_learned(ema_model, labeled_features_correct_T,
                                                                    labels_down_correct_T, labeled_bs)
                '''获得概率<0.55的掩码，并构建特征向量和伪标签'''
                mask_prediction_correctly_F = ((label_prediction_ema_feature != label_true_feature).float()*(probability_prediction_feature < 0.53).float()).bool()
                labels_down_correct_F = labels_feature[mask_prediction_correctly_F]
                labeled_features_correct_F = labeled_features_correct[mask_prediction_correctly_F, ...]
                if labeled_features_correct_F.size(0) > 1:
                    '''获得老师网络的负样本投影头'''
                    labeled_features_correct_F = ema_model.projection_head(labeled_features_correct_F)
                    '''构建负样本记忆库'''
                    feature_memory1.add_features_from_sample_learned(ema_model, labeled_features_correct_F,
                                                                     labels_down_correct_F, labeled_bs)
            loss_high_level = 0
            loss_low_level = 0
            if iter_num > RAMP_UP_ITERS+1000:
                '''低层次对比学习'''
                '''老师网络伪标签映射：2 x 112 x 112 x 80 ---> 2 x 7 x 7 x 5'''
                pseudo_label_teacher_feature = nn.functional.interpolate(
                    pseudo_label_teacher.float().unsqueeze(1),
                    size=(labeled_features_ema.shape[2],
                          labeled_features_ema.shape[3],
                          labeled_features_ema.shape[4]),
                    mode='nearest').squeeze(1)
                '''老师网络无标签最大概率映射：2 x 112 x 112 x 80 ---> 2 x 7 x 7 x 5'''
                max_probs_feature = nn.functional.interpolate(
                    max_probs.float().unsqueeze(1),
                    size=(labeled_features_ema.shape[2],
                          labeled_features_ema.shape[3],
                          labeled_features_ema.shape[4]),
                    mode='nearest').squeeze(1)
                '''将老师网络的无标签特征、伪标签、最大概率分别映射到二维：490 x 256，490 x 1，490 x 1'''
                mask = (pseudo_label_teacher_feature != 3)
                features_unlabeled_teacher = features_unlabeled_teacher.permute(0, 2, 3, 4, 1)
                features_unlabeled_teacher = features_unlabeled_teacher[mask, ...]
                pseudo_label_teacher_feature_low = pseudo_label_teacher_feature[mask]
                max_probs_feature_low = max_probs_feature[mask]
                '''根据特征向量，获得老师网络的预测头'''
                projection_features_unlabeled_teacher = ema_model.projection_head(features_unlabeled_teacher)
                pred_features_unlabeled_teacher = ema_model.prediction_head(projection_features_unlabeled_teacher)
                '''学生A的无标签数据预测的伪标签'''
                probability_prediction_studentA, pseudo_label_studentA = torch.max(
                    torch.softmax(outputs_A, dim=1), dim=1)
                '''学生A伪标签映射：2 x 112 x 112 x 80 ---> 2 x 7 x 7 x 5'''
                pseudo_label_studentA_feature = nn.functional.interpolate(
                    pseudo_label_studentA.float().unsqueeze(1),
                    size=(labeled_features_ema.shape[2],
                          labeled_features_ema.shape[3],
                          labeled_features_ema.shape[4]),
                    mode='nearest').squeeze(1)
                '''将学生A网络的无标签特征映射到二维：490 x 256'''
                features_unlabeledA = features_A[labeled_bs:].permute(0, 2, 3, 4, 1)
                features_unlabeledA_low = features_unlabeledA[mask, ...]
                '''根据特征向量，获得学生A网络的预测头'''
                projection_features_unlabeledA_low = model.projection_head(features_unlabeledA_low)
                pred_features_unlabeledA_low = model.prediction_head(projection_features_unlabeledA_low)
                '''计算学生A低层次对比学习的损失'''
                loss_contr_unlabeledA = Class_separation(max_probs_feature_low,pred_features_unlabeled_teacher, pseudo_label_teacher_feature_low,
                                                         pred_features_unlabeledA_low, num_classes,
                                                         feature_memory.memory, feature_memory1.memory)
                '''学生B的无标签数据预测的伪标签'''
                probability_prediction_studentB, pseudo_label_studentB = torch.max(
                    torch.softmax(outputs_B, dim=1), dim=1)
                '''学生B伪标签映射：2 x 112 x 112 x 80 ---> 2 x 7 x 7 x 5'''
                pseudo_label_studentB_feature = nn.functional.interpolate(
                    pseudo_label_studentB.float().unsqueeze(1),
                    size=(labeled_features_ema.shape[2],
                          labeled_features_ema.shape[3],
                          labeled_features_ema.shape[4]),
                    mode='nearest').squeeze(1)
                '''将学生B网络的无标签特征映射到二维：490 x 256'''
                features_unlabeledB = features_B[labeled_bs:].permute(0, 2, 3, 4, 1)
                features_unlabeledB_low = features_unlabeledB[mask, ...]
                '''根据特征向量，获得学生B网络的预测头'''
                projection_features_unlabeledB_low = model1.projection_head(features_unlabeledB_low)
                pred_features_unlabeledB_low = model1.prediction_head(projection_features_unlabeledB_low)
                '''计算学生B低层次对比学习的损失'''
                loss_contr_unlabeledB = Class_separation(max_probs_feature_low,pred_features_unlabeled_teacher, pseudo_label_teacher_feature_low,
                                                         pred_features_unlabeledB_low, num_classes,
                                                         feature_memory.memory, feature_memory1.memory)
                loss_low_level = loss_contr_unlabeledA + loss_contr_unlabeledB
                loss = loss + (loss_low_level) * 0.01

                '''高层次对比学习'''
                '''学生A网络的高层次对比学习'''
                '''挑选学生A网络无标签数据中伪标签正确的掩码'''
                maskA1 = (pseudo_label_studentA[labeled_bs:] == pseudo_label_teacher)
                '''挑选学生A网络无标签数据的最大预测向量>的掩码'''
                outputs_normA = F.softmax(outputs_A[labeled_bs:], dim=1)
                [pred_joined_unlabeledA_MAX, indicesA] = torch.max(outputs_normA, dim=1)
                FA = 0.7
                maskA2 = (pred_joined_unlabeledA_MAX > FA)
                '''挑选学生A网络无标签数据的信息熵最小的前20%个'''
                prob = torch.softmax(outputs_A[labeled_bs:], dim=1)
                entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
                alpha_t = 20 * (1 - epoch_num / max_epoch)
                high_thresh = np.percentile(
                    entropy[pseudo_label_studentA[labeled_bs:] != 255].cpu().detach().numpy().flatten(),
                    100 - alpha_t, )
                high_entropy_mask3 = (
                            entropy.ge(high_thresh).float() * (pseudo_label_studentA[labeled_bs:] != 255).bool())
                maskA = torch.mul(torch.mul(maskA1, maskA2), high_entropy_mask3)  # 三个掩码相乘
                '''学生B网络的高层次对比学习'''
                '''挑选学生B网络无标签数据中伪标签正确的掩码'''
                maskB1 = (pseudo_label_studentB[labeled_bs:] == pseudo_label_teacher)
                '''挑选学生A网络无标签数据的最大预测向量>的掩码'''
                outputs_normB = F.softmax(outputs_B[labeled_bs:], dim=1)
                [pred_joined_unlabeledB_MAX, indicesB] = torch.max(outputs_normB, dim=1)
                maskB2 = (pred_joined_unlabeledB_MAX > FA)
                '''挑选学生A网络无标签数据的信息熵最大的前20%个'''
                probB = torch.softmax(outputs_B[labeled_bs:], dim=1)
                entropyB = -torch.sum(probB * torch.log(probB + 1e-10), dim=1)
                high_threshB = np.percentile(
                    entropyB[pseudo_label_studentB[labeled_bs:] != 255].cpu().detach().numpy().flatten(),
                    100 - alpha_t, )
                high_entropy_maskB3 = (
                            entropyB.ge(high_threshB).float() * (pseudo_label_studentB[labeled_bs:] != 255).bool())
                maskB = torch.mul(torch.mul(maskB1, maskB2), high_entropy_maskB3)  # 三个掩码相乘
                '''学生A的比学生B在对应像素点更可靠'''
                mask_A_CL = maskA > maskB
                '''学生A的比学生B更可靠的掩码映射：2 x 112 x 112 x 80 ---> 2 x 7 x 7 x 5'''
                mask_A_CL_low = nn.functional.interpolate(mask_A_CL.float().unsqueeze(1),
                                                           size=(labeled_features_ema.shape[2],
                                                                 labeled_features_ema.shape[3],
                                                                 labeled_features_ema.shape[4]),
                                                           mode='nearest').squeeze(1)
                mask_A_CL_DOWN = mask_A_CL_low.bool()
                '''将学生A网络的无标签特征映射到二维：? x 256'''
                features_high_level_A_low = features_A[labeled_bs:].permute(0, 2, 3, 4, 1)
                features_joined_unlabeledA_high_low = features_high_level_A_low[mask_A_CL_DOWN, ...]
                '''学生A的伪标签映射：2 x 112 x 112 x 80 ---> 2 x 7 x 7 x 5'''
                pseudo_labelA_high_feature = nn.functional.interpolate(
                    pseudo_label_studentA[labeled_bs:].float().unsqueeze(1),
                    size=(labeled_features_ema.shape[2],
                          labeled_features_ema.shape[3],
                          labeled_features_ema.shape[4]),
                    mode='nearest').squeeze(1)
                pseudo_labelA_high_feature_low = pseudo_labelA_high_feature[mask_A_CL_DOWN]
                if features_joined_unlabeledA_high_low.size(0) > 1:
                    '''根据特征向量，获得预测头'''
                    projection_features_joined_unlabeledA_high_low = model.projection_head(
                        features_joined_unlabeledA_high_low)
                    pred_features_joined_unlabeledA_high_low = model.prediction_head(
                        projection_features_joined_unlabeledA_high_low)
                    '''学生A的高层次对比学习损失第一部分'''
                    loss_high_level_cl_A1 = HIGH_LEVEL_CL_MENEYBAKK(pred_features_joined_unlabeledA_high_low,
                                                                    pseudo_labelA_high_feature_low,
                                                                    feature_memory.memory,
                                                                    feature_memory1.memory,
                                                                    num_classes)
                else:
                    loss_high_level_cl_A1 = 0

                '''学生B的比学生A在对应像素点更可靠'''
                mask_B_CL = maskA < maskB
                '''学生B的比学生A在对应像素点更可靠掩码映射：2 x 112 x 112 x 80 ---> 2 x 7 x 7 x 5'''
                mask_B_CL_low = nn.functional.interpolate(mask_B_CL.float().unsqueeze(1),
                                                           size=(labeled_features_ema.shape[2],
                                                                 labeled_features_ema.shape[3],
                                                                 labeled_features_ema.shape[4]),
                                                           mode='nearest').squeeze(1)
                mask_B_CL_DOWN = mask_B_CL_low.bool()
                '''将学生B网络的无标签特征映射到二维：490 x 256'''
                features_high_level_B_low = features_B[labeled_bs:].permute(0, 2, 3, 4, 1)
                features_joined_unlabeledB_high_low = features_high_level_B_low[mask_B_CL_DOWN, ...]
                '''学生B的伪标签映射：2 x 112 x 112 x 80 ---> 2 x 7 x 7 x 5'''
                pseudo_labelB_high_feature = nn.functional.interpolate(
                    pseudo_label_studentB[labeled_bs:].float().unsqueeze(1),
                    size=(labeled_features_ema.shape[2],
                          labeled_features_ema.shape[3],
                          labeled_features_ema.shape[4]),
                    mode='nearest').squeeze(1)
                pseudo_labelB_high_feature_low = pseudo_labelB_high_feature[mask_B_CL_DOWN]
                if features_joined_unlabeledB_high_low.size(0) > 1:
                    '''根据特征向量，获得预测头'''
                    projection_features_joined_unlabeledB_high_low = model1.projection_head(
                        features_joined_unlabeledB_high_low)
                    pred_features_joined_unlabeledB_high_low = model1.prediction_head(
                        projection_features_joined_unlabeledB_high_low)
                    '''学生B的高层次对比学习损失第一部分'''
                    loss_high_level_cl_B1 = HIGH_LEVEL_CL_MENEYBAKK(pred_features_joined_unlabeledB_high_low,
                                                                  pseudo_labelB_high_feature_low,
                                                                  feature_memory.memory,
                                                                  feature_memory1.memory,
                                                                  num_classes)
                else:
                    loss_high_level_cl_B1 = 0

                '''学生A和学生B都可靠，但是学生A的对应预测向量概率更大'''
                mask_A1_CL = torch.mul(torch.mul(maskA, maskB), pred_joined_unlabeledA_MAX > pred_joined_unlabeledB_MAX)
                '''学生A和学生B都可靠，但是学生A的对应预测向量概率更大的掩码映射：2 x 112 x 112 x 80 ---> 2 x 7 x 7 x 5'''
                mask_A1_CL_DOWN = nn.functional.interpolate(mask_A1_CL.float().unsqueeze(1),
                                                            size=(labeled_features_ema.shape[2],
                                                                 labeled_features_ema.shape[3],
                                                                 labeled_features_ema.shape[4]),
                                                            mode='nearest').squeeze(1)
                mask_A1_CL_DOWN = mask_A1_CL_DOWN.bool()
                '''学生A和学生B都可靠，但是学生A的对应预测向量概率更大的特征向量和伪标签'''
                features_joined_unlabeledA_highA_A = features_high_level_A_low[mask_A1_CL_DOWN, ...]
                pseudo_labelA_memoryclA_A = pseudo_labelA_high_feature[mask_A1_CL_DOWN]
                '''学生A和学生B都可靠，但是学生A的对应预测向量概率更大的特征向量和伪标签'''
                features_joined_unlabeledB_highB_A = features_high_level_B_low[mask_A1_CL_DOWN, ...]
                pseudo_labelB_memoryclB_A = pseudo_labelB_high_feature[mask_A1_CL_DOWN]
                if features_joined_unlabeledA_highA_A.size(0) > 1 and features_joined_unlabeledB_highB_A.size(0) > 1:
                    '''根据特征向量，获得预测头'''
                    projection_features_joined_unlabeledA_highA_A = model.projection_head(
                        features_joined_unlabeledA_highA_A)
                    pred_features_joined_unlabeledA_highA_A = model.prediction_head(
                        projection_features_joined_unlabeledA_highA_A)
                    '''根据特征向量，获得预测头'''
                    projection_features_joined_unlabeledB_highB_A = model1.projection_head(
                        features_joined_unlabeledB_highB_A)
                    pred_features_joined_unlabeledB_highB_A = model1.prediction_head(
                        projection_features_joined_unlabeledB_highB_A)
                    '''学生A的高层次对比学习损失第二部分'''
                    loss_high_level_cl_A2 = HIGH_LEVEL_CL(pred_features_joined_unlabeledA_highA_A, pseudo_labelA_memoryclA_A,
                                                        pred_features_joined_unlabeledB_highB_A, pseudo_labelB_memoryclB_A,
                                                        feature_memory1.memory,
                                                        num_classes)
                else:
                    loss_high_level_cl_A2 = 0

                '''学生A和学生B都可靠，但是学生B的对应预测向量概率更大'''
                mask_B1_CL = torch.mul(torch.mul(maskA, maskB), pred_joined_unlabeledA_MAX < pred_joined_unlabeledB_MAX)
                '''学生A和学生B都可靠，但是学生B的对应预测向量概率更大的掩码映射：2 x 112 x 112 x 80 ---> 2 x 7 x 7 x 5'''
                mask_B1_CL_DOWN = nn.functional.interpolate(mask_B1_CL.float().unsqueeze(1),
                                                            size=(labeled_features_ema.shape[2],
                                                                  labeled_features_ema.shape[3],
                                                                  labeled_features_ema.shape[4]),
                                                            mode='nearest').squeeze(1)
                mask_B1_CL_DOWN = mask_B1_CL_DOWN.bool()
                '''学生A和学生B都可靠，但是学生B的对应预测向量概率更大的特征向量和伪标签'''
                features_joined_unlabeledA_highA_B = features_high_level_A_low[mask_B1_CL_DOWN, ...]
                pseudo_labelA_memoryclA_B = pseudo_labelA_high_feature[mask_B1_CL_DOWN]
                '''学生A和学生B都可靠，但是学生B的对应预测向量概率更大的特征向量和伪标签'''
                features_joined_unlabeledB_highB_B = features_high_level_B_low[mask_B1_CL_DOWN, ...]
                pseudo_labelB_memoryclB_B = pseudo_labelB_high_feature[mask_B1_CL_DOWN]
                if features_joined_unlabeledA_highA_B.size(0) > 1 and features_joined_unlabeledB_highB_B.size(0) > 1:
                    '''根据特征向量，获得预测头'''
                    projection_features_joined_unlabeledA_highA_B = model.projection_head(
                        features_joined_unlabeledA_highA_B)
                    pred_features_joined_unlabeledA_highA_B = model.prediction_head(
                        projection_features_joined_unlabeledA_highA_B)
                    '''根据特征向量，获得预测头'''
                    projection_features_joined_unlabeledB_highB_B = model1.projection_head(
                        features_joined_unlabeledB_highB_B)
                    pred_features_joined_unlabeledB_highB_B = model1.prediction_head(
                        projection_features_joined_unlabeledB_highB_B)
                    '''学生B的高层次对比学习损失第二部分'''
                    loss_high_level_cl_A_B2 = HIGH_LEVEL_CL(pred_features_joined_unlabeledA_highA_B, pseudo_labelA_memoryclA_B,
                                                          pred_features_joined_unlabeledB_highB_B, pseudo_labelB_memoryclB_B,
                                                          feature_memory1.memory,
                                                          num_classes)
                else:
                    loss_high_level_cl_A_B2 = 0

                loss_high_level=loss_high_level_cl_A1 + loss_high_level_cl_B1 + loss_high_level_cl_A2 + loss_high_level_cl_A_B2
                loss = loss + loss_high_level * 0.1

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_segA', loss_seg_A, iter_num)
            writer.add_scalar('loss/loss_segB', loss_seg_B, iter_num)
            writer.add_scalar('loss/loss_Pseudo_seg_A', loss_Pseudo_seg_A, iter_num)
            writer.add_scalar('loss/loss_Pseudo_seg_B', loss_Pseudo_seg_B, iter_num)
            writer.add_scalar('loss/loss_high_level', loss_high_level, iter_num)
            writer.add_scalar('loss/loss_low_level', loss_low_level, iter_num)
            logging.info(
                    'iteration %d : loss : %f loss_segA: %f, loss_segB: %f, loss_Pseudo_seg_A: %f, loss_Pseudo_seg_B: %f, loss_high_level: %f, loss_low_level: %f' %
                    (iter_num, loss.item(), loss_seg_A, loss_seg_B, loss_Pseudo_seg_A, loss_Pseudo_seg_B,
                     loss_high_level, loss_low_level))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, model1, ema_model, args.ema_decay, iter_num)
            iter_num = iter_num + 1

            outputs_soft = F.softmax(outputs_A, dim=1)
            outputs_soft2 = F.softmax(outputs_B, dim=1)
            ema_output_soft = F.softmax(ema_output, dim=1)
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = torch.max(outputs_soft[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_labelA', grid_image, iter_num)

                image = torch.max(outputs_soft2[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_labelB', grid_image, iter_num)

                image = label_batch[0, :, : , 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                #####
                image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel/Image', grid_image, iter_num)

                image = torch.max(outputs_soft[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_labelA', grid_image, iter_num)

                image = torch.max(outputs_soft2[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_labelB', grid_image, iter_num)

                image = torch.max(ema_output_soft[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label_teacher', grid_image, iter_num)

                image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)
            if iter_num > 0 and iter_num % 100 == 0:
                model.eval()
                model1.eval()
                metric_list = 0.0
                avg_metric = test_all_case(model, image_list, num_classes=num_classes,
                                           patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                           save_result=True, test_save_path=test_save_path)
                avg_metric1 = test_all_case(model1, image_list, num_classes=num_classes,
                                           patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                           save_result=True, test_save_path=test_save_path)
                dice = max(avg_metric[0],avg_metric1[0])
                jc = max(avg_metric[1],avg_metric1[1])
                hd95 = max(avg_metric[2],avg_metric1[2])
                asd = max(avg_metric[3],avg_metric1[3])
                writer.add_scalar('info/val_mean_dice', dice, iter_num)
                writer.add_scalar('info/val_mean_jce', jc, iter_num)
                writer.add_scalar('info/val_mean_hd95', hd95, iter_num)
                writer.add_scalar('info/val_mean_asd', asd, iter_num)
                if dice > best_performance:
                    best_performance = dice
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                model.train()
                model1.train()
            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                save_mode_path1 = os.path.join(snapshot_path, 'Biter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path1)
                logging.info("save model to {}".format(save_mode_path1))
            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
