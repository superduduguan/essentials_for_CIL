import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from PIL import Image
import scipy.misc
import random
import copy
import math
import numpy as np
import time

from data.data_loader import cifar10, cifar100, ExemplarDataset

from lib.util import moment_update, TransformTwice, weight_norm, mixup_data, mixup_criterion, LabelSmoothingCrossEntropy
from lib.augment.cutout import Cutout
from lib.augment.autoaugment_extra import CIFAR10Policy
from models import *

compute_means = True
exemplar_means_ = []
avg_acc = []
announce = 0


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # training hyperparameters
    parser.add_argument('--batch-size',
                        type=int,
                        default=100,  
                        help='batch_size')
    parser.add_argument('--num-workers',
                        type=int,
                        default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs',
                        type=int,
                        default=120,
                        help='number of training epochs')
    parser.add_argument('--epochs-sd',
                        type=int,
                        default=70,
                        help='number of training epochs for self-distillation')
    parser.add_argument('--val-freq',
                        type=int,
                        default=10,
                        help='validation frequency')

    # incremental learning
    parser.add_argument('--new-classes',
                        type=int,
                        default=10,
                        help='number of classes in new task')
    parser.add_argument('--start-classes',
                        type=int,
                        default=50,
                        help='number of classes in old task')
    parser.add_argument('--K',
                        type=int,
                        default=2000,
                        help='2000 exemplars for CIFAR-100')

    # optimization
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr-min',
                        type=float,
                        default=0.0001,
                        help='lower end of cosine decay')
    parser.add_argument('--lr-sd',
                        type=float,
                        default=0.1,
                        help='learning rate for self-distillation')
    parser.add_argument('--lr-ft',
                        type=float,
                        default=0.01,
                        help='learning rate for task-2 onwards')
    parser.add_argument('--weight-decay',
                        type=float,
                        default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='momentum for SGD')
    parser.add_argument('--cosine',
                        action='store_true',
                        help='use cosine learning rate')

    # root folders
    parser.add_argument('--data-root',
                        type=str,
                        default='./data',
                        help='root directory of dataset')
    parser.add_argument('--output-root',
                        type=str,
                        default='./output',
                        help='root directory for output')

    # save and load
    parser.add_argument('--exp-name',
                        type=str,
                        default='kd',
                        help='experiment name')
    parser.add_argument('--resume', action='store_true', help='use class moco')
    parser.add_argument(
        '--resume-path',
        type=str,
        default='./checkpoint_0.pth',
    )
    parser.add_argument('--save',
                        action='store_true',
                        help='to save checkpoint')

    # loss function
    parser.add_argument('--pow',
                        type=float,
                        default=0.66,
                        help='hyperparameter of adaptive weight')
    parser.add_argument('--lamda',
                        type=float,
                        default=5,
                        help='weighting of classification and distillation')
    parser.add_argument('--lamda-sd',
                        type=float,
                        default=10,
                        help='weighting of classification and distillation')
    parser.add_argument(
        '--const-lamda',
        action='store_true',
        help='use constant lamda value, default: adaptive weighting')

    parser.add_argument('--w-cls',
                        type=float,
                        default=1.0,
                        help='weightage of new classification loss')

    # kd loss
    parser.add_argument('--kd', action='store_true', help='use kd loss')
    parser.add_argument('--w-kd',
                        type=float,
                        default=1.0,
                        help='weightage of knowledge distillation loss')
    parser.add_argument('--T',
                        type=float,
                        default=2,
                        help='temperature scaling for KD')
    parser.add_argument('--T-sd',
                        type=float,
                        default=2,
                        help='temperature scaling for KD')

    # self-distillation
    parser.add_argument('--num-sd',
                        type=int,
                        default=0,
                        help='number of self-distillation generations')
    parser.add_argument(
        '--sd-factor',
        type=float,
        default=5.0,
        help='weighting between classification and distillation')

    args = parser.parse_args()
    return args


def train(model, old_model, epoch, lr, tempature, lamda, train_loader, use_sd,
          checkPoint):

    # 初始化设定
    tolerance_cnt = 0
    step = 0
    best_acc = 0
    T = args.T

    model.cuda()
    old_model.cuda()
    criterion_ce = nn.CrossEntropyLoss(ignore_index=-1)

    if len(test_classes) // CLASS_NUM_IN_BATCH > 1:  #对于非initial base504605， 设定低初始lr 
        lr = args.lr_ft

    optimizer = optim.SGD(model.parameters(),  #初始化网络参数优化器
                          lr=lr,
                          momentum=0.9,
                          weight_decay=args.weight_decay)

    if len(test_classes) // CLASS_NUM_IN_BATCH == 1 and use_sd == True:  #初始化lr优化器
        if args.cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epoch, eta_min=0.001)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=0.1)
    else:
        if args.cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epoch, eta_min=args.lr_min)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[60, 90], gamma=0.1)

    # 当initial base训练结束后构建exemplar的dataloader，并将旧的model设置为eval模式
    if len(test_classes) // CLASS_NUM_IN_BATCH > 1:
        exemplar_set = ExemplarDataset(exemplar_sets, transform=transform_ori)
        exemplar_loader = torch.utils.data.DataLoader(
            exemplar_set,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=True)
        exemplar_loader_iter = iter(exemplar_loader)

        old_model.eval()
        num_old_classes = old_model.fc.out_features

    # 开始训练
    for epoch_index in tqdm(range(1, epoch + 1)):
        
        # 基本设定
        dist_loss = 0.0
        sum_loss = 0
        sum_dist_loss = 0
        sum_cls_new_loss = 0
        sum_cls_old_loss = 0
        sum_cls_loss = 0

        model.train()
        old_model.eval()
        old_model.freeze_weight()
        if announce:
            for param_group in optimizer.param_groups:
                print('learning rate: {:.4f}'.format(param_group['lr']))

        # 开始迭代
        for batch_idx, (x, _, target) in enumerate(train_loader):  # 前两个参数是两次独立transform_ori的结果， x是

            optimizer.zero_grad()

            # 计算新类样本在新模型上的分类损失 cls_loss_new（仅仅在新输出头上softmax）
            x, target = x.cuda(), target.cuda()
            targets = target - (len(test_classes) - CLASS_NUM_IN_BATCH)  # 相当于target的label减去旧样本类个数  
            logits = model(x)

            cls_loss_new = criterion_ce(logits[:, -CLASS_NUM_IN_BATCH:], targets) 
            loss = args.w_cls * cls_loss_new
            sum_cls_new_loss += cls_loss_new.item()

            # center loss
            # feats = model.forward(x, feat=True)
            # for index in range(len(x)):
                


            # 默认使用动态lamda
            if args.const_lamda:
                factor = args.lamda
            elif use_sd:
                factor = args.lamda_sd
            else:
                factor = ((len(test_classes) / CLASS_NUM_IN_BATCH)
                          ** (args.pow)) * args.lamda

            # 当有了旧样本和旧模型时
            if len(test_classes) // CLASS_NUM_IN_BATCH > 1:  # 只要不是phase0

                # 计算新类样本在新旧模型上的差异 dist_loss_new
                with torch.no_grad():
                    dist_target = old_model(x)  # 新类样本在旧模型上的logits
                logits_dist = logits[:, :-CLASS_NUM_IN_BATCH]  # 新类样本在新模型上的（旧输出头的）logits
                T = args.T
                dist_loss_new = nn.KLDivLoss()(
                    F.log_softmax(logits_dist / T, dim=1),
                    F.softmax(dist_target / T, dim=1)) * (T * T)

                # 加载memory
                try:
                    batch_ex = next(exemplar_loader_iter)
                except:  # 如果exemplar被遍历完一次了，就再来一次
                    exemplar_loader_iter = iter(exemplar_loader)
                    batch_ex = next(exemplar_loader_iter)

                # 计算旧类样本在新模型上的分类损失（全部输出头softmax）
                x_old, target_old = batch_ex
                x_old, target_old = x_old.cuda(), target_old.cuda()
                logits_old = model(x_old)  # 旧数据在新模型的分类预测
                old_classes = len(test_classes) - CLASS_NUM_IN_BATCH
                cls_loss_old = criterion_ce(logits_old, target_old.long())

                loss += cls_loss_old
                sum_cls_old_loss += cls_loss_old.item()

                # 计算旧类样本在新旧模型上的差异 dist_loss_old
                with torch.no_grad():
                    dist_target_old = old_model(x_old)  # 旧数据在旧模型上的输出
                logits_dist_old = logits_old[:, :-CLASS_NUM_IN_BATCH]  # 旧数据在新模型上的输出
                dist_loss_old = nn.KLDivLoss()(
                    F.log_softmax(logits_dist_old / T, dim=1),
                    F.softmax(dist_target_old / T, dim=1)) * (T * T)  

                dist_loss = dist_loss_old + dist_loss_new
                sum_dist_loss += dist_loss.item()
                loss += factor * args.w_kd * dist_loss  
               
            sum_loss += loss.item()

            loss.backward()
            optimizer.step()
            step += 1
            if announce:
                if (batch_idx + 1) % checkPoint == 0 or (
                        batch_idx + 1) == len(trainLoader):
                    print(
                        '==>>> epoch: {}, batch index: {}, step: {}, train loss: {:.3f}, dist_loss: {:3f}, cls_new_loss: {:.3f}, cls_old_loss: {:.3f}'
                        .format(epoch_index, batch_idx + 1, step,
                                sum_loss / (batch_idx + 1),
                                sum_dist_loss / (batch_idx + 1),
                                sum_cls_new_loss / (batch_idx + 1),
                                sum_cls_old_loss / (batch_idx + 1)))
        scheduler.step()


def evaluate_net(model, transform, train_classes, test_classes, i):
    model.eval()

    train_set = cifar100(root=args.data_root,
                         train=False,
                         classes=train_classes,
                         download=False,
                         transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=num_workers)

    total = 0.0
    correct = 0.0
    compute_means = True
    for j, (_, images, labels) in enumerate(train_loader):  # if train: return img, img_aug, target
        _, preds = torch.max(torch.softmax(model(images.cuda()), dim=1),
                             dim=1,
                             keepdim=False)
        labels = [y.item() for y in labels]
        np.asarray(labels)
        total += preds.size(0)
        correct += (preds.cpu().numpy() == labels).sum()

    # Train Accuracy
    print('correct: ', correct, 'total: ', total)
    print('Train Accuracy : %.2f ,' % (100.0 * correct / total))

    test_set = cifar100(root=args.data_root,
                        train=False,
                        classes=test_classes,
                        download=True,
                        transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers)

    total = 0.0
    correct = 0.0
    old2new = 0.0
    new2old = 0.0
    old2oldwrong = 0.0
    new2newwrong = 0.0
    for j, (_, images, labels) in enumerate(test_loader):
        out = torch.softmax(model(images.cuda()), dim=1)
        _, preds = torch.max(out, dim=1, keepdim=False)
        labels = [y.item() for y in labels]
        np.asarray(labels)
        total += preds.size(0)
        preds = preds.cpu().numpy()
        for index in range(len(labels)):
            if labels[index] < i and preds[index] >= i:
                old2new += 1
            if labels[index] >= i and preds[index] < i:
                new2old += 1
            if labels[index] < i and preds[index] < i and labels[index] != preds[index]:
                old2oldwrong += 1
            if labels[index] >= i and preds[index] >= i and labels[index] != preds[index]:
                new2newwrong += 1

        correct += (preds == labels).sum()

    # Test Accuracy
    test_acc = 100.0 * correct / total
    print('old2new:', old2new / total)
    print('new2old:', new2old / total)
    print('correct: ', correct / total)
    print('old2oldwrong', old2oldwrong / total)
    print('new2newwrong:', new2newwrong / total)
    print('Test Accuracy : %.2f' % test_acc)
    print('all', old2new+new2old+correct+old2oldwrong+new2newwrong)

    return test_acc


def icarl_reduce_exemplar_sets(m):
    for y, P_y in enumerate(exemplar_sets):
        exemplar_sets[y] = P_y[:m]


#Construct an exemplar set for image set
def icarl_construct_exemplar_set(model, images, m, transform):  # TODO:reconstruction
    model.eval()
    features = []
    
    with torch.no_grad():
        


        for index in range(len(images)):
            x = Variable(transform(Image.fromarray(images[index]))).cuda()
            x = x.unsqueeze(0)  # (1, 3, 32, 32) ==> (100, 3, 32, 32)
            if index == 0:
                X = x
            else:
                X = torch.cat((X, x), 0)
            
        feats = model.forward(X, rd=True)
        feats = feats.data.cpu().numpy()
        feats = feats / np.linalg.norm(feats, axis=1, ord=2, keepdims=True)
        features = feats


        features = np.array(features)
        class_mean = np.mean(features, axis=0)  # 全部训练样本feature的中点
        class_mean = class_mean / np.linalg.norm(class_mean)  # Normalize

        exemplar_set = []
        exemplar_features = []  
        exemplar_dist = []
        
        a = time.time()
        for k in range(int(m)):

            # 计算各个feature和中点的距离
            S = np.sum(exemplar_features, axis=0)  # 选出的feature相加
            phi = features  # 全部feature （500，64）
            mu = class_mean  # 全部feature的中点 （64）
            mu_p = 1.0 / (k + 1) * (phi + S)
            mu_p = mu_p / np.linalg.norm(mu_p)
            dist = np.sqrt(np.sum((mu - mu_p)**2, axis=1))

            # save：随机选样本
            idx = np.random.randint(0, features.shape[0])

            exemplar_dist.append(dist[idx])
            exemplar_set.append(images[idx])
            tmp = np.array(features[idx])
            exemplar_features.append(tmp)

            features[idx, :] = 0.0  # 被选到的样本特征置零，下一个循环里phi的这一行就很小了，和mean的差dist会很大
        b = time.time()
        

        # 将选好的features按距离排序
        exemplar_dist = np.array(exemplar_dist)
        exemplar_set = np.array(exemplar_set)
        ind = exemplar_dist.argsort()
        exemplar_set = exemplar_set[ind]

        exemplar_sets.append(np.array(exemplar_set))
    if announce:
        print('exemplar set shape: ', len(exemplar_set))

if __name__ == '__main__':
    
    # 确认参数
    args = parse_option()
    print(args)
    if not os.path.exists(os.path.join(args.output_root, "checkpoints/cifar/")):
        os.makedirs(os.path.join(args.output_root, "checkpoints/cifar/"))
    TOTAL_CLASS_NUM = 100
    CLASS_NUM_IN_BATCH = args.start_classes  #（initial class num 50)
    T = args.T
    K = args.K
    num_workers = 2

    exemplar_sets = []
    exemplar_means = []


    # default augmentation
    normalize = transforms.Normalize((0.5071, 0.4866, 0.4409),
                                     (0.2009, 0.1984, 0.2023))
    transform_ori = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # test-time augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    transform_train = TransformTwice(transform_ori, transform_ori)

    # 生成100个class索引
    class_index = [i for i in range(0, TOTAL_CLASS_NUM)]
    np.random.seed(1993)
    np.random.shuffle(class_index)

    # 加载模型
    net = resnet32_cifar(num_classes=CLASS_NUM_IN_BATCH).cuda()  #TODO
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())  #仅仅需要回传梯度的参数// filter(判断函数function, 可迭代对象iterable)
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('number of trainable parameters: ', params)
    
    old_net = copy.deepcopy(net)
    old_net.cuda()

    
    cls_list = [0] + [a for a in range(args.start_classes, 100, args.new_classes)]  #迭代56个phase[0, 50, 60, 70, 80, 90]
    for i in cls_list:
        print("==> Current Class: ", class_index[i:i + CLASS_NUM_IN_BATCH])
        print('==> Building model..')
        
        # 增加输出头
        if i == args.start_classes:  # 当phase = 1时(phase0 结束时)
            CLASS_NUM_IN_BATCH = args.new_classes  #10
            net.change_output_dim(new_dim=i + CLASS_NUM_IN_BATCH)
        if i > args.start_classes:  # 当phase > 1时，
            net.change_output_dim(new_dim=i + CLASS_NUM_IN_BATCH,  
                                  second_iter=True)
        print("current net output dim:", net.get_output_dim())

        # 加载当前phase数据集
        train_set = cifar100(root=args.data_root,  
                             train=True,
                             classes=class_index[i:i + CLASS_NUM_IN_BATCH],
                             download=True,
                             transform=transform_train)

        trainLoader = torch.utils.data.DataLoader(train_set,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=num_workers)

        train_classes = class_index[i:i + CLASS_NUM_IN_BATCH]
        test_classes = class_index[:i + CLASS_NUM_IN_BATCH]  
        print('train_classes', train_classes)
        print('test_classes ', test_classes)  

        # 管理memory
        m = K // (i + CLASS_NUM_IN_BATCH)  #记忆中每类样本数
        if i != 0:
            icarl_reduce_exemplar_sets(m)
        for y in tqdm(range(i, i + CLASS_NUM_IN_BATCH)):
            if announce:
                print("Constructing exemplar set for class-%d..." %
                      (class_index[y]))
            images = train_set.get_image_class(y)  #获取第y类全体图片
            icarl_construct_exemplar_set(net, images, m, transform_test)  # 对每一类构建样本集

        print("exemplar set ready")
        
        # 训练模型
        if args.resume and i == 0:
            net.load_state_dict(torch.load(args.resume_path))
            net.train()
        else:
            net.train()
            train(model=net,  
                  old_model=old_net,
                  epoch=args.epochs,
                  lr=args.lr,
                  tempature=T,
                  lamda=args.lamda,
                  train_loader=trainLoader,
                  use_sd=False,
                  checkPoint=50)
        
        if i != 0 and announce:  # 打印训练好的fc1和fc2的权值范数
            weight_norm(net)

        old_net = copy.deepcopy(net)
        old_net.cuda()


        
        # 保存模型
        if args.save:
            save_path = os.path.join(args.output_root, "checkpoints/cifar/", args.exp_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(
                net.state_dict(),
                os.path.join(save_path, 'checkpoint_' + str(i + CLASS_NUM_IN_BATCH) + '.pth'))

        # 测试
        transform_val = TransformTwice(transform_test, transform_test)
        test_acc = evaluate_net(  # TODO
            model=net,
            transform=transform_val,
            train_classes=class_index[i:i + CLASS_NUM_IN_BATCH],
            test_classes=class_index[:i + CLASS_NUM_IN_BATCH],
            i=i)
        avg_acc.append(test_acc)
        print('\n----------------------------------------------------\n')

    print(avg_acc)
    print('Avg accuracy: ', sum(avg_acc) / len(avg_acc))
