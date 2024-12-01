import os
# torch进行深度学习计算
import torch
# torch.nn模块提供了神经网络的基础结构，例如模型、层和激活函数
import torch.nn as nn
# tqdm用于进度条现实
from tqdm import tqdm
# numpy用于数组操作
import numpy as np
# torch.nn.functional包含了一些神经网络操作，例如激活函数和卷积函数，提供不带参数的函数操作
import torch.nn.functional as F
# sklearn进行GMM建模
import sklearn
import pandas as pd
# 从sklearn导入GaussianMixture类，用于高斯混合模型（GMM）的实现
from sklearn.mixture import GaussianMixture
# loss模块中的自定义损失函数ASDLoss
from loss import ASDLoss
# 包含的辅助函数utils模块
import utils

from torch.utils.data import DataLoader
from dataset import AccelDataset,AccelDataset_test

output_file = "submission.csv"



import torch
import torch.nn as nn


import torch
import torch.nn as nn

def bbox_ciou_loss_1d(pred_bbox, gt_bbox):
    """
    计算预测框和真实框之间的一维CIoU损失
    Args:
        pred_bbox: Tensor of shape [batch_size, 2], [x1, x2] (预测框)
        gt_bbox: Tensor of shape [batch_size, 2], [x1, x2] (真实框)
    Returns:
        ciou_loss: Tensor of shape [batch_size], 每个样本的CIoU损失
    """
    # 预测框和真实框的宽度
    pred_x1, pred_x2 = pred_bbox[:, 0], pred_bbox[:, 1]
    gt_x1, gt_x2 = gt_bbox[:, 0], gt_bbox[:, 1]

    pred_w = pred_x2 - pred_x1
    gt_w = gt_x2 - gt_x1

    # 预测框和真实框的中心点
    pred_cx = (pred_x1 + pred_x2) / 2
    gt_cx = (gt_x1 + gt_x2) / 2

    # IoU 计算
    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_x2 = torch.min(pred_x2, gt_x2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0)
    pred_area = pred_w
    gt_area = gt_w

    union_area = pred_area + gt_area - inter_area
    iou = inter_area / (union_area + 1e-6)

    # 中心点距离
    center_dist = (pred_cx - gt_cx) ** 2

    # 包围框计算
    enclose_x1 = torch.min(pred_x1, gt_x1)
    enclose_x2 = torch.max(pred_x2, gt_x2)

    enclose_diagonal = (enclose_x2 - enclose_x1) ** 2

    # CIoU 公式
    v = (4 / (torch.pi ** 2)) * ((torch.atan(gt_w + 1e-6) - torch.atan(pred_w + 1e-6)) ** 2)
    alpha = v / (1 - iou + v + 1e-6)

    ciou = iou - (center_dist / (enclose_diagonal + 1e-6)) - alpha * v
    ciou_loss = 1 - ciou

    return ciou_loss.mean()


def bbox_ciou_loss_dynamic_1d(pred_bboxes, gt_bboxes, iou_threshold=0.5):
    """
    支持预测框和真实框数量不一致的一维CIoU损失
    Args:
        pred_bboxes: Tensor of shape [num_preds, 2], [x1, x2] (预测框)
        gt_bboxes: Tensor of shape [num_gts, 2], [x1, x2] (真实框)
        iou_threshold: float, IoU阈值，匹配的最小要求
    Returns:
        loss: Scalar, 平均CIoU损失
    """
    def iou_1d(pred_bbox, gt_bbox):
        """计算单个预测框与真实框的一维IoU"""
        x1 = torch.max(pred_bbox[0], gt_bbox[0])
        x2 = torch.min(pred_bbox[1], gt_bbox[1])
        inter_area = torch.clamp(x2 - x1, min=0)
        pred_area = pred_bbox[1] - pred_bbox[0]
        gt_area = gt_bbox[1] - gt_bbox[0]
        union_area = pred_area + gt_area - inter_area
        return inter_area / (union_area + 1e-6)

    # 计算预测框与真实框的IoU矩阵
    iou_matrix = torch.zeros((pred_bboxes.size(0), gt_bboxes.size(0)))
    for i, pred_bbox in enumerate(pred_bboxes):
        for j, gt_bbox in enumerate(gt_bboxes):
            iou_matrix[i, j] = iou_1d(pred_bbox, gt_bbox)

    # 为每个真实框匹配一个预测框（IoU > 阈值）
    matched_pred_indices = []
    matched_gt_indices = []
    for gt_idx in range(gt_bboxes.size(0)):
        max_iou, pred_idx = torch.max(iou_matrix[:, gt_idx], dim=0)
        if max_iou > iou_threshold:
            matched_pred_indices.append(pred_idx.item())
            matched_gt_indices.append(gt_idx)

    # 计算匹配框的CIoU损失
    if len(matched_pred_indices) > 0:
        matched_preds = pred_bboxes[matched_pred_indices]
        matched_gts = gt_bboxes[matched_gt_indices]
        ciou_loss = bbox_ciou_loss_1d(matched_preds, matched_gts)
    else:
        # 没有匹配框时返回0损失
        ciou_loss = torch.tensor(0.0, requires_grad=True)

    return ciou_loss
# 示例
# pred_bboxes = torch.tensor([[50, 50, 100, 100], [120, 120, 180, 180], [30, 30, 70, 70]], dtype=torch.float32)
# gt_bboxes = torch.tensor([[60, 60, 110, 110], [25, 25, 80, 80]], dtype=torch.float32)
#
# loss = bbox_ciou_loss_dynamic_1d(pred_bboxes, gt_bboxes, iou_threshold=0.5)
# print(f"Dynamic CIoU Loss: {loss.item()}")





def extract_bboxes_pre(sequence):
    # sequence 是一个长度为 2800 的 0, 1 序列
    # 输出为包含 (start, end) 的边界框列表
    bboxes = []
    in_box = False
    start = 0
    sequence = (sequence > 0.5).float()
    for i in range(len(sequence[0])):
        if sequence[0][i] == 1 and not in_box:
            # 检测到事件的开始
            start = i
            in_box = True
        elif sequence[0][i] == 0 and in_box:
            # 检测到事件的结束
            end = i - 1
            bboxes.append((start, end))
            in_box = False

    # 如果序列以1结束，则补充一个终点
    if in_box:
        bboxes.append((start, len(sequence[0]) - 1))
    if len(bboxes) > 0:
        bboxes = torch.tensor(bboxes, dtype=torch.int32)
    else:
        bboxes = torch.empty((0, 2), dtype=torch.int32)


    return bboxes
def extract_bboxes(sequence):
    # sequence 是一个长度为 2800 的 0, 1 序列
    # 输出为包含 (start, end) 的边界框列表
    bboxes = []
    in_box = False
    start = 0

    for i in range(len(sequence[0])):

        if sequence[0][i] == 1 and not in_box:
            # 检测到事件的开始
            start = i
            in_box = True
        elif sequence[0][i] == 0 and in_box:
            # 检测到事件的结束
            end = i - 1
            bboxes.append((start, end))
            in_box = False

    # 如果序列以1结束，则补充一个终点
    if in_box:
        bboxes.append((start, len(sequence[0]) - 1))
    if len(bboxes) > 0:
        bboxes = torch.tensor(bboxes, dtype=torch.int32)
    else:
        bboxes = torch.empty((0, 2), dtype=torch.int32)
    return bboxes

def bbox_loss(predicted_bboxes, label_bboxes):
    # predicted_bboxes 和 label_bboxes 都是包含 (start, end) 的边界框列表
    l1_loss = 0.0
    iou_loss = 0.0
    num_boxes_pre = len(predicted_bboxes)
    num_boxes = len(label_bboxes)
    if num_boxes_pre != num_boxes:
        total_loss = 0.0
        return total_loss
    for pred_box, label_box in zip(predicted_bboxes, label_bboxes):
        pred_start, pred_end = pred_box
        label_start, label_end = label_box

        # 计算 L1 损失
        l1_loss += abs(pred_start - label_start) + abs(pred_end - label_end)

        # 计算 IoU 损失
        intersection = max(0, min(pred_end, label_end) - max(pred_start, label_start) + 1)
        union = (pred_end - pred_start + 1) + (label_end - label_start + 1) - intersection
        iou = intersection / (union + 1e-6)
        iou_loss += (1 - iou)

    # 归一化损失
    if num_boxes > 0:
        l1_loss /= num_boxes
        iou_loss /= num_boxes

    total_loss = l1_loss + iou_loss
    return total_loss

def total_loss_function(predicted_id, label):
    # 从 predicted_id 和 label 中提取边界框
    predicted_bboxes = extract_bboxes(predicted_id)
    label_bboxes = extract_bboxes(label)

    # 计算 bbox 损失
    bbox_regression_loss = bbox_loss(predicted_bboxes, label_bboxes)

    # 计算 MSE 损失
    mse_loss = torch.nn.functional.mse_loss(torch.tensor(predicted_id, dtype=torch.float32), torch.tensor(label, dtype=torch.float32))

    # 总损失
    total_loss = mse_loss + bbox_regression_loss
    return total_loss
def iou_loss(logits, labels, threshold=0.5):
    """
    计算自定义损失，适用于时间序列检测任务
    Args:
        logits: 模型输出的预测值 (Batch, Time_Series_Length)
        labels: 真实标签 (Batch, Time_Series_Length)
        threshold: 阈值，将连续输出转为二值
    Returns:
        自定义损失，越小越好
    """
    # 将logits和labels都转化为0-1的二值
    logits_binary = (logits > threshold).float()
    labels_binary = (labels > threshold).float()

    # 计算logits_binary和labels_binary对应位置的差的绝对值的和
    loss = torch.sum(torch.abs(logits_binary - labels_binary), dim=1) / 2800  # (Batch,)

    # 返回损失的均值
    return loss.mean()
# def iou_loss(logits, labels, threshold=0.5):
#     """
#     计算IoU损失，适用于时间序列检测任务
#     Args:
#         logits: 模型输出的预测值 (Batch, Time_Series_Length)
#         labels: 真实标签 (Batch, Time_Series_Length)
#         threshold: 阈值，将连续输出转为二值
#     Returns:
#         IoU损失，越小越好
#     """
#     # 将logits和labels都转化为0-1的二值
#     logits_binary = (logits > threshold).float()
#     labels_binary = (labels > threshold).float()
#
#     # 计算交集和并集
#     intersection = torch.sum(logits_binary * labels_binary, dim=1)  # (Batch,)
#     union = torch.sum(logits_binary, dim=1) + torch.sum(labels_binary, dim=1) - intersection  # (Batch,)
#
#     # 计算IoU
#     iou = (intersection + 1e-6) / (union + 1e-6)  # 防止除以0
#
#     # IoU 损失为 1 - IoU
#     iou_loss_value = 1 - iou  # (Batch,)
#
#     # 返回 IoU 损失的均值
#     return iou_loss_value.mean()

def mixup_data(x1, x2, x3, x4, x5, x6, labels, alpha=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).to(x1.device)

    # 对输入进行线性组合
    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    mixed_x3 = lam * x3 + (1 - lam) * x3[index, :]
    mixed_x4 = lam * x4 + (1 - lam) * x4[index, :]
    mixed_x5 = lam * x5 + (1 - lam) * x5[index, :]
    mixed_x6 = lam * x6 + (1 - lam) * x6[index, :]

    # 对标签进行线性组合
    labels_a, labels_b = labels, labels[index]

    return mixed_x1, mixed_x2, mixed_x3, mixed_x4, mixed_x5, mixed_x6, labels_a, labels_b ,lam

# def generate_csv(ts, predict_id, subject, output_file):
#     # 检查两个列表长度是否相等
#     if len(ts) != len(predict_id):
#         print(len(ts))
#         print(len(predict_id[0]))
#         raise ValueError("The length of timestamp and predict_id must be the same")
#
#     # 检查文件是否存在，如果存在则读取文件并获取最大id
#     if os.path.exists(output_file):
#         existing_df = pd.read_csv(output_file)
#         start_id = existing_df['id'].max() + 1
#     else:
#         start_id = 0
#
#     # 生成自增序列的id
#     ids = list(range(start_id, start_id + len(ts)))
#
#     # 创建DataFrame
#     data = {
#         'id': ids,
#         'subject': [subject] * len(ts),
#         'timestamp': ts,
#         'label': predict_id
#     }
#     df = pd.DataFrame(data)
#
#     # 将数据写入CSV文件，如果文件存在则追加写入，不覆盖
#     df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
global_df_l = pd.DataFrame(columns=['id', 'subject', 'timestamp', 'label'])

def generate_combined_df(ts, predict_id, subject):


    # 检查两个列表长度是否相等
    if len(ts) != len(predict_id):
        print(len(ts))
        print(len(predict_id[0]))
        raise ValueError("The length of timestamp and predict_id must be the same")

    # 获取最大id
    # if not global_df.empty:
    #     start_id = global_df['id'].max() + 1
    # else:
    #     start_id = 0

    # 生成自增序列的id
    ids = list(range(0, 0 + len(ts)))

    # 创建DataFrame
    data = {
        'id': ids,
        'subject': [subject.item()] * len(ts),
        'timestamp': predict_id.tolist(),
        'label': predict_id.tolist()
    }
    new_df = pd.DataFrame(data)

    # 合并新的DataFrame到全局DataFrame
    global_df = pd.concat([global_df_l, new_df], ignore_index=True)

    return global_df
# 初始化Trainer类，接受可变参数（*args和**kwargs）以方便传入各种参数
class Trainer:
    # _init是构造函数，*args和**kwargs用于接受可变数量的参数
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        # 从kwargs中提取net，该参数表示模型网络对象
        self.net = kwargs['net']
        # 从kwargs中提供optimizer，用于优化模型参数
        self.optimizer = kwargs['optimizer']
        # 从kwargs中提取scheduler，控制学习率的调整策略
        self.scheduler = kwargs['scheduler']
        # 从args中提取writer和logger，分别用于记录训练过程中的数据和日志
        self.writer = self.args.writer
        self.logger = self.args.logger
        # 初始化损失函数ASDLoss并将其移动到指定设备（CPU或GPU),用于计算模型的损失。
        self.criterion = ASDLoss().to(self.args.device)
        # 从kwargs中提取transform，它是一个数据转换器，用于处理输入数据
        # self.transform = kwargs['transform']

    # 定义train函数，用于管理模型的训练过程
    # train_loader是数据加载器，用于加载训练数据
    def train(self, train_loader):
        # self.test(save=False)
        # model_dir定义了模型的保存路径，将日志文件夹下的model文件夹作为保存模型的目录
        model_dir = os.path.join(self.writer.log_dir, 'model')
        # 创建model_dir路径；如果目录以及存在，则跳过创建（exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        # 从args中提取相关配置
        # 总的训练轮数（epochs）
        epochs = self.args.epochs
        # 验证频率（valid_every_epochs）
        valid_every_epochs = self.args.valid_every_epochs
        # 提前停止的轮数（early_stop_epochs）
        early_stop_epochs = self.args.early_stop_epochs
        # 开始验证的轮数（start_valid_epoch）
        start_valid_epoch = self.args.start_valid_epoch
        # 每轮的训练步骤数（num_steps）
        num_steps = len(train_loader)
        # 初始化训练的总步数
        self.sum_train_steps = 0
        # 初始化验证的总步数
        self.sum_valid_steps = 0
        # best_metric表示最佳性能指标
        best_metric = 0
        # no_better_epoch表示没有提升的轮数，用于提前停止
        no_better_epoch = 0
        # 按照指定的轮数epochs，逐轮迭代进行训练
        for epoch in range(0, 300):
            # train
            # 重置sum_loss以累积每轮的总损失，将模型转换到训练模型self.net.train()
            sum_loss = 0
            self.net.train()
            # 使用tqdm创建进度条，显示当前的训练进度
            # desc=f'Epoch-{epoch}'表示当前的轮数
            train_bar = tqdm(train_loader, total=num_steps, desc=f'Epoch-{epoch}')
            # 对每个批次的数据进行循环，x_wavs和x_mels是输入数据，labels是标签
            for (x_xwavs, x_ywavs, x_zwavs, x_xmels, x_ymels, x_zmels, labels) in train_bar:
                # forward
                # 将x_xwavs、x_ywavs、x_zwavs、x_xmels、x_ymels、x_zmels转换为浮点型（float()）并移动到指定设备
                x_xwavs,x_ywavs,x_zwavs = (
                    x_xwavs.float().to(self.args.device),
                    x_ywavs.float().to(self.args.device),
                    x_zwavs.float().to(self.args.device),
                )
                x_xmels,x_ymels,x_zmels = (
                    x_xmels.float().to(self.args.device),
                    x_ymels.float().to(self.args.device),
                    x_zmels.float().to(self.args.device),
                )

                # labels重塑为一维并转换为长整型（long()）以适应分类任务
                labels = labels.float().to(self.args.device)


                mixed_x_xwavs, mixed_x_ywavs, mixed_x_zwavs, mixed_x_xmels, mixed_x_ymels, mixed_x_zmels, labels_a, labels_b,  lam = mixup_data(
                    x_xwavs, x_ywavs, x_zwavs, x_xmels, x_ymels, x_zmels, labels, alpha=0.5)

                # 前向传播，将输入传递到模型self.net，获得logits（模型输出）
                logits,_ = self.net(
                    mixed_x_xwavs, mixed_x_ywavs, mixed_x_zwavs,
                    mixed_x_xmels, mixed_x_ymels, mixed_x_zmels, labels
                )

                # 计算混合损失 (使用 mixup)
                # criterion = torch.nn.CrossEntropyLoss()  # 根据你使用的任务可能需要调整损失函数
                loss_mse = lam * self.criterion(logits, labels_a) + (1 - lam) * self.criterion(logits, labels_b)
                # predicted_bboxes = extract_bboxes_pre(logits)
                # label_bboxes_a = extract_bboxes(labels_a)
                # label_bboxes_b = extract_bboxes(labels_b)
                # c_iou_loss_a  =  bbox_ciou_loss_dynamic_1d(predicted_bboxes,label_bboxes_a)
                # c_iou_loss_b  =  bbox_ciou_loss_dynamic_1d(predicted_bboxes,label_bboxes_b)





                iou_loss_a = iou_loss(logits, labels_a, threshold=0.5)
                iou_loss_b = iou_loss(logits, labels_b, threshold=0.5)
                loss_iou = lam * iou_loss_a + (1 - lam) * iou_loss_b
                loss = loss_mse +  loss_iou
                # loss = loss_mse

                # labels = labels.reshape(-1).long().to(self.args.device)
                # print(x_wavs.shape, x_mels.shape, labels.shape)
                # 前向传播，将输入传递到模型self.net,获得logits（模型输出）
                # print("shape of x_xwavs:",x_xwavs.shape)
                # print("shape of x_xmels:",x_xmels.shape)
                # logits, _ = self.net(x_xwavs, x_ywavs, x_zwavs, x_xmels, x_ymels, x_zmels, labels)
                #
                # # 使用损失函数计算损失loss，表示logits与真实标签labels之间的差异
                # loss = self.criterion(logits, labels)
                # 使用tqdm显示当前损失值，更新进度条上的损失信息
                train_bar.set_postfix(
                    mse_loss=f'{loss_mse.item():.5f}',
                    iou_loss=f'{loss_iou.item():.5f}',
                    total_loss=f'{loss.item():.5f}'

                )
                # backward
                # 梯度清零(zero_grad)
                self.optimizer.zero_grad()
                # 计算梯度(backward)
                loss.backward()
                # 更新模型参数(step)
                self.optimizer.step()
                # visualization
                # 使用writer记录损失值train_loss，
                self.writer.add_scalar(f'train_loss', loss.item(), self.sum_train_steps)
                # 累计该批次损失到sum_loss，
                sum_loss += loss.item()
                # 并更新sum_train_steps
                self.sum_train_steps += 1
            # 计算平均损失avg_loss，用该轮的累计损失sum_loss除以步数num_steps
            avg_loss = sum_loss / num_steps
            # 如果学习率调度器存在，并且已到达指定轮数，则更新学习率
            if self.scheduler is not None and epoch >= self.args.start_scheduler_epoch:
                self.scheduler.step()
            # 使用日志记录器记录当前轮数和平均损失
            self.logger.info(f'Epoch-{epoch}\tloss:{avg_loss:.5f}')
            # valid
            # 验证与早停策略
            # 检查是否满足验证条件：从start_valid_epoch开始，并每隔valid_every_epochs进行验证
            # if epoch % 2==0  :
            #
            #     # 调用test函数获取avg_auc和avg_pauc，并记录AUC和pAUC以便监控模型性能
            #     # avg_auc, avg_pauc = self.test(save=False, gmm_n=False)
            #     self.test(save=False)
            if epoch % 10==0:
                model_save_path = f"final_model_mixmse_mixiou_100_epoch_fianl_{epoch}.pth"
                torch.save(self.net.state_dict(), model_save_path)
            #     self.writer.add_scalar(f'auc', acc_balanced , epoch)
            #     # 如果当前性能(avg_auc+avg_pauc)优于最佳性能best_metric，则更新最佳性能，重置no_better_epoch
            #     if acc_balanced >= best_metric:
            #         no_better_epoch = 0
            #         best_metric = acc_balanced
            #         # 保存当前最佳模型的状态字典(state_dict)到best_model_path
            #         best_model_path = os.path.join(model_dir, 'best_checkpoint.pth.tar')
            #         utils.save_model_state_dict(best_model_path, epoch=epoch,
            #                                     net=self.net.module if self.args.dp else self.net,
            #                                     optimizer=None)
            #         self.logger.info(f'Best epoch now is: {epoch:4d}')
            #     # 否则，no_better_epoch增加1，记录连续无提升的轮次数
            #     else:
            #         # early stop
            #         no_better_epoch += 1
            #         # 如果连续无提升的轮次数超过early_stop_epochs，则停止训练，并记录日志信息
            #         if no_better_epoch > early_stop_epochs > 0: break
            # # save last 10 epoch state dict
            # # 当epoch大于等于start_save_model_epochs时，模型每隔一定的save_model_interval_epochs进行保存
            # if epoch >= self.args.start_save_model_epochs:
            #     if (epoch - self.args.start_save_model_epochs) % self.args.save_model_interval_epochs == 0:
            #         # 根据条件创建模型文件路径并保存模型状态字典
            #         model_path = os.path.join(model_dir, f'{epoch}_checkpoint.pth.tar')
            #         utils.save_model_state_dict(model_path, epoch=epoch,
            #                                     net=self.net.module if self.args.dp else self.net,
            #                                     optimizer=None)

                                                            

    def test(self, save=False):
        """
        测试模型在测试集上的性能，并输出 chunk_id 和预测的 label 到 CSV 文件。
        """
        # 初始化列表，用于存储预测结果
        csv_output = []
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        
        # 设置模型为评估模式
        self.net.eval()
        net = self.net.module if self.args.dp else self.net
        net1 = self.net.module if self.args.dp else self.net
        net2 = self.net.module if self.args.dp else self.net
        print('\n' + '=' * 20)
        #加载模型权重文件
        model_save_path = "final_model_mixmse_mixiou_100_epoch_fianl_80.pth"
        model_save_path1 = "model_mixmse_mixiou_100_epoch_fianl_80.pth"
        model_save_path2 = "model_mixmse_mixiou_epoch_80.pth"
        if os.path.exists(model_save_path):
            checkpoint = torch.load(model_save_path, map_location=self.args.device)

            # 处理因 DataParallel 保存的模型多了 'module.' 前缀的问题
            new_state_dict = {}
            for key, value in checkpoint.items():
                new_key = key.replace("module.", "")  # 移除 'module.' 前缀
                new_state_dict[new_key] = value

            net.load_state_dict(new_state_dict)
            print(f"加载模型权重文件：{model_save_path}")
        if os.path.exists(model_save_path1):
            checkpoint = torch.load(model_save_path, map_location=self.args.device)

            # 处理因 DataParallel 保存的模型多了 'module.' 前缀的问题
            new_state_dict = {}
            for key, value in checkpoint.items():
                new_key = key.replace("module.", "")  # 移除 'module.' 前缀
                new_state_dict[new_key] = value

            net1.load_state_dict(new_state_dict)
            print(f"加载模型权重文件：{model_save_path1}")
        if os.path.exists(model_save_path2):
            checkpoint = torch.load(model_save_path, map_location=self.args.device)

            # 处理因 DataParallel 保存的模型多了 'module.' 前缀的问题
            new_state_dict = {}
            for key, value in checkpoint.items():
                new_key = key.replace("module.", "")  # 移除 'module.' 前缀
                new_state_dict[new_key] = value

            net2.load_state_dict(new_state_dict)
            print(f"加载模型权重文件：{model_save_path2}")
        # 加载测试数据集
        test_dataset = AccelDataset_test(self.args)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 假设一次处理一个文件

        y_true, y_pred = [], []
        accs=[]
        for idx, (x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel, ts , subject) in enumerate(test_loader):
            # 将数据移动到指定设备
            x_xwav, x_ywav, x_zwav = x_xwav.to(self.args.device), x_ywav.to(self.args.device), x_zwav.to(self.args.device)
            x_xmel, x_ymel, x_zmel = x_xmel.to(self.args.device), x_ymel.to(self.args.device), x_zmel.to(self.args.device)

            # 前向传播，预测结果

            with torch.no_grad():
                predict_ids,_ = net(x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel, None)
                predict_ids1,_ = net1(x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel, None)
                predict_ids2,_ = net2(x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel, None)
                stacked_tensor = torch.stack([predict_ids, predict_ids1, predict_ids2])

                # 对堆叠的 tensor 取平均值
                predict_ids=torch.mean(stacked_tensor, dim=0)

                predict_ids = (predict_ids > 0.5).int().cpu()
                #计算 predict_ids 和 label 对应位置元素是否相同
                df = generate_combined_df(ts, predict_ids[0], subject)
                df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
        print("over")
        #         correct_matches = (predict_ids == label).sum().item()  # 相同元素的数量
        #
        #         # 计算准确率
        #         accuracy = correct_matches / 2800
        #         accs.append(accuracy)
        #
        # print("acc:",np.mean(accs))

        #     # 获取 chunk_id
        #     chunk_id = idx  # 或从文件名解析得到，如 os.path.basename(file_path)
        #
        #     # 将结果保存到列表
        #     csv_output.append([chunk_id, predicted_label])
        #
        # # 计算TP, TN, FP, FN
        # TP = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
        # TN = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
        # FP = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
        # FN = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
        #
        # # 计算 TPR 和 TNR
        # TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        # TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
        #
        # # 计算 Acc_balanced
        # acc_balanced = 0.5 * (TPR + TNR)
        #
        # # 打印和保存结果
        # self.logger.info(f'Acc_balanced: {acc_balanced:.3f}')
        #
        # # 保存到 CSV 文件
        # if save:
        #     csv_path = os.path.join(result_dir, 'test_predictions.csv')
        #     with open(csv_path, 'w') as f:
        #         f.write("chunk_id,label\n")  # 写入标题
        #         for chunk_id, label in csv_output:
        #             f.write(f"{chunk_id},{label}\n")
        #
        #     # 打印日志
        #     self.logger.info(f"Test results saved to {csv_path}")
        #
        # return acc_balanced

    # def test(self, save=False):
    #     """
    #     测试模型在测试集上的性能，使用 Acc_balanced = 0.5 * (TPR + TNR) 作为指标。
    #     """
    #     # 初始化csv_lines列表，用于记录结果信息
    #     csv_lines = []
    #     # 累加用于计算平均指标的变量
    #     sum_acc_balanced, num = 0, 0
    #     # 定义结果保存目录
    #     result_dir = os.path.join(self.args.result_dir, self.args.version)
    #     os.makedirs(result_dir, exist_ok=True)
    #     # 设置模型为评估模式
    #     self.net.eval()
    #     net = self.net.module if self.args.dp else self.net
    #     print('\n' + '=' * 20)

    #     # 加载测试数据集
    #     test_dataset = AccelDataset(self.args)
    #     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 假设一次处理一个文件

    #     y_true, y_pred = [], []

    #     for x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel, label in test_loader:
    #         # 将数据移动到指定设备
    #         x_xwav, x_ywav, x_zwav = x_xwav.to(self.args.device), x_ywav.to(self.args.device), x_zwav.to(self.args.device)
    #         x_xmel, x_ymel, x_zmel = x_xmel.to(self.args.device), x_ymel.to(self.args.device), x_zmel.to(self.args.device)
    #         label = label.to(self.args.device)

    #         # 将 x_xmel, x_ymel, x_zmel 合并，形成模型输入
    #         with torch.no_grad():
    #             predict_ids, _ = net(x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel, label)
    #             pred = torch.argmax(predict_ids, dim=1).cpu().item()
    #             y_pred.append(pred)
    #             y_true.append(label.cpu().item())
        
    #     # 计算TP, TN, FP, FN
    #     TP = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    #     TN = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
    #     FP = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    #     FN = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)

    #     # 计算 TPR 和 TNR
    #     TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    #     TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    #     # 计算 Acc_balanced
    #     acc_balanced = 0.5 * (TPR + TNR)
    
    #     # 打印和保存结果
    #     self.logger.info(f'Acc_balanced: {acc_balanced:.3f}')
    #     csv_lines.append(['Acc_balanced', acc_balanced])
    #     if save:
    #         result_path = os.path.join(result_dir, 'result.csv')
    #         utils.save_csv(result_path, csv_lines)

    #     return acc_balanced

    # 用于评估模型，对所有测试文件计算异常分数并保存到指定目录
    def evaluator(self, save=True, gmm_n=False):
        result_dir = os.path.join('./evaluator/teams', self.args.version)
        if gmm_n:
            result_dir = os.path.join('./evaluator/teams', self.args.version + f'-gmm-{gmm_n}')
        os.makedirs(result_dir, exist_ok=True)

        self.net.eval()
        net = self.net.module if self.args.dp else self.net
        print('\n' + '=' * 20)
        for index, (target_dir, train_dir) in enumerate(zip(sorted(self.args.test_dirs), sorted(self.args.add_dirs))):
            machine_type = target_dir.split('/')[-2]
            # get machine list
            machine_id_list = utils.get_machine_id_list(target_dir)
            for id_str in machine_id_list:
                meta = machine_type + '-' + id_str
                label = self.args.meta2label[meta]
                test_files = utils.get_filename_list(target_dir, pattern=f'{id_str}*')
                csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{id_str}.csv')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
                if gmm_n:
                    train_files = utils.get_filename_list(train_dir, pattern=f'normal_{id_str}*')
                    features = self.get_latent_features(train_files)
                    means_init = net.arcface.weight[label * gmm_n: (label + 1) * gmm_n, :].detach().cpu().numpy() \
                        if self.args.use_arcface and (gmm_n == self.args.sub_center) else None
                    # means_init = None
                    gmm = self.fit_GMM(features, n_components=gmm_n, means_init=means_init)
                for file_idx, file_path in enumerate(test_files):
                    x_wav, x_mel, label = self.transform(file_path)
                    x_wav, x_mel = x_wav.unsqueeze(0).float().to(self.args.device), x_mel.unsqueeze(0).float().to(
                        self.args.device)
                    label = torch.tensor([label]).long().to(self.args.device)
                    with torch.no_grad():
                        predict_ids, feature = net(x_wav, x_mel, label)
                    if gmm_n:
                        if self.args.use_arcface: feature = F.normalize(feature).cpu().numpy()
                        y_pred[file_idx] = - np.max(gmm._estimate_log_prob(feature))
                    else:
                        probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                        y_pred[file_idx] = probs[label]
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                if save:
                    utils.save_csv(csv_path, anomaly_score_list)

    # 遍历训练文件，将潜在特则会给你提取并归一化
    def get_latent_features(self, train_files):
        pbar = tqdm(enumerate(train_files), total=len(train_files))
        self.net.eval()
        classifier = self.net.module if self.args.dp else self.net
        features = []
        for file_idx, file_path in pbar:
            x_wav, x_mel, label = self.transform(file_path)
            x_wav, x_mel = x_wav.unsqueeze(0).float().to(self.args.device), x_mel.unsqueeze(0).float().to(
                self.args.device)
            label = torch.tensor([label]).long().to(self.args.device)
            with torch.no_grad():
                _, feature, _ = classifier(x_wav, x_mel, label)
            if file_idx == 0:
                features = feature.cpu()
            else:
                features = torch.cat((features.cpu(), feature.cpu()), dim=0)
        if self.args.use_arcface: features = F.normalize(features)
        return features.numpy()


    # fit_GMM方法，使用sklearn中的GaussianMixture训练GMM，并使用训练集的潜在特征
    def fit_GMM(self, data, n_components, means_init=None):
        print('=' * 40)
        print('Fit GMM in train data for test...')
        np.random.seed(self.args.seed)
        gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                              means_init=means_init, reg_covar=1e-3, verbose=2)
        gmm.fit(data)
        print('Finish GMM fit.')
        print('=' * 40)
        return gmm