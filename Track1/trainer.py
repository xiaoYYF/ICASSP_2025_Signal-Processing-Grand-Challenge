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

# 从sklearn导入GaussianMixture类，用于高斯混合模型（GMM）的实现
from sklearn.mixture import GaussianMixture
# loss模块中的自定义损失函数ASDLoss
from loss import ASDLoss
# 包含的辅助函数utils模块
import utils

from torch.utils.data import DataLoader
from dataset import AccelDataset


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
        for epoch in range(0, epochs + 1):
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
                x_xwavs, x_ywavs, x_zwavs = (
                    x_xwavs.float().to(self.args.device),
                    x_ywavs.float().to(self.args.device),
                    x_zwavs.float().to(self.args.device),
                )
                x_xmels, x_ymels, x_zmels = (
                    x_xmels.float().to(self.args.device),
                    x_ymels.float().to(self.args.device),
                    x_zmels.float().to(self.args.device),
                )
                # labels重塑为一维并转换为长整型（long()）以适应分类任务
                labels = labels.reshape(-1).long().to(self.args.device)
                # print(x_wavs.shape, x_mels.shape, labels.shape)
                # 前向传播，将输入传递到模型self.net,获得logits（模型输出）
                # print("shape of x_xwavs:",x_xwavs.shape)
                # print("shape of x_xmels:",x_xmels.shape)
                logits, _ = self.net(x_xwavs, x_ywavs, x_zwavs, x_xmels, x_ymels, x_zmels, labels)
                # 使用损失函数计算损失loss，表示logits与真实标签labels之间的差异
                loss = self.criterion(logits, labels)
                # 使用tqdm显示当前损失值，更新进度条上的损失信息
                train_bar.set_postfix(loss=f'{loss.item():.5f}')
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
            if (epoch - start_valid_epoch) % valid_every_epochs == 0 and epoch >= start_valid_epoch:
                # 调用test函数获取avg_auc和avg_pauc，并记录AUC和pAUC以便监控模型性能
                # avg_auc, avg_pauc = self.test(save=False, gmm_n=False)
                acc_balanced = self.test(save=False)
                self.writer.add_scalar(f'auc', acc_balanced, epoch)
                # 如果当前性能(avg_auc+avg_pauc)优于最佳性能best_metric，则更新最佳性能，重置no_better_epoch
                if acc_balanced >= best_metric:
                    no_better_epoch = 0
                    best_metric = acc_balanced
                    # 保存当前最佳模型的状态字典(state_dict)到best_model_path
                    best_model_path = os.path.join(model_dir, 'best_checkpoint.pth.tar')
                    utils.save_model_state_dict(best_model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)
                    self.logger.info(f'Best epoch now is: {epoch:4d}')
                # 否则，no_better_epoch增加1，记录连续无提升的轮次数
                else:
                    # early stop
                    no_better_epoch += 1
                    # 如果连续无提升的轮次数超过early_stop_epochs，则停止训练，并记录日志信息
                    if no_better_epoch > early_stop_epochs > 0: break
            # save last 10 epoch state dict
            # 当epoch大于等于start_save_model_epochs时，模型每隔一定的save_model_interval_epochs进行保存
            if epoch >= self.args.start_save_model_epochs:
                if (epoch - self.args.start_save_model_epochs) % self.args.save_model_interval_epochs == 0:
                    # 根据条件创建模型文件路径并保存模型状态字典
                    model_path = os.path.join(model_dir, f'{epoch}_checkpoint.pth.tar')
                    utils.save_model_state_dict(model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)

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
        print('\n' + '=' * 20)

        # 加载测试数据集
        test_dataset = AccelDataset(self.args)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 假设一次处理一个文件

        y_true, y_pred = [], []

        for idx, (x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel, label) in enumerate(test_loader):
            # 将数据移动到指定设备
            x_xwav, x_ywav, x_zwav = x_xwav.to(self.args.device), x_ywav.to(self.args.device), x_zwav.to(
                self.args.device)
            x_xmel, x_ymel, x_zmel = x_xmel.to(self.args.device), x_ymel.to(self.args.device), x_zmel.to(
                self.args.device)

            # 前向传播，预测结果
            with torch.no_grad():
                predict_ids, _ = net(x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel, None)
                predicted_label = torch.argmax(predict_ids, dim=1).cpu().item()  # 获取预测的标签 (0 或 1)
                y_pred.append(predicted_label)
                y_true.append(label.cpu().item())

            # 获取 chunk_id
            chunk_id = idx  # 或从文件名解析得到，如 os.path.basename(file_path)

            # 将结果保存到列表
            csv_output.append([chunk_id, predicted_label])

        # 计算TP, TN, FP, FN
        TP = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
        TN = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
        FP = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
        FN = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)

        # 计算 TPR 和 TNR
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0

        # 计算 Acc_balanced
        acc_balanced = 0.5 * (TPR + TNR)

        # 打印和保存结果
        self.logger.info(f'Acc_balanced: {acc_balanced:.3f}')

        # 保存到 CSV 文件
        if save:
            csv_path = os.path.join(result_dir, 'test_predictions.csv')
            with open(csv_path, 'w') as f:
                f.write("chunk_id,label\n")  # 写入标题
                for chunk_id, label in csv_output:
                    f.write(f"{chunk_id},{label}\n")

            # 打印日志
            self.logger.info(f"Test results saved to {csv_path}")

        return acc_balanced

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