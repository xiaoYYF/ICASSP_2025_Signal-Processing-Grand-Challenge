# torch：PyTorch的核心库，用于构建和训练深度学习模型
import torch
# torch.nn：包含神经网络层和常用的损失函数的模块
import torch.nn as nn


# ASDLoss继承自nn.Module类，这意味着是PyTorch中的一个神经网络模块，可以直接在模型训练过程中作为损失函数使用
class ASDLoss(nn.Module):
    # 在__init__()方法中，通过super(ASDLoss,self).__init__()调用父类nn.Module的构造函数，初始化该类的基础属性
    def __init__(self):
        super(ASDLoss, self).__init__()
        # self.ce = nn.CrossEntropyLoss(),在构造函数中初始化了一个交叉熵损失函数(CrossEntropyLoss)，并将其赋值给self.ce。
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    # forward()方法定义了ASDLoss类在前向传播(即损失计算)中的行为
    def forward(self, logits, labels): # logits:模型的输出，通常是未经过softmax的分类得分，形状一般为[batch_size,num_classed]，表示每个样本对每个类别的得分   labels:真实标签，形状一般为[batch_size],表示每个样本的真实类别，每个标签是一个整数，表示相应类别的索引。
        # 计算交叉熵损失，self.ce是在__init__()方法中初始化的交叉熵损失函数，直接接受logits和labels作为输入
        # 交叉熵损失函数会自动将logits转换为概率分布(通过softmax)，并与真实标签进行对比计算损失值
        loss = self.mse(logits, labels)
        # 返回
        return loss