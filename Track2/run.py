# 用于文件路径操作
import os
# 用于时间管理
import time
# 用于命令行参数解析
import argparse

# 导入PyTorch库，用于深度学习的张良操作
import torch
# PyTorch中用于批量加载数据的类
import torch.nn as nn
# DataLoader，PyTorch中用于批量加载数据的类
from torch.utils.data import DataLoader
# SummaryWriter，用于记录训练过程中的指标和损失，以便通过TensorBoard可视化
from torch.utils.tensorboard import SummaryWriter

# 从net模块导入的模型架构
from net import STgramMFN
# 训练器类，封装了训练和测试的过程
from trainer import Trainer
# 数据集类，用于加载和预处理数据
from dataset import AccelDataset
# 实用工具模块，包含辅助功能（例如日志管理和配置文件处理）
import utils
import json

sep = os.sep

def main(args):
    # set random seed
    # 设置随机种子，以确保实验的可重复性
    utils.setup_seed(args.random_seed)
    # set device
    # 根据用户指定的cuda和device_ids参数选择设备（CPU或GPU）并判断是否需要数据并行（多GPU）
    cuda = args.cuda
    device_ids = args.device_ids
    args.dp = False
    if not cuda or device_ids is None:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{device_ids[0]}')
        if len(device_ids) > 1: args.dp = True
    # load data
    # train_dirs = args.train_dirs
    # 加载训练数据，将训练数据的文件路径存储在train_file_list中，并使用AccelDataset类加载数据集，再使用DataLoader创建数据迭代器
    
    # args.meta2label, args.label2meta = utils.metadata_to_label(train_dirs)

    # train_file_list = []
    # for train_dir in train_dirs:
    #    train_file_list.extend(utils.get_filename_list(train_dir))
    # train_dataset = AccelDataset(args, train_file_list, load_in_memory=False)
    # 打开并加载 JSON 文件
    train_dataset = AccelDataset(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
    # set model
    # 初始化模型：使用STgramMFN作为主模型，根据参数决定是否使用Arcface损失，并将模型移到指定设备上
    args.num_classes = len(args.meta2label.keys())
    args.logger.info(f'Num classes: {args.num_classes}')
    net = STgramMFN(num_classes=args.num_classes, use_arcface=args.use_arcface,
                    m=float(args.m), s=float(args.s), sub=args.sub_center)
    if args.dp:
        net = nn.DataParallel(net, device_ids=args.device_ids)
    net = net.to(args.device)
    # optimizer & scheduler
    # 定义优化器和学习率调度器，使用Adam优化器和余弦退货调度器
    optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr))
    # optimizer = torch.optim.SGD(net.parameters(), lr=float(args.lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.1*float(args.lr))
    # trainer
    # 创建Trainer对象，用于管理训练和测试的过程
    trainer = Trainer(args=args,
                      net=net,
                      optimizer=optimizer,
                      scheduler=scheduler)
    # train model
    # 训练模型：如果参数中没有指定加载的模型检查点，则执行模型的训练过程
    trainer.train(train_dataloader)
    # if not args.load_epoch:
    #     trainer.train(train_dataloader)
    # # 测试模型：加载保存的模型权重并运行测试，同时保存评估结果
    # # test model
    # load_epoch = args.load_epoch if args.load_epoch else 'best'
    # # model_path = os.path.join(args.writer.log_dir, 'model', f'{load_epoch}_checkpoint.pth.tar')
    # model_path = "/home/competition/2024ICASSPGC-8/ICASSP_GC-8_3090/runs/STgram-MFN(m=0.7,s=30)/model/best_checkpoint.pth.tar"
    # if args.dp:
    #     trainer.net.module.load_state_dict(torch.load(model_path)['model'])
    # else:
    #     trainer.net.load_state_dict(torch.load(model_path)['model'])
    trainer.test(save=True)
    # trainer.evaluator(save=True, gmm_n=args.gmm_n)


# 负责初始化并启动整个训练和测试流程
def run():
    # 加载配置文件config.yaml中的参数，并使用argparse解析命令行参数
    # init config parameters
    # 下面代码，utils.load_yaml函数读取了config.yaml文件的内容，并将其解析为字典params。然后使用argparse库将这些参数添加到命令行参数解析器中，生成args对象。
    params = utils.load_yaml(file_path='./config.yaml')
    parser = argparse.ArgumentParser(description=params['description'])
    for key, value in params.items():
        parser.add_argument(f'--{key}', default=value, type=utils.set_type)
    args = parser.parse_args()
    # 初始化日志和TensorBoard写入器，并设置日志文件目录
    # init logger and writer
    time_str = time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))
    args.version = f'STgram-MFN(m={args.m},s={args.s})'
    args.version = f'{time_str}-{args.version}' if not args.load_epoch and args.time_version else args.version
    log_dir = f'runs/{args.version}'
    writer = SummaryWriter(log_dir=log_dir)
    logger = utils.get_logger(filename=os.path.join(log_dir, 'running.log'))
    # 保存模型版本文件、配置文件，并调用main函数运行整个训练和测试过程
    # save version files
    if args.save_version_files: utils.save_load_version_files(log_dir, args.save_version_file_patterns, args.pass_dirs)
    # run
    args.writer, args.logger = writer, logger
    args.logger.info(args.version)
    main(args)
    # save config file
    utils.save_yaml_file(file_path=os.path.join(log_dir, 'config.yaml'), data=vars(args))

# 脚本的入口，确保run()函数仅在直接执行脚本时被调用，而不是在导入模块时被调用
if __name__ == '__main__':
    run()
