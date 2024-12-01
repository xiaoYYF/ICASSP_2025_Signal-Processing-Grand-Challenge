import os
import torch
import argparse
from torch.utils.data import DataLoader
from dataset import AccelDataset  # 假设你已有 AccelDataset 类
from net import STgramMFN  # 假设你已有 STgramMFN 类
import utils  # 包含日志或其他工具函数

class Tester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self._setup_model()
        self.logger = utils.get_logger(filename=os.path.join(self.args.result_dir, 'test.log'))

    def _setup_model(self):
        # 初始化模型
        self.net = STgramMFN(num_classes=len(self.args.meta2label), use_arcface=self.args.use_arcface,
                             m=self.args.m, s=self.args.s, sub=self.args.sub_center)
        if self.args.device_ids and len(self.args.device_ids) > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=self.args.device_ids)
        self.net = self.net.to(self.device)

        # 加载模型权重
        model_path = os.path.join("/home/competition/2024ICASSPGC-8/ICASSP_GC-8_3090/runs/STgram-MFN(m=0.7,s=30) copy/model/best_checkpoint.pth.tar")
        if self.args.device_ids and len(self.args.device_ids) > 1:
            self.net.module.load_state_dict(torch.load(model_path)['model'])
        else:
            self.net.load_state_dict(torch.load(model_path)['model'])

    def test(self, save=False):
        """
        测试模型在测试集上的性能，并输出 chunk_id 和预测的 label 到 CSV 文件。
        """
        # 初始化列表，用于存储预测结果
        csv_output = []
        result_dir = os.path.join(self.args.result_dir, 'test_results')
        os.makedirs(result_dir, exist_ok=True)
        
        # 设置模型为评估模式
        self.net.eval()
        net = self.net.module if self.args.device_ids and len(self.args.device_ids) > 1 else self.net
        print('\n' + '=' * 20)

        # 加载测试数据集
        test_dataset = AccelDataset(args)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


        for idx, (x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel) in enumerate(test_loader):
            # 将数据移动到指定设备
            x_xwav, x_ywav, x_zwav = x_xwav.to(self.device), x_ywav.to(self.device), x_zwav.to(self.device)
            x_xmel, x_ymel, x_zmel = x_xmel.to(self.device), x_ymel.to(self.device), x_zmel.to(self.device)

            # 前向传播，预测结果
            with torch.no_grad():
                predict_ids, _ = net(x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel, None)
                predicted_label = torch.argmax(predict_ids, dim=1).cpu().item()  # 获取预测的标签 (0 或 1)
            
            # 获取 chunk_id
            chunk_id = idx  # 假设用索引作为 ID，也可以从文件名解析
            
            # 将结果保存到列表
            csv_output.append([chunk_id, predicted_label])

        # 保存到 CSV 文件
        if save:
            csv_path = os.path.join(result_dir, 'test_predictionscopy.csv')
            with open(csv_path, 'w') as f:
                f.write("chunk_id,label\n")  # 写入标题
                for chunk_id, label in csv_output:
                    f.write(f"{chunk_id},{label}\n")

            # 打印日志
            self.logger.info(f"Test results saved to {csv_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Test Script")
    # 从配置文件加载参数
    params = utils.load_yaml(file_path='./config.yaml')
    for key, value in params.items():
        parser.add_argument(f'--{key}', default=value, type=utils.set_type)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tester = Tester(args)
    tester.test(save=True)
