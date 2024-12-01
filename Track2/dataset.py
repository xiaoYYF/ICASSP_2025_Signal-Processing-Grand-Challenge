import json
import torch
from torch.utils.data import Dataset
import numpy as np
import utils  # 假设 utils 中有一个用于转换为 Mel 频谱图的 Wave2Mel 类
import os

def generate_event_info(event_list):
    length = len(event_list)
    is_middle_frame = [0] * length
    event_start_index = [-1] * length
    event_end_index = [-1] * length

    current_event_start = -1
    current_event_value = None

    for i in range(length):
        if current_event_value is None or event_list[i] != current_event_value:
            if current_event_start != -1:
                event_length = i - current_event_start
                middle_index = current_event_start + (event_length // 2)
                if event_length % 2 == 0:
                    middle_index -= 1  # Choose the closest to the middle for even length
                for j in range(current_event_start, i):
                    event_start_index[j] = current_event_start
                    event_end_index[j] = i - 1
                    if j == middle_index:
                        is_middle_frame[j] = 1
            current_event_start = i
            current_event_value = event_list[i]
        event_end_index[i] = i  # Update as we move through the event

    # Handle the last event if it ends at the last element
    if current_event_start != -1:
        event_length = length - current_event_start
        middle_index = current_event_start + (event_length // 2)
        if event_length % 2 == 0:
            middle_index -= 1  # Choose the closest to the middle for even length
        for j in range(current_event_start, length):
            event_start_index[j] = current_event_start
            event_end_index[j] = length - 1
            if j == middle_index:
                is_middle_frame[j] = 1

    return is_middle_frame, event_start_index, event_end_index

class AccelDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        """
        初始化数据集对象，加载数据并设置必要的预处理参数。
        """
        # 从 args 中获取 JSON 文件路径
        self.file_path = args.train_json if hasattr(args, 'train_json') else None
        if not self.file_path:
            raise ValueError("JSON file path is not provided in args.")

        # 读取并加载 JSON 文件内容
        with open(self.file_path, 'r') as f:
            self.data = json.load(f)  # `self.data` 是一个包含多个数据项的列表

        self.args = args

        # 初始化 Wave2Mel，用于将波形转换为 Mel 频谱图
        self.wav2mel = utils.Wave2Mel(
            sr=args.sr, power=args.power, n_fft=args.n_fft,
            n_mels=args.n_mels, win_length=args.win_length,
            hop_length=args.hop_length
        )

    def __len__(self):
        """
        返回数据集中数据项的数量。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据项，包括加速度波形数据和标签。
        """
        # 获取数据项
        data_item = self.data[idx]
        
        # 提取加速度数据和标签
        accel_data = np.array(data_item["accel"])  # Shape: (num_samples, 3)
        label = data_item["labels"]
        label = torch.tensor(label)


        # 分别提取 x、y、z 轴的加速度数据，并转换为 PyTorch 张量
        x_xwav = torch.from_numpy(accel_data[:, 0]).float()  # X-axis waveform

        x_ywav = torch.from_numpy(accel_data[:, 1]).float()  # Y-axis waveform
        x_zwav = torch.from_numpy(accel_data[:, 2]).float()  # Z-axis waveform
        # 使用 Wave2Mel 将波形转换为 Mel 频谱图
        x_xmel = self.wav2mel(x_xwav)  # Shape: [mel_bins, time_frames]
        x_ymel = self.wav2mel(x_ywav)  # Shape: [mel_bins, time_frames]
        x_zmel = self.wav2mel(x_zwav)  # Shape: [mel_bins, time_frames]
        
        # 返回 x、y、z 轴的 Mel 频谱图、波形数据和标签
        return x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel,label


class AccelDataset_test(torch.utils.data.Dataset):
    def __init__(self, args):
        """
        初始化数据集对象，加载数据并设置必要的预处理参数。
        """
        # 从 args 中获取 JSON 文件路径
        self.file_path = args.validation_json

        if not self.file_path:
            raise ValueError("JSON file path is not provided in args.")

        # 读取并加载 JSON 文件内容
        with open(self.file_path, 'r') as f:
            self.data = json.load(f)  # `self.data` 是一个包含多个数据项的列表

        self.args = args

        # 初始化 Wave2Mel，用于将波形转换为 Mel 频谱图
        self.wav2mel = utils.Wave2Mel(
            sr=args.sr, power=args.power, n_fft=args.n_fft,
            n_mels=args.n_mels, win_length=args.win_length,
            hop_length=args.hop_length
        )

    def __len__(self):
        """
        返回数据集中数据项的数量。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据项，包括加速度波形数据和标签。
        """
        # 获取数据项
        data_item = self.data[idx]

        # 提取加速度数据和标签
        accel_data = np.array(data_item["accel"])  # Shape: (num_samples, 3)

        ts = data_item["ts"]
        subject = data_item["subject"]

        # 分别提取 x、y、z 轴的加速度数据，并转换为 PyTorch 张量
        x_xwav = torch.from_numpy(accel_data[:, 0]).float()  # X-axis waveform

        x_ywav = torch.from_numpy(accel_data[:, 1]).float()  # Y-axis waveform
        x_zwav = torch.from_numpy(accel_data[:, 2]).float()  # Z-axis waveform
        # 使用 Wave2Mel 将波形转换为 Mel 频谱图
        x_xmel = self.wav2mel(x_xwav)  # Shape: [mel_bins, time_frames]
        x_ymel = self.wav2mel(x_ywav)  # Shape: [mel_bins, time_frames]
        x_zmel = self.wav2mel(x_zwav)  # Shape: [mel_bins, time_frames]

        # 返回 x、y、z 轴的 Mel 频谱图、波形数据和标签
        return x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel, ts , subject