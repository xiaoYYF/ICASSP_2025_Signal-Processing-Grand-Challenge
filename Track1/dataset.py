import json
import torch
from torch.utils.data import Dataset
import numpy as np
import utils
import os


class AccelDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        """
        Initialize dataset, load data, and set preprocessing parameters.
        """
        # Get JSON file path
        self.file_path = args.train_json if hasattr(args, 'train_json') else None
        if not self.file_path:
            raise ValueError("JSON file path is not provided.")

        # Load data from JSON file
        with open(self.file_path, 'r') as f:
            self.data = json.load(f)  # List of data items

        self.args = args

        # Initialize Wave2Mel for waveform to Mel spectrogram conversion
        self.wav2mel = utils.Wave2Mel(
            sr=args.sr, power=args.power, n_fft=args.n_fft,
            n_mels=args.n_mels, win_length=args.win_length,
            hop_length=args.hop_length
        )

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get data item at index, including acceleration waveform and label.
        """
        # Get the data item
        data_item = self.data[idx]

        # Extract acceleration data
        accel_data = np.array(data_item["accel"])  # Shape: (num_samples, 3)
        label = data_item["label"]

        # Convert x, y, z axis waveforms to PyTorch tensors
        x_xwav = torch.from_numpy(accel_data[:, 0]).float()  # X-axis waveform
        x_ywav = torch.from_numpy(accel_data[:, 1]).float()  # Y-axis waveform
        x_zwav = torch.from_numpy(accel_data[:, 2]).float()  # Z-axis waveform

        # Convert waveforms to Mel spectrograms
        x_xmel = self.wav2mel(x_xwav)  # Shape: [mel_bins, time_frames]
        x_ymel = self.wav2mel(x_ywav)  # Shape: [mel_bins, time_frames]
        x_zmel = self.wav2mel(x_zwav)  # Shape: [mel_bins, time_frames]

        # Return waveforms and Mel spectrograms
        return x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel, label
