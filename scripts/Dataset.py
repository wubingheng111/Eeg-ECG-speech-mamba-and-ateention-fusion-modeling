from datasets import load_from_disk
import torch
from torch.utils.data import Dataset
import numpy as np
import math
import logging

logging.basicConfig(level=logging.INFO)

class Merged_Dataset(Dataset):
    def __init__(self, dataset_path, split, max_length=1024):
        self.dataset = load_from_disk(dataset_path)[split]
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        # eeg 特征
        eeg_signals = self.dataset[idx]["ecgsignals"][:self.max_length]
        eeg_signals = np.array(eeg_signals).flatten()[::4]  # 展平 ecgsignals
        eeg_ids = torch.tensor(self.float_list_2_int_list(eeg_signals), dtype=torch.long)

        # ecg 特征
        ecg_signals = self.dataset[idx]["ecgsignals"][:self.max_length]
        ecg_signals = np.array(ecg_signals).flatten()[::4]
        ecg_ids = torch.tensor(self.float_list_2_int_list(ecg_signals), dtype=torch.long)

        # speech 特征
        speech_signals = self.dataset[idx]["mel_spectrogram"][:self.max_length]
        speech_signals = np.array(speech_signals).flatten()[::4]
        speech_ids = torch.tensor(self.float_list_2_int_list(speech_signals), dtype=torch.long)
 
        # 如果 eeg_ids 的长度大于 max_length，那么截断 eeg_ids
        if eeg_ids.size(0) > self.max_length:
            eeg_ids = eeg_ids[:self.max_length]
        else:
            # 向左填充
            eeg_ids = torch.nn.functional.pad(eeg_ids, (0, self.max_length - eeg_ids.size(0)), mode='constant', value=0)

        # 如果 ecg_ids 的长度大于 max_length，那么截断 ecg_ids
        if ecg_ids.size(0) > self.max_length:
            ecg_ids = ecg_ids[:self.max_length]
        else:
            # 向左填充
            ecg_ids = torch.nn.functional.pad(ecg_ids, (0, self.max_length - ecg_ids.size(0)), mode='constant', value=0)

        # 如果 speech_ids 的长度大于 max_length，那么截断 speech_ids
        if speech_ids.size(0) > self.max_length:
            speech_ids = speech_ids[:self.max_length]
        else:
            # 向左填充
            speech_ids = torch.nn.functional.pad(speech_ids, (0, self.max_length - speech_ids.size(0)), mode='constant', value=0)

        # 类别标签
        labels = torch.tensor(self.dataset[idx]["label"], dtype=torch.long)

        return {
            "eeg_input_ids": eeg_ids,
            "ecg_input_ids": ecg_ids,
            "speech_input_ids": speech_ids,
            "labels": labels
        }
    
    def float_to_int(self, x):
        bins = np.linspace(0, 1, 4096)
        discrete = np.digitize(x, bins)
        return discrete

    def float_list_2_int_list(self, x):
        return [self.float_to_int(i) for i in x]
    
