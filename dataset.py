import numpy as np
import os
import librosa
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from config import SR, MAX_SEC, MAX_LEN, N_MELS, N_MFCC, FEATURE_TYPE

# grab utterance id and label list from protocol
def parse_protocol(protocol_path):
    utt_list, label_list = [], []
    with open(protocol_path, 'r') as f:
        for line in f:
            entry = line.strip().split()
            utt_id = entry[1]
            label = entry[-1].lower()
            utt_list.append(utt_id)
            label_list.append(1 if label == 'bonafide' else 0)
    return utt_list, label_list

# same thing but keep labels as strings
def parse_protocol_with_strings(protocol_path):
    utt_list, label_list = [], []
    with open(protocol_path, 'r') as f:
        for line in f:
            entry = line.strip().split()
            utt_id = entry[1]
            label = entry[-1].lower()
            utt_list.append(utt_id)
            label_list.append(label)
    return utt_list, label_list

# just load and pad/cut the audio
def load_audio(filepath):
    audio, _ = librosa.load(filepath, sr=SR)
    if len(audio) < MAX_LEN:
        audio = np.pad(audio, (0, MAX_LEN-len(audio)))
    else:
        audio = audio[:MAX_LEN]
    return audio

# extract features based on config (if u choose mfcc it supports that as well)
def extract_feature(audio):
    if FEATURE_TYPE == 'mfcc':
        features = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC, n_fft=512, hop_length=160, n_mels=N_MELS)
    elif FEATURE_TYPE == 'logmel':
        melspec = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=512, hop_length=160, n_mels=N_MELS)
        features = librosa.power_to_db(melspec, ref=np.max)
    else:
        raise NotImplementedError()
    return features.astype(np.float32)

# main dataset for training
class ASVspoofDataset(Dataset):
    def __init__(self, flac_dir, protocol_path, normalize=True, transform=None):
        self.utt_list, self.label_list = parse_protocol(protocol_path)
        self.dir = flac_dir
        self.transform = transform
        self.normalize = normalize
        self.mean = None
        self.std = None
        if self.normalize:
            self._compute_norm_params()

    # estimate normalization for first 500 utterances
    def _compute_norm_params(self):
        print('[INFO] Computing normalization statistics (first 500 utterances)')
        feats = []
        for utt_id in tqdm(self.utt_list[:500]):
            wav_path = os.path.join(self.dir, utt_id + '.flac')
            audio = load_audio(wav_path)
            feat = extract_feature(audio)
            feats.append(feat)
        feats = np.stack(feats)
        self.mean = np.mean(feats, axis=(0,2), keepdims=True)
        self.std = np.std(feats, axis=(0,2), keepdims=True) + 1e-9

    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, idx):
        utt_id = self.utt_list[idx]
        label = self.label_list[idx]
        wav_path = os.path.join(self.dir, utt_id + '.flac')
        audio = load_audio(wav_path)
        feat = extract_feature(audio)
        if (self.normalize and self.mean is not None and self.std is not None):
            feat = (feat - self.mean[0,:,0][:, None]) / self.std[0,:,0][:, None]
        if self.transform is not None:
            feat = self.transform(feat)
        return torch.tensor(feat).unsqueeze(0), torch.tensor(label, dtype=torch.float32)

# similar but for dev/val
class ASVspoofDevDataset(ASVspoofDataset):
    def __init__(self, flac_dir, protocol_path, normalize=True, transform=None):
        self.utt_list, self.label_list = parse_protocol(protocol_path)
        self.dir = flac_dir
        self.transform = transform
        self.normalize = normalize
        self.mean = None
        self.std = None

    def set_norm(self, mean, std):
        self.mean = mean
        self.std = std