import os
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import librosa
import soundfile

class Dataset(Dataset):

    def __init__(self, split="train", device="cpu", length_distribution="lists/length_distribution.txt", info="lists/video_info.txt", max_len=1500, audio_feature_len = 512, video_feature_len = 1280, own_feature=False, both_feature=False):

        if split == 'train':
            self.filename = 'lists/train_list.txt'
        else:
            self.filename = 'lists/validation_list.txt'

        self.info = {}
        for line in open(info, 'r', encoding='utf-8'):
            line = line.strip().split('@')
            name = line[0]
            A = line[1].split('#')
            B = [int(x) for x in A]
            self.info[name] = B

        self.scenes = [] # List of(name, l, r, 0/1)
        for line in open(self.filename, 'r', encoding='utf-8'):
            line = line.strip().split('@')
            name = line[0]
            A = line[1].split('#')
            r = 0
            for x in A:
                x = x.split('-')
                l = int(x[0])
                if l - r > 40:
                    self.scenes.append((name, r, l, 0))
                r = int(x[1])
                self.scenes.append((name, l, r, 1))
            total_f = self.total_frame_num(name)
            if total_f - r > 40:
                self.scenes.append((name, r, total_f, 0))
        
        self.distribution = [int(line.strip()) for line in open(length_distribution, 'r', encoding='utf-8')]

        self.max_len = max_len
        self.audio_feature_len = audio_feature_len
        self.video_feature_len = video_feature_len
        self.feature_len = (self.audio_feature_len + self.video_feature_len * 2) if both_feature else (self.audio_feature_len + self.video_feature_len)

        self.device = device
        self.split = split
        self.own_feature = own_feature
        self.both_feature = both_feature
    
    def total_frame_num(self, name):
        sum = 0
        for x in self.info[name]:
            sum += x
        return sum

    def __len__(self):
        return len(self.scenes)

    def get_sample_len(self, len, label):
        if label == 0:
            L = random.sample(self.distribution, 1)[0]
            len = min(len, L)
        len = len * 4 // 5
        len = min(len, self.max_len)
        return len

    def audio_feature(self, name, l, r):
        f_n = self.info[name]
        path = '/mnt/sda1/tangzitian/FootballHighlights/games/'+name+'/'+name
        x = 0
        A = []
        for i in range(len(f_n)):
            if x >= r:
                break
            y = x + f_n[i]
            if y > l:
                if i<10:
                    filename = path+'[0%d].npy'%i
                else:
                    filename = path+'[%d].npy'%i
                a = np.load(filename)
                left = (max(l, x) - x) * 1024
                right = (min(r, y) - x) * 1024
                A.append(a[left: right])
            x = y
        A = np.concatenate(A)
        feature = librosa.feature.melspectrogram(A, 25600, n_fft=4096, hop_length=5120, n_mels=self.audio_feature_len).transpose()[:(r-l)//5, :]
        return torch.from_numpy(feature)

    def video_feature(self, name, l, r):
        f_n = self.info[name]
        if self.own_feature:
            path = '/mnt/sda1/songzimeng/GameHighlight_feature/'+name+'/'+name
        else:
            path = '/mnt/sda1/songzimeng/Classifier_feature/'+name+'/'+name
        L = (r - l) // 5
        l = (l + 4) // 5
        r = l + L
        x = 0
        sum = 0
        A = []
        for i in range(len(f_n)):
            if x >= r:
                break
            sum += f_n[i]
            y = (sum + 4) // 5
            if y > l:
                if i<10:
                    filename = path+'[0%d].pt'%i
                else:
                    filename = path+'[%d].pt'%i
                a = torch.load(filename)
                left = max(l, x) - x
                right = min(r, y) - x
                A.append(a[left: right, :])
            x = y
        feature = torch.cat(A, dim = 0)
        return feature

    # @profile
    def __getitem__(self, index):
        name, l, r, label = self.scenes[index]
        len = self.get_sample_len(r-l, label)
        start = random.randint(l, r-len)
        if self.both_feature:
            a_feature = self.audio_feature(name, start, start+len) / 10
            self.own_feature = False
            v_feature1 = self.video_feature(name, start, start+len)
            self.own_feature = True
            v_feature2 = self.video_feature(name, start, start+len)
            feature = torch.cat((a_feature, v_feature1, v_feature2), dim = -1)
        elif self.video_feature_len==0:
            feature = self.audio_feature(name, start, start+len)
        elif self.audio_feature_len==0:
            feature = self.video_feature(name, start, start+len)
        else:
            a_feature = self.audio_feature(name, start, start+len) / 10
            v_feature = self.video_feature(name, start, start+len)
            feature = torch.cat((a_feature, v_feature), dim = -1)

        return {
            "length": len//5,
            "x": feature,
            "label": label
        }


    # @profile
    def collate_fn(self, samples):
        lens = [sample["length"] for sample in samples]
        max_len = max(lens)
        bsz = len(lens)

        x = torch.FloatTensor(bsz, max_len, self.feature_len).fill_(0.0)
        label = []

        for idx, sample in enumerate(samples):
            x[idx, 0: sample["length"], :] = sample["x"]
            label.append(sample["label"])
        
        return {
            "length": torch.LongTensor(lens).to(self.device),
            "x": x.to(self.device),
            "label": torch.LongTensor(label).to(self.device)
        }

class DatasetSegmentation(Dataset):

    def __init__(self, split="train", device="cpu", length_distribution="lists/length_distribution.txt", info="lists/video_info.txt", max_len=1500, transformation=lambda x:x, own_feature=False, both_feature=False):

        if split == 'train':
            self.filename = 'lists/train_list.txt'
        elif split == 'dev':
            self.filename = 'lists/validation_list.txt'
        else:
            self.filename = 'lists/tmp.txt'
        
        self.own_feature = own_feature
        self.both_feature = both_feature
        self.info = {}
        for line in open(info, 'r', encoding='utf-8'):
            line = line.strip().split('@')
            name = line[0]
            A = line[1].split('#')
            B = [int(x) for x in A]
            self.info[name] = B

        self.distribution = [int(line.strip()) for line in open(length_distribution, 'r', encoding='utf-8')]

        self.max_len = max_len
        self.audio_feature_len = 512
        self.video_feature_len = 1280
        if self.both_feature:
            self.feature_len = self.audio_feature_len + self.video_feature_len*2
        else:
            self.feature_len = self.audio_feature_len + self.video_feature_len

        self.device = device
        self.split = split
        self.transformation = transformation

        self.scenes = [] #list of (l, r, target)
        for line in tqdm.tqdm(open(self.filename, 'r', encoding='utf-8'), desc='Preparing segmentation {} dataset'.format(split)):
            line = line.strip().split('@')
            name = line[0]
            A = line[1].split('#')
            total_f = self.total_frame_num(name)
            for x in A:
                x = x.split('-')
                l, r = int(x[0]), int(x[1])
                l, r = l//5*5, r//5*5
                self.scenes.append((name, l, r, total_f))
    
    def total_frame_num(self, name):
        sum = 0
        for x in self.info[name]:
            sum += x
        return sum

    def __len__(self):
        return len(self.scenes)

    def get_sample_len(self, len, label):
        if label == 0:
            L = random.sample(self.distribution, 1)[0]
            len = min(len, L)
        len = len * 4 // 5
        len = min(len, self.max_len)
        return len

    def audio_feature(self, name, l, r):
        f_n = self.info[name]
        path = '/mnt/sda1/tangzitian/FootballHighlights/games/'+name+'/'+name
        x = 0
        A = []
        for i in range(len(f_n)):
            if x >= r:
                break
            y = x + f_n[i]
            if y > l:
                if i<10:
                    filename = path+'[0%d].npy'%i
                else:
                    filename = path+'[%d].npy'%i
                a = np.load(filename)
                left = (max(l, x) - x) * 1024
                right = (min(r, y) - x) * 1024
                A.append(a[left: right])
            x = y
        A = np.concatenate(A)
        feature = librosa.feature.melspectrogram(A, 25600, n_fft=4096, hop_length=5120, n_mels=self.audio_feature_len).transpose()[:(r-l)//5, :]
        return torch.from_numpy(feature)
    
    
    def video_feature(self, name, l, r, own_feature=False):
        f_n = self.info[name]
        if own_feature:
            path = '/mnt/sda1/songzimeng/GameHighlight_feature/'+name+'/'+name
        else:
            path = '/mnt/sda1/songzimeng/Classifier_feature/'+name+'/'+name
        L = (r - l) // 5
        l = (l + 4) // 5
        r = l + L
        x = 0
        sum = 0
        A = []
        for i in range(len(f_n)):
            if x >= r:
                break
            sum += f_n[i]
            y = (sum + 4) // 5
            if y > l:
                if i<10:
                    filename = path+'[0%d].pt'%i
                else:
                    filename = path+'[%d].pt'%i
                a = torch.load(filename)
                left = max(l, x) - x
                right = min(r, y) - x
                A.append(a[left: right, :])
            x = y
        feature = torch.cat(A, dim = 0)
        return feature

    # @profile
    def __getitem__(self, index):
        # TODO
        name, l, r, total_f = self.scenes[index]
        left_shift_min = max(0,300-((total_f//5)-r//5))
        left_shift_max = min(300,l//5)
        if left_shift_min < left_shift_max:
            left_shift = np.random.randint(low = left_shift_min, high = left_shift_max)
        else:
            left_shift = left_shift_min
        
        target = torch.zeros((r//5-l//5+300)).long()
        target[left_shift:left_shift-l//5+r//5].fill_(1)

        
        audio_feature = self.audio_feature(name, l-left_shift*5, r+(300-left_shift)*5)
        
        if self.both_feature:
            video_feature = torch.cat(
                (
                    self.video_feature(name, l-left_shift*5, r+(300-left_shift)*5, True),
                    self.video_feature(name, l-left_shift*5, r+(300-left_shift)*5)
                ),
                dim=-1
            )
        elif self.own_feature:
            video_feature = self.video_feature(name, l-left_shift*5, r+(300-left_shift)*5, True)
        else:
            video_feature = self.video_feature(name, l-left_shift*5, r+(300-left_shift)*5)
        feature = torch.cat((audio_feature, video_feature), dim=-1)
        feature, target = self.transformation((feature, target))

        if feature.size(0) != target.size(0):
            print(index)
            print(feature.size(0), target.size(0))
            print(l, r)
            print(total_f)
            print(left_shift_min, left_shift_max)
            
            assert False

        return {
            "length": feature.size(0),
            "x": feature,
            "target": target
        }


    # @profile
    def collate_fn(self, samples):
        # TODO
        lens = [sample["length"] for sample in samples]
        max_len = max(lens)
        bsz = len(lens)

        x = torch.FloatTensor(bsz, max_len, self.feature_len).fill_(0.0)
        target = torch.LongTensor(bsz, max_len).fill_(0)

        for idx, sample in enumerate(samples):
            x[idx, 0: sample["length"], :] = sample["x"]
            target[idx, 0:sample["length"]] = sample["target"]
        
        return {
            "length": torch.LongTensor(lens).to(self.device),
            "x": x.to(self.device),
            "label": target.to(self.device)
        }


if __name__ == "__main__":
    train_set = Dataset()
    print(len(train_set))
    print(train_set[0])
