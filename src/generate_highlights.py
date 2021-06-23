import librosa
from scipy.ndimage.measurements import label
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from moviepy.editor import *
import argparse

def get_file_num(path):
    path = path.replace('[', '\[')
    path = path.replace(']', '\]')
    b = os.popen('ls -lA '+path+'/*.mp4 |grep "^-"|wc -l')
    for line in b:
        a = line
    return int(a.strip())

def cut(path, name, frame_num, l ,r):
    clips = []
    x = 0
    for i in range(len(frame_num)):
        if x >= r:
            break
        y = x + frame_num[i]
        if y > l:
            if i<10:
                filename = path+'/'+name+'[0%d].mp4'%i
            else:
                filename = path+'/'+name+'[%d].mp4'%i

            left = (max(l, x) - x) * 0.04
            right = (min(r, y) - x) * 0.04
            clip = VideoFileClip(filename).subclip(left, right)
            clips.append(clip)
        x = y
    return clips

def length(x):
    l,r = x
    return r-l

def get_two(p):
    l = 0
    L = p.shape[0]
    A = []
    while l<L:
        r = l
        while r<L and p[r]>=0.5:
            r += 1
        if r>l:
            A.append((l,r))
            if len(A)>2:
                A.sort(key=length, reverse=True)
                A = A[:2]
        l = r + 1
    return A

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="bi_best_val.pt", type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = get_args()
    
    audio_feature_len = 512
    video_feature_len = 1280

    window = 30 #second
    stride = 1 #second
    minu = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"


    name = '[WCQ2022]China_vs_Maldives_2021-06-12'
    
    model_name = args.model_name

    model = torch.load("models/"+model_name)

    model2 = torch.load("models/bilstm_va_seg_best_val.pt")

    path = '/mnt/sda1/tangzitian/FootballHighlights/games/' + name
    f_n = get_file_num(path)

    A = []
    B = []
    C = []
    frame_num = []
    sum = 0
    x = 0
    for i in range(f_n):
        if i<10:
            filename = path+'/'+name+'[0%d].npy'%i
        else:
            filename = path+'/'+name+'[%d].npy'%i
        A.append(np.load(filename))
        f = torch.load(filename.replace('.npy', '.pt').replace('tangzitian/FootballHighlights/games', 'songzimeng/Classifier_feature'))
        cap = cv2.VideoCapture(filename.replace('.npy', '.mp4'))
        f_n = int(cap.get(7))
        frame_num.append(f_n)
        sum += f_n
        y = (sum + 4) // 5
        B.append(f[: y-x, :])
        x = y

    A = np.concatenate(A, axis=0)
    audio_feature = librosa.feature.melspectrogram(A, 25600, n_fft=4096, hop_length=5120, n_mels=audio_feature_len).transpose()
    audio_feature = torch.from_numpy(audio_feature).float()
    video_feature = torch.cat(B, dim=0)
    audio_feature = audio_feature[: video_feature.size(0), :] / 10
    a_f = audio_feature * 10

    feature = torch.cat((audio_feature, video_feature), dim=-1)
    feature2 = torch.cat((a_f, video_feature), dim=-1)

    feature_len = feature.size(-1)
    blank = torch.zeros((window - stride) * 5, feature_len)
    feature = torch.cat((blank, feature), dim=0).unsqueeze(0).to(device)
    feature2 = feature2.unsqueeze(0).to(device)

    L = feature.size(1)
    i = 0


    prob = np.zeros((L))

    while i + window * 5 <= L:
        x = feature[:, i: i + window * 5, :]
        p = model.logits(x=x).squeeze()
        p = F.softmax(p)
        print(p)
        prob[i: i + window * 5] += p[1].item()

        i += stride * 5

    prob = prob / (window // stride)

    prob = prob[(window - stride) * 5: ]
    print(prob.shape)
    L -= (window - stride) * 5
    minute = [i/300 for i in range(L)]
    
    plt.figure(figsize=(30,10))
    plt.plot(minute, prob, label='prob')
    plt.xlabel('Minute')
    plt.ylabel('Prob')

    plt.legend()
    
    plt.savefig('prob.png')
    plt.close()

    b = prob.tolist()
    print(len(b))
    b.sort(reverse=True)
    threshold = b[minu * 300]
    print('thres,', threshold)

    
    cnt = 0
    l = 0
    final_clip = []
    final_p = np.zeros_like(prob)
    while l<L:
        r = l
        while r<L and prob[r]>=threshold:
            r += 1
        if r>l:
            cnt += 1
            feat = feature2[:, max(0, l-150): min(L, r+150), :]
            p = model2.logits(x=feat).transpose(1,2).squeeze()
            p = F.softmax(p)
            p = p[:, 1].detach().cpu().numpy()
            li = get_two(p)
            for x,y in li:
                x += l - 150
                y += l - 150
                final_p[x:y] = 1
        l = r + 1
    
    l = 0
    final_clip = []
    while l<L:
        r = l
        while r<L and final_p[r]==1:
            r += 1
        if r>l:
            final_clip.extend(cut(path, name, frame_num, l*5 ,r*5))
        l = r + 1
    final_clip = concatenate_videoclips(final_clip)
    final_clip.write_videofile('highlights.mp4')
