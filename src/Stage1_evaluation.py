import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from dataset import Dataset
from tqdm import tqdm
import argparse

@torch.no_grad()
def evaluate(model, dataset, batch_size = 64):
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

    losses = []
    hit = 0
    TP, FP, TN, FN = 0, 0, 0, 0
    for samples in tqdm(dataloader, desc="validation"):
        label = samples.pop("label")
        
        logits = model.logits(**samples)
        predict_label = logits.argmax(dim=-1)
        hit += (predict_label == label).sum()

        TP += ((predict_label==1)*(label==1)).sum()
        FP += ((predict_label==1)*(label==0)).sum()
        TN += ((predict_label==0)*(label==0)).sum()
        FN += ((predict_label==0)*(label==1)).sum()

        losses.append(F.cross_entropy(logits, label).item())
    print("%s: loss: %.3f, acc: %.4f" % (dataset.split, np.mean(losses), hit / len(dataset)))
    print("%s: Acc: %.4f, Percision: %.4f, Recall: %.4f, Specificity: %.4f" % (dataset.split, (TP+TN)/(TP+TN+FP+FN), TP/(TP+FP), TP/(TP+FN), TN/(FP+TN)))
    return np.mean(losses), hit / len(dataset)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--video-feature", default=1280, type=int)
    parser.add_argument("--audio-feature", default=512, type=int)
    parser.add_argument("--own-feature", action="store_true", default=False)
    parser.add_argument("--both-feature", action="store_true", default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    dataset = Dataset(split="val", device="cuda", audio_feature_len = args.audio_feature, video_feature_len = args.video_feature, own_feature = args.own_feature, both_feature=args.both_feature)
    try:
        model = torch.load("models/"+args.model_name)
    except FileNotFoundError as e:
        print(e)
        exit()
    evaluate(model, dataset, args.batch_size)
