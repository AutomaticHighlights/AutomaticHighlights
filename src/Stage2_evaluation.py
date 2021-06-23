import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from dataset import DatasetSegmentation as Dataset
from tqdm import tqdm
import argparse

@torch.no_grad()
def evaluate(model, dataset, batch_size = 1):
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

    losses = []
    iou_list = []
    TP, FP, TN, FN = 0, 0, 0, 0
    hit = 0 
    pbar = tqdm(dataloader)
    for samples in pbar:
        label = samples.pop("label")

        logits = model.logits(**samples)
        predict_label = logits.argmax(dim=1)

        hit += (predict_label == label).sum()

        TP += ((predict_label==1)*(label==1)).sum()
        FP += ((predict_label==1)*(label==0)).sum()
        TN += ((predict_label==0)*(label==0)).sum()
        FN += ((predict_label==0)*(label==1)).sum()

        pbar.set_description("Validation, Acc: {:.4f}".format((TP+TN)/(TP+TN+FP+FN)))

    print("%s: acc: %.4f" % (dataset.split, (TP+TN)/(TP+TN+FP+FN)))
    print("%s: Acc: %.4f, Percision: %.4f, Recall: %.4f, Specificity: %.4f" % (dataset.split, (TP+TN)/(TP+TN+FP+FN), TP/(TP+FP), TP/(TP+FN), TN/(FP+TN)))
    return 0.0, (TP+TN)/(TP+TN+FP+FN)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--batch-size", default=1, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    dataset = Dataset(split="train", device="cuda")
    try:
        model = torch.load("models/"+args.model_name)
    except FileNotFoundError as e:
        print(e)
        exit()
    evaluate(model, dataset, args.batch_size)
