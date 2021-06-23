import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from dataset import DatasetSegmentation as Dataset
from Stage2_evaluation import evaluate
from model import NetU_Segmentation, LSTMSegmentation, LSTMBidirectionalSegmentation
import argparse
import numpy as np
import os
from tqdm import tqdm
import numpy as np

curdir = os.path.dirname(__file__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dim", default=3072, type=int)
    parser.add_argument("--hidden-size", default=512, type=int)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--num-epoch", default=100, type=int)
    parser.add_argument("--save-interval", default=1, type=int)
    parser.add_argument("--save-dir", default=os.path.join(curdir, "models"))
    parser.add_argument('--gradient-accumulation-steps',
        type=int,
        default=2,
        help="Number of updates steps to accumualte before performing a backward/update pass.")
    args = parser.parse_args()
    return args



def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    print(args)
    
    train_set = Dataset(device=device, both_feature=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn, shuffle=True)

    valid_set = Dataset(split="dev", device=device, both_feature=True)
    model = LSTMBidirectionalSegmentation(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr/args.gradient_accumulation_steps, weight_decay=args.weight_decay)

    global_step = 0
    evaluate(model, valid_set)
    best_val_acc = 0.0
    best_train_iou = 0.0
    for epoch in range(args.num_epoch):
        model.train()
        with tqdm(train_loader, desc="training") as pbar:
            losses = []
            iou_list = []
            n = 0
            for step, samples in enumerate(pbar):
                # optimizer.zero_grad()
                loss, iou = model.get_loss(**samples)
                iou_list.append(iou)
                n += samples['label'].size(0)
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()    # We have accumulated enought gradients
                    model.zero_grad()
                    global_step += 1
                losses.append(loss.item())
                pbar.set_description("Epoch: %d, Loss: %0.8f, Train IoU: %.4lf, lr: %0.9f, step: %d" % (epoch + 1, np.mean(losses), np.mean(iou_list), optimizer.param_groups[0]['lr'], global_step))
                if optimizer.param_groups[0]['lr'] == 0:
                    break
        if optimizer.param_groups[0]['lr'] == 0:
            break
        loss, acc = evaluate(model, valid_set)
        if acc>best_val_acc:
            best_val_acc = acc
            torch.save(model, args.save_dir + "/bilstm_va_both_seg_best_val.pt")
        if np.mean(iou_list)>best_train_iou:
            best_train_iou = np.mean(iou_list)
            torch.save(model, args.save_dir + "/bilstm_va_both_seg_best_train.pt")

if __name__ == "__main__":
    args = get_args()
    train(args)
