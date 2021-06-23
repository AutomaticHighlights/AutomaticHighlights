import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from dataset import Dataset
from Stage1_evaluation import evaluate
from model import Net
import argparse
import numpy as np
import os
from tqdm import tqdm

curdir = os.path.dirname(__file__)

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--feature-dim", default=1792, type=int)
    parser.add_argument("--video-feature", default=1280, type=int)
    parser.add_argument("--audio-feature", default=512, type=int)
    parser.add_argument("--own-feature", action="store_true", default=False)
    parser.add_argument("--both-feature", action="store_true", default=False)
    parser.add_argument("--hidden-size", default=512, type=int)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--num-epoch", default=100, type=int)
    parser.add_argument("--save-interval", default=1, type=int)
    parser.add_argument("--save-dir", default=os.path.join(curdir, "models"))
    parser.add_argument('--gradient-accumulation-steps',type=int,default=1)
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    return args


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    print(args)
    
    train_set = Dataset(device=device, audio_feature_len=args.audio_feature, video_feature_len=args.video_feature, own_feature=args.own_feature, both_feature=args.both_feature)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn, shuffle=True)

    valid_set = Dataset(split="dev", device=device, audio_feature_len=args.audio_feature, video_feature_len=args.video_feature, own_feature=args.own_feature, both_feature=args.both_feature)
    model = Net(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    evaluate(model, valid_set)
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_loss = 1e10
    for epoch in range(args.num_epoch):
        model.train()
        with tqdm(train_loader, desc="training") as pbar:
            losses = []
            hit_sum = 0
            n = 0
            for step, samples in enumerate(pbar):
                # optimizer.zero_grad()
                loss, hit = model.get_loss(**samples)
                hit_sum += hit
                n += samples['label'].size(0)
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()    # We have accumulated enought gradients
                    model.zero_grad()
                    global_step += 1
                losses.append(loss.item())
                pbar.set_description("Epoch: %d, Loss: %0.8f, Train acc: %.4lf, lr: %0.9f, step: %d" % (epoch + 1, np.mean(losses), hit_sum/n, optimizer.param_groups[0]['lr'], global_step))
                if optimizer.param_groups[0]['lr'] == 0:
                    break
        if optimizer.param_groups[0]['lr'] == 0:
            break
        loss, acc = evaluate(model, valid_set)
        if acc>best_val_acc:
            best_val_acc = acc
            torch.save(model, args.save_dir + "/" + args.name + "_best_val.pt")
        if loss<best_loss:
            best_loss = loss
            torch.save(model, args.save_dir + "/" + args.name + "_best_loss.pt")
        if hit_sum/n>best_train_acc:
            best_train_acc = hit_sum/n
            torch.save(model, args.save_dir + "/" + args.name + "_best_train.pt")

if __name__ == "__main__":
    args = get_args()
    train(args)
