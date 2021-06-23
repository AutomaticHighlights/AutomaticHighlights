import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.device = args.device
        self.hidden_size = args.hidden_size
        v_f = args.video_feature * 2 if args.both_feature else args.video_feature
        self.lstm = nn.LSTM(args.audio_feature + v_f, args.hidden_size, num_layers=args.num_layers, batch_first=True, bidirectional=True)

        self.linear = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size, 2)

    def logits(self, x, **unused):

        output, hidden = self.lstm(x)
        x, _ = output.max(1)
        x = F.relu(self.linear(x))
        x = self.linear2(x)

        return x
    
    def get_loss(self, x, label, **unused):
        logits = self.logits(x)
        predict_label = logits.argmax(dim=-1)
        hit = (predict_label == label).sum()
        loss = F.cross_entropy(logits, label)
        return loss, hit

class LSTMBidirectionalSegmentation(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.device = args.device
        self.hidden_size = args.hidden_size
        self.lstm = nn.LSTM(args.feature_dim, args.hidden_size, num_layers=args.num_layers, batch_first=True, bidirectional=True)

        self.linear = nn.Linear(args.hidden_size*2, args.hidden_size*2)
        self.linear2 = nn.Linear(args.hidden_size*2, 2)

    def logits(self, x, **unused):

        x, hidden = self.lstm(x)
        x = F.relu(self.linear(x))
        x = self.linear2(x)

        return x.transpose(1,2)
    
    def get_loss(self, x, label, **unused):
        logits = self.logits(x) #(b,2,t)
        weight = torch.Tensor([2/5,8/5]).to(logits.device)

        loss = F.cross_entropy(logits, label, weight=weight)
        predict_label = logits.argmax(dim=1)
        target_positive = (label == 1)
        pred_positive = (predict_label == 1)

        union, intersection = torch.logical_or(target_positive, pred_positive), torch.logical_and(target_positive, pred_positive)
        iou = (torch.sum(intersection*1)/torch.sum(union*1)).item()

        return loss, iou








# Belows are some models we tried in our experiments

class NetU(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.device = args.device
        self.hidden_size = args.hidden_size
        self.in_proj = nn.Linear(args.feature_dim, args.hidden_size)
        self.linear = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size, 2)

        self.conv_list = nn.ModuleList([nn.Conv1d(in_channels=args.hidden_size*(2**i), out_channels=args.hidden_size*(2**(i+1)), kernel_size=3, stride=2, padding=1) for i in range(3)])
        self.conv1_list_down = nn.ModuleList([nn.Conv1d(in_channels=args.hidden_size*(2**i), out_channels=args.hidden_size*(2**(i)), kernel_size=1, stride=1) for i in range(4)])
        self.trans_conv_list = nn.ModuleList([nn.ConvTranspose1d(in_channels=args.hidden_size*(2**(i+1)), out_channels=args.hidden_size*(2**i), kernel_size=3, stride=2) for i in range(3)])
        self.conv1_list_up = nn.ModuleList([nn.Conv1d(in_channels=args.hidden_size*(2**(i+1)), out_channels=args.hidden_size*(2**i), kernel_size=1) for i in range(3)])
        self.relu = nn.ReLU()
        

    def logits(self, x, **unused):
        # (B, T, n_h)
        length_list = []
        conv_1_list = []
        x=torch.transpose(self.relu(self.in_proj(x)),1,2)
        length_list.append(x.size(2))
        for i in range(len(self.conv_list)):
            down_sample=self.relu(self.conv_list[i](x))
            conv_1_list.append(self.conv1_list_down[i](x))
            x=down_sample
            length_list.append(x.size(2))
        x=self.conv1_list_down[-1](x)
        for i in range(len(self.conv_list)-1, -1, -1):
            x=self.relu(self.trans_conv_list[i](x))
            x=x[:,:,1:length_list[i]+1]
            x=torch.cat([x, conv_1_list[i]], dim=1)
            x=self.conv1_list_up[i](x)
        x=torch.transpose(x, 1, 2)
        #(B, t, n_h)
        x = x.mean(1)
        x = F.relu(self.linear(x))
        x = self.linear2(x)

        return x
        #(B,t,2)
    
    def get_loss(self, x, label, **unused):
        logits = self.logits(x)
        predict_label = logits.argmax(dim=-1)
        hit = (predict_label == label).sum()
        loss = F.cross_entropy(logits, label)
        return loss, hit


class NetU_Segmentation(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.device = args.device
        self.hidden_size = args.hidden_size
        self.in_proj = nn.Linear(args.feature_dim, args.hidden_size)
        self.linear = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size, 2)

        self.conv_list = nn.ModuleList([nn.Conv1d(in_channels=args.hidden_size*(2**i), out_channels=args.hidden_size*(2**(i+1)), kernel_size=3, stride=2, padding=1) for i in range(3)])
        self.conv1_list_down = nn.ModuleList([nn.Conv1d(in_channels=args.hidden_size*(2**i), out_channels=args.hidden_size*(2**(i)), kernel_size=1, stride=1) for i in range(4)])
        self.trans_conv_list = nn.ModuleList([nn.ConvTranspose1d(in_channels=args.hidden_size*(2**(i+1)), out_channels=args.hidden_size*(2**i), kernel_size=3, stride=2) for i in range(3)])
        self.conv1_list_up = nn.ModuleList([nn.Conv1d(in_channels=args.hidden_size*(2**(i+1)), out_channels=args.hidden_size*(2**i), kernel_size=1) for i in range(3)])
        self.relu = nn.ReLU()
        

    def logits(self, x, **unused):
        # (B, T, n_h)
        length_list = []
        conv_1_list = []
        x=torch.transpose(self.relu(self.in_proj(x)),1,2)
        length_list.append(x.size(2))
        for i in range(len(self.conv_list)):
            down_sample=self.relu(self.conv_list[i](x))
            conv_1_list.append(self.conv1_list_down[i](x))
            x=down_sample
            length_list.append(x.size(2))
        x=self.conv1_list_down[-1](x)
        for i in range(len(self.conv_list)-1, -1, -1):
            x=self.relu(self.trans_conv_list[i](x))
            x=x[:,:,1:length_list[i]+1]
            x=torch.cat([x, conv_1_list[i]], dim=1)
            x=self.conv1_list_up[i](x)
        x=torch.transpose(x, 1, 2)
        #(B, t, n_h)
        x = F.relu(self.linear(x))
        x = self.linear2(x)

        return x.transpose(1,2)
        #(B,2,t)
    
    def get_loss(self, x, label, **unused):
        logits = self.logits(x) #(b,2,t)
        weight = torch.Tensor([2/5,8/5]).to(logits.device)

        loss = F.cross_entropy(logits, label, weight=weight)
        predict_label = logits.argmax(dim=1)
        target_positive = (label == 1)
        pred_positive = (predict_label == 1)

        union, intersection = torch.logical_or(target_positive, pred_positive), torch.logical_and(target_positive, pred_positive)
        iou = (torch.sum(intersection*1)/torch.sum(union*1)).item()

        return loss, iou

class LSTMSegmentation(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.device = args.device
        self.hidden_size = args.hidden_size
        self.lstm = nn.LSTM(args.feature_dim, args.hidden_size, num_layers=args.num_layers, batch_first=True, bidirectional=False)

        self.linear = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size, 2)

    def logits(self, x, **unused):

        x, hidden = self.lstm(x)
        x = F.relu(self.linear(x))
        x = self.linear2(x)

        return x.transpose(1,2)
    
    def get_loss(self, x, label, **unused):
        logits = self.logits(x) #(b,2,t)
        loss = F.cross_entropy(logits, label)
        predict_label = logits.argmax(dim=1)
        
        target_positive = (label == 1)
        pred_positive = (predict_label == 1)

        union, intersection = torch.logical_or(target_positive, pred_positive), torch.logical_and(target_positive, pred_positive)
        iou = (torch.sum(intersection*1)/torch.sum(union*1)).item()

        return loss, iou