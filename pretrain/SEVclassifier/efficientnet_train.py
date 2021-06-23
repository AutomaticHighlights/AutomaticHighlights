import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from SEVdataset import SEVDataset
import argparse
import numpy as np
import os
from tqdm import tqdm
import time
import IPython
from torchvision import datasets, transforms
import warnings
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.dataloader import default_collate
from efficientnet_pytorch import EfficientNet  # EfficientNet的使用需要倒入的库
from label_smooth import LabelSmoothSoftmaxCE
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

curdir = os.path.dirname(__file__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--num-epoch", default=10, type=int)
    parser.add_argument("--save-interval", default=1, type=int)
    parser.add_argument("--step-interval", default=10, type=int)
    parser.add_argument("--step-save", default=1000, type=int)
    parser.add_argument("--evaluate-step", default=100, type=int)
    parser.add_argument("--save-dir", default=os.path.join(curdir, "SEVmodels/"))
    parser.add_argument("--total-updates", default=50000, type=int)
    parser.add_argument('--gradient-accumulation-steps',
                        type=int,
                        default=10,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--model-type", default='efficientnet-b0')
    parser.add_argument("--class_num", default=10, type=int)
    parser.add_argument("--feature_extract", default=True, type=bool)
    parser.add_argument("--cuda_num", default=3, type=int)
    args = parser.parse_args()
    return args

def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        wp = int((1280 - w) / 2)
        hp = int((720 - h) / 2)
        padding = (wp, hp, wp, hp)

        try:
            image = F.pad(image, padding, 0, 'constant')
        except Exception as e:
            pass

        return image


def train(args):
    print(args)
    args.save_dir = args.save_dir + args.model_type
    args.save_dir += "_SEV_"
    args.save_dir = args.save_dir + time.strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs(args.save_dir, exist_ok=True)
    print(args.save_dir, 'make!')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.cuda_num)
    print(device)
    if torch.cuda.is_available():
        print('device: ', torch.cuda.current_device())

    data_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize([224, 398]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_set = SEVDataset('/mnt/sda1/songzimeng/officialSEV/train/', transform=data_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    valid_set = SEVDataset('/mnt/sda1/songzimeng/officialSEV/test/', transform=data_transform)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)

    model = EfficientNet.from_pretrained(args.model_type, num_classes=args.class_num).to(device)

    params_to_update = model.parameters()
    print("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    criterion = LabelSmoothSoftmaxCE()
    optimizer = optim.Adam(params_to_update, lr=args.lr, betas=(0.9, 0.999), eps=1e-9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)


    global_step = 0
    # evaluate(model, valid_set)


    best_acc = 0

    for epoch in range(args.num_epoch):
        print('epoch: ', epoch+1)

        losses = []
        total = 0
        correct = 0
        for step, samples in enumerate(train_loader, 0):
            model.train()
            imgs, labels = samples['image'].to(device).float(), samples['label'].to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).cpu().sum()

            if (step + 1) % args.step_interval == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, step + 1, np.mean(losses), 100. * float(correct) / float(total)))


            if (step + 1) % args.step_save == 0:
                torch.save(model, args.save_dir + "/SEV_step_save.pt")
                losses = []
                total = 0
                correct = 0

                with torch.no_grad():
                    print('Evaluate')
                    eval_correct = 0
                    eval_total = 0
                    evaluate_step = 0
                    for samples in valid_loader:
                        model.eval()
                        imgs, labels = samples['image'].to(device).float(), samples['label'].to(device)
                        outputs = model(imgs)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        eval_total += labels.size(0)
                        eval_correct += (predicted == labels).cpu().sum()

                        evaluate_step += 1
                        if evaluate_step >= args.evaluate_step :
                            break

                    print('Evaluated acc：%.3f%%' % (100. * float(eval_correct) / float(eval_total)))
                    acc = 100. * float(eval_correct) / float(eval_total)
                    #scheduler.step(acc)

                    if acc > best_acc:
                        torch.save(model, args.save_dir + "/SEV_best_save.pt")
                        best_acc = acc


        if (epoch + 1) % args.save_interval == 0 or epoch == 0:
            torch.save(model, args.save_dir + "/SEV_{}.pt".format(epoch + 1))

        if optimizer.param_groups[0]['lr'] == 0:
            break

        with torch.no_grad():
            print('Evaluate')
            eval_correct = 0
            eval_total = 0
            evaluate_step = 0
            for samples in valid_loader:
                model.eval()
                imgs, labels = samples['image'].to(device).float(), samples['label'].to(device)
                outputs = model(imgs)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                eval_total += labels.size(0)
                eval_correct += (predicted == labels).cpu().sum()

                evaluate_step += 1
                if evaluate_step >= (args.evaluate_step * 10):
                    break

            print('Evaluated acc：%.3f%%' % (100. * float(eval_correct) / float(eval_total)))
            eval_acc = 100. * float(eval_correct) / float(eval_total)
            scheduler.step(eval_acc)



if __name__ == "__main__":
    args = get_args()
    train(args)
