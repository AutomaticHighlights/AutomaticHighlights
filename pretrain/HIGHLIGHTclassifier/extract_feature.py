import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from HIGHLIGHTdataset import SEVDataset
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
import cv2
from PIL import Image

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
    parser.add_argument("--evaluate-step", default=10, type=int)
    parser.add_argument("--save-dir", default=os.path.join(curdir, "highlightmodels/"))
    parser.add_argument("--load-path", default=os.path.join(curdir, "load_model/highlight_efficientnet.pt"))
    parser.add_argument("--total-updates", default=50000, type=int)
    parser.add_argument('--gradient-accumulation-steps',
                        type=int,
                        default=10,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--model-type", default='efficientnet-b0')
    parser.add_argument("--class_num", default=2, type=int)
    parser.add_argument("--feature_extract", default=True, type=bool)
    parser.add_argument("--cuda_num", default=1, type=int)
    parser.add_argument("--extract_interval", default=5, type=int)
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

def evaluate(args, model, device):

    data_transform = transforms.Compose([
        transforms.Resize(efficientnet_params(args.model_type)[2]),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    valid_set = SEVDataset('/mnt/sda1/songzimeng/NewSEVdata/valid/', transform=data_transform)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)

    model = torch.load(args.load_path).to(device)

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
            if evaluate_step >= args.evaluate_step:
                break

        print('Evaluated acc:', (100. * float(eval_correct) / float(eval_total)))

def extract_prob(args, model, videoPath, device):

    data_transform = transforms.Compose([
        transforms.Resize(efficientnet_params(args.model_type)[2]),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    timeF = args.extract_interval

    prob = []

    cam = cv2.VideoCapture(videoPath)
    # reading from frame
    ret, frame = cam.read()  # ret为布尔值 frame保存着视频中的每一帧图像 是个三维矩阵

    frameNum = 0  # 浏览帧数
    imageNum = 0  # 保存图片数量

    while ret:  # 如果视频仍然存在，继续创建图像
        if frameNum % timeF == 0:
            # 呈现输出图片的数量
            imageNum += 1
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = data_transform(image).to(device).unsqueeze(0)

            output = model(image)
            logsoftmax = torch.nn.LogSoftmax(dim=-1)
            output_softmax = logsoftmax(output)

            prob.append(output_softmax[0][1].cpu())

        frameNum += 1

        ret, frame = cam.read()

    # 一旦完成释放所有的空间和窗口
    cam.release()
    cv2.destroyAllWindows()

    return np.array(prob)

def extract_features(args, model, videoPath, device):

    data_transform = transforms.Compose([
        transforms.Resize(efficientnet_params(args.model_type)[2]),
        # transforms.Resize([efficientnet_params(args.model_type)[2], efficientnet_params(args.model_type)[2]]),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    timeF = args.extract_interval

    features = []

    cam = cv2.VideoCapture(videoPath)
    # reading from frame
    ret, frame = cam.read()  # ret为布尔值 frame保存着视频中的每一帧图像 是个三维矩阵

    frameNum = 0  # 浏览帧数  idx actually
    imageNum = 0  # 保存图片数量

    while ret:  # 如果视频仍然存在，继续创建图像

        if frameNum % timeF == 0:
            # 呈现输出图片的数量
            imageNum += 1
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = data_transform(image).to(device).unsqueeze(0)

            output_feature = model.extract_features(image)
            output_feature = model._avg_pooling(output_feature).squeeze().cpu()

            features.append(output_feature.numpy().tolist())

        frameNum += 1

        ret, frame = cam.read()

    # 一旦完成释放所有的空间和窗口
    cam.release()
    cv2.destroyAllWindows()


    return torch.Tensor(features)


if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.cuda_num)
    print(device)
    if torch.cuda.is_available():
        print('device: ', torch.cuda.current_device())

    model = torch.load(args.load_path).to(device)

    #evaluate(args, model, device)


    def test():
        # video_path = '/mnt/sda1/tangzitian/FootballHighlights/games/[UCL20-21]ManchesterCity_vs_Chelsea_2021_05_30/[UCL20-21]ManchesterCity_vs_Chelsea_2021_05_30[00].mp4'
        model.eval()
        matchPath = '/mnt/sda1/tangzitian/FootballHighlights/games/[UCL20-21]ManchesterCity_vs_Chelsea_2021_05_30/'

        total_prob = np.array([])
        for video in os.listdir(matchPath):

            videoPath = matchPath + video
            if videoPath[-4:] != '.mp4':
                continue

            print(len(total_prob))

            with torch.no_grad():
                prob = extract_prob(args, model, videoPath, device)
                # feature = extract_features(args, model, video_path, device)

            total_prob = np.concatenate((total_prob, prob), axis=0)

        np.save('gh_ucl.npy', total_prob)

        IPython.embed()
        os._exit(0)


    #test()

    train_list_file = 'train_list.txt'
    valid_list_file = 'validation_list.txt'
    special_list_file = 'special_list.txt'

    train_list = []
    valid_list = []
    special_list = []

    train_file = open(train_list_file)
    for line in train_file:
        match_name = ''
        for s in line:
            if s != '@':
                match_name = match_name + s
            else:
                train_list.append(match_name)
                break
    train_file.close()

    valid_file = open(valid_list_file)
    for line in valid_file:
        match_name = ''
        for s in line:
            if s != '@':
                match_name = match_name + s
            else:
                valid_list.append(match_name)
                break
    valid_file.close()

    special_file = open(special_list_file)
    for line in special_file:
        match_name = ''
        for s in line:
            if s != '@':
                match_name = match_name + s
            else:
                special_list.append(match_name)
                break
    special_file.close()

    model.eval()
    with torch.no_grad():

        dataPath = '/mnt/sda1/tangzitian/FootballHighlights/'
        gamePath = dataPath + 'games/'

        total_savePath = '/mnt/sda1/songzimeng/GameHighlight_feature/'

        matchNum = 0
        for match in os.listdir(gamePath):
            matchNum += 1
            print(matchNum)

            # if not ((match in train_list) or (match in valid_list)):
            if not ((match in special_list)):  # select which file to be used
                print(match, "not used!")
                continue
            else:
                print(match, "used!")

            # IPython.embed()
            # os._exit(0)

            matchPath = gamePath + match + '/'
            match_savePath = total_savePath + match + '/'
            if not os.path.isdir(matchPath):
                continue
            if not os.path.isdir(match_savePath):
                os.mkdir(match_savePath)

            videoNum = 0
            for video in os.listdir(matchPath):

                videoPath = matchPath + video
                if videoPath[-4:] != '.mp4':
                    continue


                if os.path.exists(match_savePath + video[:-4] + '.pt'):
                    print(videoPath + ' : ' + str(videoNum) + ', Exist ' + video)
                    continue

                try:
                    feature = extract_features(args, model, videoPath, device)
                    torch.save(feature, match_savePath + video[:-4] + '.pt')
                    videoNum += 1
                    print(videoPath + ' : ' + str(videoNum) + ', Finish ' + video)
                except Exception as e:
                    videoNum += 1
                    print(videoPath + ' : ' + str(videoNum) + ', Wrong ' + video)




