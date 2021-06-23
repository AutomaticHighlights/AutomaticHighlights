# https://www.pythonf.cn/read/110040

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import IPython



def get_images_and_labels(dir_path):
    '''
    从图像数据集的根目录dir_path下获取所有类别的图像名列表和对应的标签名列表
    :param dir_path: 图像数据集的根目录
    :return: images_list, labels_list
    '''
    dir_path = Path(dir_path)
    classes = []  # 类别名列表

    #for category in dir_path.iterdir():
    #    if category.is_dir():
    #        classes.append(category.name)
    classes = ['Penalty', 'Free-Kick', 'Yellow-Cards', 'Corner', 'Left', 'To-Subtitue', 'Red-Cards', 'Center', 'Tackle', 'Right']
    print(classes)
    images_list = []  # 文件名列表
    labels_list = []  # 标签列表

    for index, name in enumerate(classes):
        class_path = dir_path / name
        if not class_path.is_dir():
            continue
        for img_path in class_path.glob('*.jpg'):
            images_list.append(str(img_path))
            labels_list.append(int(index))
    print(min(labels_list), max(labels_list))
    return images_list, labels_list


class SEVDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path    # 数据集根目录
        self.transform = transform
        self.images, self.labels = get_images_and_labels(self.dir_path)

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        #img = cv2.imread(img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(img_path).convert('RGB')
        #img = torch.tensor(img)
        #label = torch.tensor(label)
        sample = {'image': img, 'label': label}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample


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

if __name__ == '__main__':
    data_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize([224, 398]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dataset = SEVDataset('/mnt/sda1/songzimeng/officialSEV/train/', transform=data_transform)
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    for index, batch_data in enumerate(dataloader):
        print(index, batch_data['image'].shape, batch_data['label'].shape)

        import IPython
        IPython.embed()
        os._exit(0)
