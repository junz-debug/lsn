from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
import os
import torchvision.transforms as transforms
import random
import torch.utils.data as data
from show_pictures import show_pictures
class few_shot_imagenet(Dataset):
    def __init__(self, root_path, num_shot, transform, cls_num=100, train = True):
        self.root = f"{root_path}"
        self.num_shot = num_shot
        self.transform = transform

        label_name_list = os.listdir(self.root)#n0140764.....
        sorted_list = sorted(label_name_list)
        #print(sorted_list)
        self.label = []
        self.data = []
        
        subset_list = []
        with open(os.path.join('class_list_'+str(cls_num)+'.txt')) as file:
            for line in file.readlines():
                cls = line.strip()
                subset_list.append(cls)
        idx = 0
        for label_name in sorted_list:
            if label_name not in subset_list:     #不在子类中!!!!900类
                continue
            images_list = os.listdir(f"{self.root}/{label_name}")#某一类所有的图片名
            
            if train:
                targetfile = random.sample(images_list, self.num_shot)#训练时加载few shot
            else:
                targetfile = images_list#测试时加载所有图片
            for name in targetfile:
                img_path = f"{self.root}/{label_name}/{name}"
                self.label.append(idx)#idx作为类别
                self.data.append(img_path)
            idx = idx + 1
        self.label = torch.tensor(self.label, dtype=torch.long)
    def __getitem__(self, index):
        img_path, target = self.data[index], self.label[index]
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print('image error')
            img_path, target = self.data[index + 1], self.label[index + 1]
            img = Image.open(img_path).convert('RGB')
        #img = Image.open(img_path).convert('RGB')
        #img = np.array(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)
    
if __name__ == '__main__':
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    num_cls = 100
    few_shot_dataset = few_shot_imagenet('/data/linshiqi047/imagenet/train', 4, preprocess, cls_num = num_cls, train = True)
    few_shot_loader = data.DataLoader(few_shot_dataset, 
                              batch_size=4, 
                              shuffle=False,
                              num_workers=2)
    with open(os.path.join('cls_name_' + str(num_cls) + '.npy'), 'rb') as f:
        imagenet_cls = np.load(f)#所有类别名
    for idx, (image,label) in enumerate(few_shot_loader):
        print(label)
        print(imagenet_cls[label])
        print(image.shape)
        print(torch.max(image))
        show_pictures(image,1,1,2)