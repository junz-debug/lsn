import argparse
import os
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import random
import itertools
import random
import matplotlib.pyplot as plt
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
####################################
# We extract image features in advance to improve computational efficiency
# If you want to perform experiments on imagenet100, you should use "get_subset_few_shot_imagenet"
# If you want to perform experiments on imagenet1k, you should use "get_few_shot_imagenet"
####################################
from get_subset_few_shot_imagenet import few_shot_imagenet 
import torchvision.transforms as transforms
import torch.utils.data as data
from coop import CustomCLIP
from torch.cuda.amp import autocast as autocast
from get_score import get_measures
from torchvision import datasets
import torch.nn as nn

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    num_cls = 100
    base_model, _ = clip.load("ViT-B/16", device=device)
    base_model.eval()


    parser = argparse.ArgumentParser("Training ")
    parser.add_argument('--datapath', type=str, default='')
    parser.add_argument('--savepath', type=str, default='')
    parser.add_argument('--num_shot', type=int, default=128)
    parser.add_argument('--train', type=bool, default=True)


    args = parser.parse_args()
    print(args)



    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                            std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    val_preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])


    few_shot_sub_trainset = few_shot_imagenet(args.datapath, args.num_shot, val_preprocess, train = args.train)
    print(len(few_shot_sub_trainset))

    trainloader = data.DataLoader(few_shot_sub_trainset, batch_size = 200, shuffle=False,num_workers=6)

    print('start extrat')
    image_tensor_list = []
    image_target_tensor_list = []
    with torch.no_grad():
        for idx,(image,label) in enumerate(trainloader, 1):
            print(idx)
            image = image.cuda()
            embedding = base_model.visual(image.half())
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            image_tensor_list.append(embedding.half().cpu())
            image_target_tensor_list.append(label)
            
    path = args.savepath    
    image_tensor = torch.cat(image_tensor_list, dim=0)
    image_target_tensor = torch.cat(image_target_tensor_list, dim=0)
    torch.save(image_tensor, path +"feature128.pt")
    torch.save(image_target_tensor, path +"target128.pt")
