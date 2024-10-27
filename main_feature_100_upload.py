import argparse
import os
import numpy as np
import torch
import clip
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from coop import CustomCLIP_feaure
from torch.cuda.amp import autocast as autocast
from get_score import get_measures


parser = argparse.ArgumentParser("Training ")
parser.add_argument('--ctx_num_no', type=int, default=16)
parser.add_argument('--ctx_num_yes', type=int, default=16)
parser.add_argument('--train_epochs_no', type=int, default=100)
parser.add_argument('--train_epochs_yes', type=int, default=50)
parser.add_argument('--lr_no', type=float, default=0.002)
parser.add_argument('--lr_yes', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--path_id_label', type=str, default='cls_name_100.npy')


args = parser.parse_args()
print(args)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
num_cls = 100
def obtain_ImageNet_classes():
    with open(os.path.join(args.path_id_label), 'rb') as f:
        imagenet_cls = np.load(f)
    return imagenet_cls
test_labels = obtain_ImageNet_classes()

print(test_labels)
clip_model, preprocess = clip.load("ViT-B/16", device = 'cpu')
_tokenizer = _Tokenizer()
model_yes = CustomCLIP_feaure(test_labels, clip_model, ctx_num = args.ctx_num_yes,  class_specific = False, class_token_position = 'end')
model_no1 = CustomCLIP_feaure(test_labels, clip_model, ctx_num = args.ctx_num_no,  class_specific = True, class_token_position = 'end')
model_no2 = CustomCLIP_feaure(test_labels, clip_model, ctx_num = args.ctx_num_no,  class_specific = True, class_token_position = 'end')
model_no3 = CustomCLIP_feaure(test_labels, clip_model, ctx_num = args.ctx_num_no,  class_specific = True, class_token_position = 'end')

print("Turning off gradients in both the image and the text encoder")
for name, param in model_yes.named_parameters():
    if "prompt_learner" not in name:
        param.requires_grad_(False)
for name, param in model_no1.named_parameters():
    if "prompt_learner" not in name:
        param.requires_grad_(False)

for name, param in model_no2.named_parameters():
    if "prompt_learner" not in name:
        param.requires_grad_(False)

for name, param in model_no3.named_parameters():
    if "prompt_learner" not in name:
        param.requires_grad_(False)

model_yes.to(device)
model_no1.to(device)
model_no2.to(device)
model_no3.to(device)

    
max_epochs_yes = args.train_epochs_yes
max_epochs_no = args.train_epochs_no
# optim_yes = torch.optim.SGD(model_yes.prompt_learner.parameters(), lr=args.lr_yes)
optim_yes = torch.optim.SGD(model_yes.prompt_learner.parameters(), lr=args.lr_yes, momentum=0.9, weight_decay=5e-4)
sched_yes = torch.optim.lr_scheduler.CosineAnnealingLR(optim_yes, T_max = max_epochs_yes)
optim_no1 = torch.optim.Adam(model_no1.prompt_learner.parameters(), lr=args.lr_no)
sched_no1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_no1, T_max = max_epochs_no)
optim_no2 = torch.optim.Adam(model_no2.prompt_learner.parameters(), lr=args.lr_no)
sched_no2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_no2, T_max = max_epochs_no)
optim_no3 = torch.optim.Adam(model_no3.prompt_learner.parameters(), lr=args.lr_no)
sched_no3 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_no3, T_max = max_epochs_no)

parser.add_argument('--train_feature', type=str, default='G:/dataset/lsn/trainfeature128shot.pt')
parser.add_argument('--train_label', type=str, default='G:/dataset/lsn/traintarget128shot.pt')
parser.add_argument('--test_in_feature', type=str, default='G:/dataset/lsn/testfeature_yes.pt')
parser.add_argument('--test_in_label', type=str, default='G:/dataset/lsn/testtarget_yes.pt')
parser.add_argument('--test_out1_feature', type=str, default='G:/dataset/lsn/testfeatureiNaturalist.pt')
parser.add_argument('--test_out1_label', type=str, default='G:/dataset/lsn/testtargetiNaturalist.pt')
parser.add_argument('--test_out2_feature', type=str, default='G:/dataset/lsn/testfeatureSUN.pt')
parser.add_argument('--test_out2_label', type=str, default='G:/dataset/lsn/testfeatureSUN.pt')
parser.add_argument('--test_out3_feature', type=str, default='G:/dataset/lsn/testfeaturePlaces.pt')
parser.add_argument('--test_out3_label', type=str, default='G:/dataset/lsn/testtargetPlaces.pt')
parser.add_argument('--test_out4_feature', type=str, default='G:/dataset/lsn/testfeaturedtd.pt')
parser.add_argument('--test_out4_label', type=str, default='G:/dataset/lsn/testtargetdtd.pt')


class clip_feature(Dataset):
    def __init__(self, path_data, path_label, train = True):
        super().__init__()
        self.features = torch.load(path_data)
        self.targets =  torch.load(path_label)
        if train == True:
            self.features = self.features.cuda()
            self.targets = self.targets.cuda()
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    
args = parser.parse_args()
print(args)

load_path_data = args.train_feature
load_path_label = args.train_label

train_data_128 = clip_feature(load_path_data, load_path_label, train =True)
#train_data_64 = clip_feature(load_path_64_data, load_path_64_label, train =True)
print(len(train_data_128))

train_loader = torch.utils.data.DataLoader(
    train_data_128, batch_size=args.batch_size, shuffle=True)


load_path_test_in_data = args.test_in_feature
load_path_test_in_label = args.test_in_label
test_data_in = clip_feature(load_path_test_in_data, load_path_test_in_label)
test_loader_in = torch.utils.data.DataLoader(
            test_data_in,batch_size=256, shuffle=False)

load_path_test_data_out1 = args.test_out1_feature
load_path_test_label_out1 = args.test_out1_label
test_data_out1 = clip_feature(load_path_test_data_out1, load_path_test_label_out1, train =False)
test_loader_out1 = torch.utils.data.DataLoader(
            test_data_out1,batch_size=256, shuffle=False)

load_path_test_data_out2 = args.test_out2_feature
load_path_test_label_out2 = args.test_out2_label
test_data_out2 = clip_feature(load_path_test_data_out2, load_path_test_label_out2, train =False)
test_loader_out2 = torch.utils.data.DataLoader(
            test_data_out2,batch_size=256, shuffle=False)

load_path_test_data_out3 = args.test_out3_feature
load_path_test_label_out3= args.test_out3_label
test_data_out3 = clip_feature(load_path_test_data_out3, load_path_test_label_out3, train =False)
test_loader_out3 = torch.utils.data.DataLoader(
            test_data_out3,batch_size=256, shuffle=False)

load_path_test_data_out4 = args.test_out4_feature
load_path_test_label_out4 = args.test_out4_label
test_data_out4 = clip_feature(load_path_test_data_out4, load_path_test_label_out4, train =False)
test_loader_out4 = torch.utils.data.DataLoader(
            test_data_out4,batch_size=256, shuffle=False)

def test(model_yes, model_no1, model_no2, model_no3, test_loader_in, test_loader_out):#用最低的位置的正分数相加
    score_in_yes = []
    score_out_yes = []
    
    score_in_plus = []
    score_out_plus = []
    
    score_in_no = []
    score_out_no = []
    
    model_yes.eval()
    model_no1.eval()
    model_no2.eval()
    model_no3.eval()
    print('test')
    with torch.no_grad():
        text_features_yes = model_yes()
        text_features_no1 = model_no1()
        text_features_no2 = model_no2()
        text_features_no3 = model_no3()

    with torch.no_grad():
        for idx,(images,labels) in tqdm(enumerate(test_loader_in)):
            images = images.to(device)
            image_features = images 
            with autocast():
                outputs_yes = image_features @ text_features_yes.t()
                outputs_no1 = image_features @ text_features_no1.t()
                outputs_no2 = image_features @ text_features_no2.t()
                outputs_no3 = image_features @ text_features_no3.t()
            
            similarity_yes = F.softmax(outputs_yes, dim=1)
            similarity_no1 = F.softmax((1-outputs_no1)/5 , dim=1)
            similarity_no2 = F.softmax((1-outputs_no2)/5 , dim=1)
            similarity_no3 = F.softmax((1-outputs_no3)/5 , dim=1)

            # similarity_no1 = -F.softmax((outputs_no1)/5 , dim=1)
            # similarity_no2 = -F.softmax((outputs_no2)/5 , dim=1)
            # similarity_no3 = -F.softmax((outputs_no3)/5 , dim=1)

            max_in_yes, index_in_yes = torch.max(similarity_yes,dim = 1)
            max_in_no1, index_in_no = torch.max(similarity_no1,dim = 1)
            max_in_no2, index_in_no = torch.max(similarity_no2,dim = 1)
            max_in_no3, index_in_no = torch.max(similarity_no3,dim = 1)

            max_in_no = (max_in_no1 + max_in_no2 + max_in_no3) /3
            max_in_yes_plus = max_in_yes + max_in_no
                
            smax_yes = to_np(max_in_yes)
            smax_no = to_np(max_in_no)
            smax_plus = to_np(max_in_yes_plus)

            score_in_yes.append(smax_yes)
            score_in_no.append(smax_no)
            score_in_plus.append(smax_plus)
        
        in_score_yes = concat(score_in_yes)[:len(test_loader_in.dataset)].copy() 
        in_score_no = concat(score_in_no)[:len(test_loader_in.dataset)].copy() 
        in_score_plus = concat(score_in_plus)[:len(test_loader_in.dataset)].copy() 
    
        for idx,(images,labels) in tqdm(enumerate(test_loader_out)):
            images = images.to(device)

            image_features = images 
            with autocast():
                outputs_yes = image_features @ text_features_yes.t()
                outputs_no1 = image_features @ text_features_no1.t()
                outputs_no2 = image_features @ text_features_no2.t()
                outputs_no3 = image_features @ text_features_no3.t()
            
            similarity_yes = F.softmax(outputs_yes, dim=1)
            similarity_no1 = F.softmax((1-outputs_no1)/5 , dim=1)
            similarity_no2 = F.softmax((1-outputs_no2)/5 , dim=1)
            similarity_no3 = F.softmax((1-outputs_no3)/5 , dim=1)

            # similarity_no1 = -F.softmax((outputs_no1)/5 , dim=1)
            # similarity_no2 = -F.softmax((outputs_no2)/5 , dim=1)
            # similarity_no3 = -F.softmax((outputs_no3)/5 , dim=1)
            
            max_out_yes, index_out_yes = torch.max(similarity_yes,dim = 1)
            max_out_no1, index_out_no = torch.max(similarity_no1,dim = 1)
            max_out_no2, index_out_no = torch.max(similarity_no2,dim = 1)
            max_out_no3, index_out_no = torch.max(similarity_no3,dim = 1)
            max_out_no = (max_out_no1 + max_out_no2 + max_out_no3)/3
            max_out_yes_plus = max_out_yes + max_out_no

            smax_yes = to_np(max_out_yes)
            smax_no = to_np(max_out_no)
            smax_plus = to_np(max_out_yes_plus)

            score_out_yes.append(smax_yes)
            score_out_no.append(smax_no)
            score_out_plus.append(smax_plus)
        out_score_yes = concat(score_out_yes)[:len(test_loader_out.dataset)].copy() 
        out_score_no = concat(score_out_no)[:len(test_loader_out.dataset)].copy() 
        out_score_plus = concat(score_out_plus)[:len(test_loader_out.dataset)].copy() 
        auroc, aupr, fpr = get_measures(in_score_plus, out_score_plus)
        auroc1, aupr1, fpr1 = get_measures(in_score_yes, out_score_yes)
        auroc2, aupr2, fpr2 = get_measures(in_score_no, out_score_no)
        print('method:lsn, auroc is %f,fpr is %f'%(auroc,fpr))
        print('method:yes, auroc is %f,fpr is %f'%(auroc1,fpr1))
        print('method:no, auroc is %f,fpr is %f'%(auroc2,fpr2))
        return auroc, fpr, auroc1, fpr1, auroc2, fpr2 
    
to_np = lambda x: x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)
score_in = []
score_out = []
print('start training')
def calculate_cosine_similarity(tensor1, tensor2):
    return torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=-1)
for epoch in range(max_epochs_no):
    print(epoch)
    with torch.no_grad():
        model_no1.eval()
        model_no2.eval()
        model_no3.eval()
        text_features_no1 = model_no1()
        text_features_no2 = model_no2()
        text_features_no3 = model_no3()

    for idx,(images,labels) in tqdm(enumerate(train_loader, 1)):

        if epoch < max_epochs_yes:
            model_yes.train()
            model_yes.zero_grad()
            optim_yes.zero_grad()
            with autocast():
                text_features = model_yes()
                image_features = images
                similarity = image_features @ text_features.t()
                output = F.softmax(similarity * 100, dim = 1)

                ce_loss = F.cross_entropy(output, labels)
                loss = ce_loss 

                loss.backward()
                optim_yes.step()
    sched_yes.step()

    for idx,(images,labels) in tqdm(enumerate(train_loader, 1)):
        #images, labels = images.cuda(), labels.cuda()
        model_no1.train()
        model_no1.zero_grad()
        optim_no1.zero_grad()
        with autocast():
            text_features_no1 = model_no1() 
            image_features = images
            
            similarity = image_features @ text_features_no1.t() 
            output = F.softmax((1-similarity) * 20, dim = 1)
            ce_loss = F.cross_entropy(output, labels)
            loss_m = torch.mean((calculate_cosine_similarity(text_features_no1, text_features_no2) + calculate_cosine_similarity(text_features_no1, text_features_no3))/2)
            loss = ce_loss + loss_m * 1
            if idx == 100:
                print(ce_loss)
                print(loss_m)
                
            loss.backward()
            optim_no1.step()
    sched_no1.step()
    with torch.no_grad():
        model_no1.eval()
        text_features_no1 = model_no1()
    for idx,(images,labels) in tqdm(enumerate(train_loader, 1)):
        #images, labels = images.cuda(), labels.cuda()
        model_no2.train()
        model_no2.zero_grad()
        optim_no2.zero_grad()
        with autocast():
            text_features_no2 = model_no2() 
            image_features = images
            
            similarity = image_features @ text_features_no2.t() 
            output = F.softmax((1-similarity) * 20, dim = 1)
            ce_loss = F.cross_entropy(output, labels)
            loss_m = torch.mean((calculate_cosine_similarity(text_features_no1, text_features_no2) + calculate_cosine_similarity(text_features_no2, text_features_no3))/2)
            loss = ce_loss + loss_m * 1
                
            loss.backward()
            optim_no2.step()
    sched_no2.step()
    with torch.no_grad():
        model_no2.eval()
        text_features_no2 = model_no2()
    for idx,(images,labels) in tqdm(enumerate(train_loader, 1)):
        #images, labels = images.cuda(), labels.cuda()
        model_no3.train()
        model_no3.zero_grad()
        optim_no3.zero_grad()
        with autocast():
            text_features_no3 = model_no3() 
            image_features = images
            
            similarity = image_features @ text_features_no3.t() 
            output = F.softmax((1-similarity) * 20, dim = 1)
            ce_loss = F.cross_entropy(output, labels)
            loss_m = torch.mean((calculate_cosine_similarity(text_features_no1, text_features_no3) + calculate_cosine_similarity(text_features_no2, text_features_no3))/2)
            loss = ce_loss + loss_m * 1
            loss.backward()
            optim_no3.step()
    sched_no3.step()
    # if idx%10==0:
    #     auroc1, fpr1, auroc1_yes, fpr1_yes, auroc1_no, fpr1_no = test(model_yes, model_no1, model_no2, model_no3, test_loader_in, test_loader_out1)
        
    auroc1, fpr1, auroc1_yes, fpr1_yes, auroc1_no, fpr1_no = test(model_yes, model_no1, model_no2, model_no3, test_loader_in, test_loader_out1)
    auroc2,fpr2, auroc2_yes, fpr2_yes, auroc2_no, fpr2_no = test(model_yes, model_no1, model_no2, model_no3, test_loader_in, test_loader_out2)
    auroc3,fpr3, auroc3_yes, fpr3_yes, auroc3_no, fpr3_no = test(model_yes, model_no1, model_no2, model_no3, test_loader_in, test_loader_out3)
    auroc4,fpr4, auroc4_yes, fpr4_yes, auroc4_no, fpr4_no = test(model_yes, model_no1, model_no2, model_no3, test_loader_in, test_loader_out4)
    auroc = (auroc1 + auroc2 + auroc3 + auroc4) / 4
    fpr = (fpr1 + fpr2 + fpr3 + fpr4) / 4
    print('average:')
    print(auroc)
    print(fpr)
