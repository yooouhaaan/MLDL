from __future__ import print_function, division
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
import math
import pandas as pd

from PIL import Image


import cv2


import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter
import torchvision
from torchvision import datasets, models, transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class MS1MDataset(Dataset):

    def __init__(self, split):
        self.file_list = 'D:\인하대학교\CVLab\DATA\ID_List.txt'
        self.images = []
        self.labels = []
        self.transformer = data_transforms['train']
        with open(self.file_list) as f:
            files = f.read().splitlines()
        for i, fi in enumerate(files):
            fi = fi.split()
            image = "D:/인하대학교/CVLab/DATA/" + fi[1]
            label = int(fi[0])
            self.images.append(image)
            self.labels.append(label)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = self.transformer(img)

        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.images)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, use_se=True, im_size=112):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()

        if im_size == 112:
            self.fc = nn.Linear(512 * 7 * 7, 512)
        else:  # 224
            self.fc = nn.Linear(512 * 14 * 14, 512)
        self.bn3 = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)

        return x



def resnet50(use_se = False):
    model = ResNet(IRBlock, [3, 4, 6, 3], use_se=False, im_size=112)
    return model

def train_model(model, net, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    features = model(inputs)
                    outputs = net(features, labels)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_corrects.double() / len(dataloader)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'train' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best train Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'model_save.pt')
    return model


class arcface(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(arcface, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        #print(output)

        return output

train_dataset = MS1MDataset('train')
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 디바이스 설정

num_classes = 86876
# normal classifier
net = arcface(512, num_classes, s=30, m=0.5, easy_margin=False)#nn.Sequential(nn.Linear(512, num_classes))
# Feature extractor backbone, input is 112x112 image output is 512 feature vector
model_ft = resnet50()

net = net.to(device)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.1)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.1)

model = train_model(model_ft, net, criterion, optimizer_ft, exp_lr_scheduler)

#model_file = torch.load("./model_save.pt")# 사전학습 모델 다운 후 알맞은 경로 지정
#model = ResNet(IRBlock, [3, 4, 6, 3], use_se=False, im_size=112).to(device) # 모델 정의
#model.load_state_dict(model_file) # 사전 학습된 모델의 weight 로 업데이트
model.eval() # 모델을 평가 모드 설정

submission = pd.read_csv("D:\인하대학교\CVLab\DATA\sample_submission.csv")

left_test_paths = list()
right_test_paths = list()

for i in range(len(submission)):
    left_test_paths.append(submission['face_images'][i].split()[0])
    right_test_paths.append(submission['face_images'][i].split()[1])

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Left Side Image Processing

left_test = list()
for left_test_path in left_test_paths:
    img = Image.open("D:\인하대학교\CVLab\DATA\test" + left_test_path + '.jpg').convert("RGB")# 경로 설정 유의(ex .inha/test)
    img = data_transform(img) # 이미지 데이터 전처리
    left_test.append(img)
left_test = torch.stack(left_test)
#print(left_test.size()) # torch.Size([6000, 3, 112, 112])

left_infer_result_list = list()
with torch.no_grad():
    '''
    메모리 부족으로 6,000개 (배치) 한번에 입력으로 넣지 않고 1,000개 씩 입력으로 줌
    '''
    batch_size = 1000
    for i in range(0, 6):
        i = i * batch_size
        tmp_left_input = left_test[i:i+batch_size]
        #print(tmp_input.size()) # torch.Size([1000, 3, 112, 112])
        left_infer_result = model(tmp_left_input.to(device))
        #print(left_infer_result.size()) # torch.Size([1000, 512])
        left_infer_result_list.append(left_infer_result)

    left_infer_result_list = torch.stack(left_infer_result_list, dim=0).view(-1, 512)
    #print(left_infer_result_list.size()) # torch.Size([6000, 512])

# Right Side Image Processing

right_test = list()
for right_test_path in right_test_paths:
    img = Image.open("D:\인하대학교\CVLab\DATA\test" + right_test_path + '.jpg').convert("RGB") # 경로 설정 유의 (ex. inha/test)
    img = data_transform(img)# 이미지 데이터 전처리
    right_test.append(img)
right_test = torch.stack(right_test)
#print(right_test.size()) # torch.Size([6000, 3, 112, 112])

right_infer_result_list = list()
with torch.no_grad():
    '''
    메모리 부족으로 6,000개 (배치) 한번에 입력으로 넣지 않고 1,000개 씩 입력으로 줌
    '''
    batch_size = 1000
    for i in range(0, 6):
        i = i * batch_size
        tmp_right_input = right_test[i:i+batch_size]
        #print(tmp_input.size()) # torch.Size([1000, 3, 112, 112])
        right_infer_result = model(tmp_right_input.to(device))
        #print(left_infer_result.size()) # torch.Size([1000, 512])
        right_infer_result_list.append(right_infer_result)

    right_infer_result_list = torch.stack(right_infer_result_list, dim=0).view(-1, 512)
    #print(right_infer_result_list.size()) # torch.Size([6000, 512])

def cos_sim(a, b):
    return F.cosine_similarity(a, b)

cosin_similarity = cos_sim(left_infer_result_list, right_infer_result_list)

submission = pd.read_csv("D:\인하대학교\CVLab\DATA\sample_submission.csv")
submission['answer'] = cosin_similarity.tolist()
#submission.loc['answer'] = submission['answer']
submission.to_csv('D:\인하대학교\CVLab\DATA\sample_submission.csv', index=False)

print(submission)