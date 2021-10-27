import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import pretrainedmodels
from models.Asoftmax_linear import AngleLinear

from config import pretrained_model

import pdb

class MainModel(nn.Module):
    def __init__(self, config,args):
        super(MainModel, self).__init__()
        self.use_cdrm = config.use_cdrm
        self.num_classes = config.numcls
        self.backbone_arch = config.backbone
        self.use_Asoftmax = config.use_Asoftmax
        print(self.backbone_arch)

        if self.backbone_arch in dir(models):
            self.model = getattr(models, self.backbone_arch)()

            if self.backbone_arch in pretrained_model:
                self.model.load_state_dict(torch.load(pretrained_model[self.backbone_arch]))
        else:
            if self.backbone_arch in pretrained_model:
                self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000, pretrained=None)
            else:
                self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000)

        if self.backbone_arch == 'resnet50' or self.backbone_arch == 'se_resnet50':
            self.model = nn.Sequential(*list(self.model.children())[:-2])  # remove avgpool and fc
        if self.backbone_arch == 'senet154':
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if self.backbone_arch == 'se_resnext101_32x4d':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'se_resnet101':
            self.model = nn.Sequential(*list(self.model.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)

        if self.use_cdrm:
            if config.cls_2:
                self.classifier_swap = nn.Linear(2048, 2, bias=False)
            if config.cls_2xmul:
                self.classifier_swap = nn.Linear(2048, 2*self.num_classes, bias=False)

            self.blockN = config.swap_num[0]*config.swap_num[1]
            self.classifier_cova = nn.Linear(2048, self.blockN*9, bias=False)

        if self.use_Asoftmax:
            self.Aclassifier = AngleLinear(2048, self.num_classes, bias=False)

    def forward(self, x, last_cont=None):
        x = self.model(x)  # resnet50 backbone

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        out = []
        out.append(self.classifier(x))

        if self.use_cdrm:
            out.append(self.classifier_swap(x))
            out.append(self.classifier_cova(x))

        if self.use_Asoftmax:
            if last_cont is None:
                x_size = x.size(0)
                out.append(self.Aclassifier(x[0:x_size:2]))
            else:
                last_x = self.model(last_cont)
                last_x = self.avgpool(last_x)
                last_x = last_x.view(last_x.size(0), -1)
                out.append(self.Aclassifier(last_x))

        return out
