import os
import pandas as pd
import torch

from transforms import transforms
from utils.autoaugment import ImageNetPolicy

# pretrained model checkpoints
pretrained_model = {'resnet50': '../pretrained_models/resnet50-19c8e357.pth'}
# pretrained_model = {'resnet50': '/export/home/s5058775/kiki/CDRM_fc_PR/models/pretrained/resnet50-19c8e357.pth',}


# transforms dict
def load_data_transformers(resize_reso=440, crop_reso=384, swap_num=[2, 2]):
    Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
       	'swap': transforms.Compose([
            transforms.Randomswap((swap_num[0], swap_num[1])),
        ]),
        'common_aug': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso,crop_reso)),
            transforms.RandomHorizontalFlip(),
        ]),
        'train_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test_totensor': transforms.Compose([
            # transforms.Resize((resize_reso, resize_reso)),
            # transforms.CenterCrop((crop_reso, crop_reso)),  # fine-grained datasets
            transforms.Resize((crop_reso, crop_reso)),   # ultra fine-grained datasets: soybean / cotton
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'None': None,
    }
    return data_transforms


class LoadConfig(object):
    def __init__(self, args, version):
        if version == 'train':
            get_list = ['train', 'val']
        elif version == 'val':
            get_list = ['val']
        elif version == 'test':
            get_list = ['test']
        else:
            raise Exception("train/val/test ???\n")

        ###############################
        #### add dataset info here ####
        ###############################

        # put image data in $PATH/data
        # put annotation txt file in $PATH/anno

        if args.dataset == 'CUB':
            self.dataset = args.dataset
            self.rawdata_root = '../Datasets/CUB_200_2011/images'
            self.anno_root = '../Datasets/CUB_200_2011/anno'
            self.numcls = 200
        elif args.dataset == 'STCAR':
            self.dataset = args.dataset
            self.rawdata_root = '../Datasets/st_car/images'
            self.anno_root = '../Datasets/st_car/anno'
            self.numcls = 196
        elif args.dataset == 'COTTON':
            self.dataset = args.dataset
            self.rawdata_root = '../data/COTTON/images'
            self.anno_root = '../data/COTTON/anno'
            self.numcls = 80
        elif args.dataset == 'Soybean200':
            self.dataset = args.dataset
            self.rawdata_root = '../data/soybean200/images'
            self.anno_root = '../data/soybean200/anno'
            self.numcls = 200
        elif args.dataset == 'Soybean2000':
            self.dataset = args.dataset
            self.rawdata_root = '../data/soybean2000/images'
            self.anno_root = '../data/soybean2000/anno'
            self.numcls = 1938
        elif args.dataset == 'R1':
            self.dataset = args.dataset
            self.rawdata_root = '../data/R1/images'
            self.anno_root = '../data/R1/anno'
            self.numcls = 198
        elif args.dataset == 'R3':
            self.dataset = args.dataset
            self.rawdata_root = '../data/R3/images'
            self.anno_root = '../data/R3/anno'
            self.numcls = 198
        elif args.dataset == 'R4':
            self.dataset = args.dataset
            self.rawdata_root = '../data/R4/images'
            self.anno_root = '../data/R4/anno'
            self.numcls = 198
        elif args.dataset == 'R5':
            self.dataset = args.dataset
            self.rawdata_root = '../data/R5/images'
            self.anno_root = '../data/R5/anno'
            self.numcls = 198
        elif args.dataset == 'R6':
            self.dataset = args.dataset
            self.rawdata_root = '../data/R6/images'
            self.anno_root = '/export/home/s5058775/kiki/data/R6/anno'
            self.numcls = 198
        elif args.dataset == 'soybean_gene':
            self.dataset = args.dataset
            self.rawdata_root = '../data/soybean_gene/images'
            self.anno_root = '..//data/soybean_gene/anno'
            self.numcls = 1110
        else:
            raise Exception('dataset not defined ???')

        # annotation file organized as :
        # path/image_name cls_num\n

        if 'train' in get_list:
             self.train_anno = pd.read_csv(os.path.join(self.anno_root, 'train.txt'),
                                           sep=" ",
                                           header=None,
                                           names=['ImageName', 'label'])

        if 'val' in get_list:
            self.val_anno = pd.read_csv(os.path.join(self.anno_root, 'val.txt'),
                                           sep=" ",
                                           header=None,
                                           names=['ImageName', 'label'])

        if 'test' in get_list:
            self.test_anno = pd.read_csv(os.path.join(self.anno_root, 'test.txt'),
                                           sep=" ",
                                           header=None,
                                           names=['ImageName', 'label'])

        self.swap_num = args.swap_num

        self.log_folder = './logs/'
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)
        self.save_dir = self.log_folder + args.dataset
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # self.save_dir = '/export/home/s5058775/kiki/CDRM_fc_PR/results_kiki/' + args.dataset  # './net_model/'

        self.use_cdrm = True
        self.mask_num = 1
        self.backbone = args.backbone
        self.use_backbone = False if self.use_cdrm else True
        self.use_Asoftmax = False
        self.use_focal_loss = False
        self.use_fpn = False
        self.use_hier = False

        self.weighted_sample = False
        self.cls_2 = True
        self.cls_2xmul = False
