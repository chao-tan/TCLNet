import os.path
from datasets.cdataset import cdataset
import cv2

import random
import numpy as np
import torch
from torchvision import transforms


def normalization(data,norm_type="standard"):
    if norm_type == "standard":
        mean = [0.5,0.5,0.5]
        std = [0.5,0.5,0.5]
    elif norm_type == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise NotImplementedError("norm_type mismatched!")

    transform = transforms.Normalize(mean,std,inplace=False)
    return transform(data)



class heatmapDataset(cdataset):
    def __init__(self, config):
        cdataset.__init__(self, config)
        self.input_paths = os.path.join(config['dataroot'], str.upper(config['status'])+"_INPUT")
        self.heatmap_paths = os.path.join(config['dataroot'], str.upper(config['status'])+"_HEATMAP")
        self.labels = np.load(os.path.join(config['dataroot'],str.upper(config['status'])+"_LABEL"+".npy"))
        self.dataset_len = len(os.listdir(self.input_paths))
        self.config = config


    def __getitem__(self, index):
        image_input = cv2.imread(os.path.join(self.input_paths,str(index+1)+".png")).astype(np.float)
        image_heatmap = cv2.imread(os.path.join(self.heatmap_paths,str(index+1)+".png"),cv2.IMREAD_GRAYSCALE).astype(np.float)
        kpoint = self.labels[index+1]

        if self.config['status'] == 'train':
            if self.config['flip'] is True:
                flip_tag = random.sample(range(-1,2),1)[0]
                image_input = cv2.flip(image_input,flip_tag)
                image_heatmap = cv2.flip(image_heatmap,flip_tag)
                if flip_tag == -1: kpoint = 1.-kpoint
                elif flip_tag == 0: kpoint[1] = 1.-kpoint[1]
                else:kpoint[0] = 1.-kpoint[0]

            if self.config['crop_scale'] > 1:
                img_size = self.config['img_size']
                crop_size = int(self.config['crop_scale']*img_size)
                image_input = cv2.resize(image_input,(crop_size,crop_size))
                image_heatmap = cv2.resize(image_heatmap,(crop_size,crop_size))
                ranp = random.sample(range(int((crop_size-img_size)*0.5)),2)

                image_input = image_input[ranp[0]:ranp[0]+img_size,ranp[1]:ranp[1]+img_size]
                image_heatmap = image_heatmap[ranp[0]:ranp[0] + img_size, ranp[1]:ranp[1] + img_size]
                kpoint[0] = (kpoint[0]*img_size*self.config['crop_scale']-ranp[1])/img_size
                kpoint[1] = (kpoint[1]*img_size*self.config['crop_scale']-ranp[0])/img_size

        image_input_show = torch.from_numpy(image_input).permute(2,0,1).float()/255.
        image_heatmap = torch.from_numpy(image_heatmap).unsqueeze(0).float()

        image_input = normalization(image_input_show,str(self.config['norm_type']))
        image_heatmap = image_heatmap / 255.

        label = torch.Tensor([kpoint[0],kpoint[1]]).float()

        return {'IMAGE': image_input,
                "IMAGE_SHOW": image_input_show,
                'HEATMAP':image_heatmap,
                "LABEL": label,
                "PATH": index+1
                }


    def __len__(self):
        return self.dataset_len
