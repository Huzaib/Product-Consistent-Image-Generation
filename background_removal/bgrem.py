import cv2
import gradio as gr
import os
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import gdown
import matplotlib.pyplot as plt
import warnings
from math import ceil

warnings.filterwarnings("ignore")

# project imports
from background_removal.data_loader_cache import normalize, im_reader, im_preprocess 
from background_removal.models import *

#Helpers
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
    
class GOSNormalize(object):
    '''
    Normalize the Image using torch.transforms
    '''
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,image):
        image = normalize(image,self.mean,self.std)
        return image


transform =  transforms.Compose([GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0])])

def load_image(im_path, hypar):
    im = im_reader(im_path)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im,255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0) # make a batch of image, shape


def build_model(hypar,device):
    net = hypar["model"]#GOSNETINC(3,1)

    # convert to half precision
    if(hypar["model_digit"]=="half"):
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    if(hypar["restore_model"]!=""):
        net.load_state_dict(torch.load(hypar["model_path"]+"/"+hypar["restore_model"], map_location=device))
        net.to(device)
    net.eval()  
    return net

    
def predict(net,  inputs_val, shapes_val, hypar, device):
    '''
    Given an Image, predict the mask
    '''
    net.eval()

    if(hypar["model_digit"]=="full"):
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

  
    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device) # wrap inputs in Variable
   
    ds_val = net(inputs_val_v)[0] # list of 6 results

    pred_val = ds_val[0][0,:,:,:] # B x 1 x H x W    # we want the first one which is the most accurate prediction

    ## recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val,0),(shapes_val[0][0],shapes_val[0][1]),mode='bilinear'))

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val-mi)/(ma-mi) # max = 1

    if device == 'cuda': torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy()*255).astype(np.uint8) # it is the mask we need
    
# Set Parameters
hypar = {} # paramters for inferencing


hypar["model_path"] ="background_removal/bg_models" ## load trained weights from this path
hypar["restore_model"] = "isnet-general-use.pth" ## name of the to-be-loaded weights
hypar["interm_sup"] = False ## indicate if activate intermediate feature supervision

##  choose floating point accuracy --
hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
hypar["seed"] = 0

hypar["cache_size"] = [1024, 1024] ## cached input spatial resolution, can be configured into different size

## data augmentation parameters ---
hypar["input_size"] = [1024, 1024] ## mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
hypar["crop_size"] = [1024, 1024] ## random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation

hypar["model"] = ISNetDIS()

 # Build Model
net = build_model(hypar, device)

def bgrem_inference(image):
    image_path = image
    print(f"Image Path: {image_path}")
  
    image_tensor, orig_size = load_image(image_path, hypar) 
    mask = predict(net, image_tensor, orig_size, hypar, device)
  
    pil_mask = Image.fromarray(mask).convert('L')
    im_rgb = Image.open(image).convert("RGB")
  
    im_rgba = im_rgb.copy()
    im_rgba.putalpha(pil_mask)
    
    image_path = '.'.join(image.split('.')[:-1])+'.png'
    im_rgba.save(image_path)
    return image_path


title = "Highly Accurate Dichotomous Image Segmentation"
description = "This is an unofficial demo for DIS, a model that can remove the background from a given image. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below.<br>GitHub: https://github.com/xuebinqin/DIS<br>Telegram bot: https://t.me/restoration_photo_bot<br>[![](https://img.shields.io/twitter/follow/DoEvent?label=@DoEvent&style=social)](https://twitter.com/DoEvent)"
article = "<div><center><img src='https://visitor-badge.glitch.me/badge?page_id=max_skobeev_dis_cmp_public' alt='visitor badge'></center></div>"


if __name__ == '__main__':
    
    filename = "./test_img.png"
    bgrem_inference(filename)