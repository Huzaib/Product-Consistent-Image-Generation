import os
import time
import datetime
from types import SimpleNamespace
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import argparse
# from deepfillv2 import test_dataset
from deepfillv2 import utils
from torch.utils.data import Dataset

class InpaintDataset(Dataset):
    def __init__(self):
        self.imglist = [INIMAGE]
        self.masklist = [MASKIMAGE]
        self.setsize = RESIZE_TO

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        # image
        img = cv2.imread(self.imglist[index])
        mask = cv2.imread(self.masklist[index])[:, :, 0]
        ## COMMENTING FOR NOW
        # h, w = mask.shape
        # # img = cv2.resize(img, (w, h))
        img = cv2.resize(img, self.setsize)
        mask = cv2.resize(mask, self.setsize)
        ##
        # find the Minimum bounding rectangle in the mask
        """
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cidx, cnt in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(cnt)
            mask[y:y+h, x:x+w] = 255
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = (
            torch.from_numpy(img.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .contiguous()
        )
        mask = (
            torch.from_numpy(mask.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .contiguous()
        )
        return img, mask

def WGAN_tester():

    # Save the model if pre_train == True
    def load_model_generator():
        pretrained_dict = torch.load(
            DEEPFILL_MODEL_PATH, map_location=torch.device(GPU_DEVICE)
        )
        generator.load_state_dict(pretrained_dict)

    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # configurations
    results_path = os.path.dirname(OUTIMAGE)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Build networks
    opt = SimpleNamespace(
        pad_type=PAD_TYPE,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        latent_channels=LATENT_CHANNELS,
        activation=ACTIVATION,
        norm=NORM,
        init_type=INIT_TYPE,
        init_gain=INIT_GAIN,
        use_cuda=CUDA,
        gpu_device=GPU_DEVICE,
    )
    generator = utils.create_generator(opt).eval()
    print("-- INPAINT: Loading Pretrained Model --")
    load_model_generator()

    # To device
    generator = generator.to(GPU_DEVICE)

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = InpaintDataset()

    # Define the dataloader
    dataloader = DataLoader(
        trainset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # ----------------------------------------
    #            Testing
    # ----------------------------------------
    # Testing loop
    for batch_idx, (img, mask) in enumerate(dataloader):
        img = img.to(GPU_DEVICE)
        mask = mask.to(GPU_DEVICE)

        # Generator output
        with torch.no_grad():
            first_out, second_out = generator(img, mask)

        # forward propagation
        first_out_wholeimg = (
            img * (1 - mask) + first_out * mask
        )  # in range [0, 1]
        second_out_wholeimg = (
            img * (1 - mask) + second_out * mask
        )  # in range [0, 1]

        masked_img = img * (1 - mask) + mask
        mask = torch.cat((mask, mask, mask), 1)
        img_list = [second_out_wholeimg]
        name_list = ["second_out"]
        utils.save_sample_png(
            sample_folder=results_path,
            sample_name=os.path.basename(OUTIMAGE),
            img_list=img_list,
            name_list=name_list,
            pixel_max_cnt=255,
        )
        print("-- Inpainting is finished --")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="sample argument parser")
    parser.add_argument(
        "--input_image", type=str, required=True,
        help="Image",
    )
    parser.add_argument(
        "--input_mask", type=str, required=True,
        help="Mask",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output",
    )

    args=parser.parse_args()
    
    
    GPU_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CUDA = True if torch.cuda.is_available() else False
    GPU_ID = 0
    # GPU_DEVICE, CUDA, GPU_ID = "cpu", False, -1
    INIMAGE = args.input_image
    MASKIMAGE = args.input_mask
    OUTIMAGE = args.output_path
    RESIZE_TO = (768, 512)
    DEEPFILL_MODEL_PATH = "/app/model_weights/deepfillv2_WGAN.pth"
    INIT_TYPE = "xavier"
    INIT_GAIN = 0.02
    PAD_TYPE = "zero"
    IN_CHANNELS = 4
    OUT_CHANNELS = 3
    LATENT_CHANNELS = 48
    ACTIVATION = "elu"
    NORM = "in"
    NUM_WORKERS = 0

    WGAN_tester()
