import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np
from glob import glob

random.seed(1143)

#import torch.nn.functional as F
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

class PatchDataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(PatchDataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))

        self.inp_filenames = [os.path.join(rgb_dir, 'low', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'high', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, filename




class PatchDataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(PatchDataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))

        self.inp_filenames = [os.path.join(rgb_dir, 'low', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'high', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target
        self.mul = 16

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')
        #inp_img = TF.to_tensor(inp_img)
        #tar_img = TF.to_tensor(tar_img)
        w, h = inp_img.size
        #h, w = inp_img.shape[2], inp_img.shape[3]
        H, W = ((h + self.mul) // self.mul) * self.mul, ((w + self.mul) // self.mul) * self.mul
        padh = H - h if h % self.mul != 0 else 0
        padw = W - w if w % self.mul != 0 else 0
        inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, filename


def populate_train_list(images_path, mode='train'):
    # print(images_path)
    image_list_lowlight = glob(images_path + '*.png')
    train_list = image_list_lowlight
    if mode == 'train':
        random.shuffle(train_list)

    return train_list

class wholeDataLoader(Dataset):

    def __init__(self, images_path, mode='train'):
        images_path = images_path + '/low/'
        self.train_list = populate_train_list(images_path, mode)
        self.mode = mode
        self.data_list = self.train_list
        print("Total examples:", len(self.train_list))
    

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        #ps = 256 # Training Patch Size 
        if self.mode == 'train':
            data_lowlight = Image.open(data_lowlight_path).convert('RGB')
            data_highlight = Image.open(data_lowlight_path.replace('low', 'high')).convert('RGB')
            w, h = data_lowlight.size
            data_lowlight = TF.to_tensor(data_lowlight)
            data_highlight = TF.to_tensor(data_highlight)
            hh, ww = data_highlight.shape[1], data_highlight.shape[2]

            # rr = random.randint(0, hh - ps)
            # cc = random.randint(0, ww - ps)
            aug = random.randint(0, 3)

            # Crop patch
            # data_lowlight = data_lowlight[:, rr:rr + ps, cc:cc + ps]
            # data_highlight = data_highlight[:, rr:rr + ps, cc:cc + ps]

            # Data Augmentations
            if aug == 1:
                data_lowlight = data_lowlight.flip(1)
                data_highlight = data_highlight.flip(1)
            elif aug == 2:
                data_lowlight = data_lowlight.flip(2)
                data_highlight = data_highlight.flip(2)
            # elif aug == 3:
            #     data_lowlight = torch.rot90(data_lowlight, dims=(1, 2))
            #     data_highlight = torch.rot90(data_highlight, dims=(1, 2))
            # elif aug == 4:
            #     data_lowlight = torch.rot90(data_lowlight, dims=(1, 2), k=2)
            #     data_highlight = torch.rot90(data_highlight, dims=(1, 2), k=2)
            # elif aug == 5:
            #     data_lowlight = torch.rot90(data_lowlight, dims=(1, 2), k=3)
            #     data_highlight = torch.rot90(data_highlight, dims=(1, 2), k=3)
            # elif aug == 6:
            #     data_lowlight = torch.rot90(data_lowlight.flip(1), dims=(1, 2))
            #     data_highlight = torch.rot90(data_highlight.flip(1), dims=(1, 2))
            # elif aug == 7:
            #     data_lowlight = torch.rot90(data_lowlight.flip(2), dims=(1, 2))
            #     data_highlight = torch.rot90(data_highlight.flip(2), dims=(1, 2))

            filename = os.path.splitext(os.path.split(data_lowlight_path)[-1])[0]

            return data_lowlight, data_highlight, filename

        elif self.mode == 'val':
            data_lowlight = Image.open(data_lowlight_path).convert('RGB')
            data_highlight = Image.open(data_lowlight_path.replace('low', 'high')).convert('RGB')
            # Validate on center crop

            data_lowlight = TF.to_tensor(data_lowlight)
            data_highlight = TF.to_tensor(data_highlight)

            filename = os.path.splitext(os.path.split(data_lowlight_path)[-1])[0]

            return data_lowlight, data_highlight, filename

        elif self.mode == 'test':
            data_lowlight = Image.open(data_lowlight_path).convert('RGB')
            data_highlight = Image.open(data_lowlight_path.replace('low', 'high')).convert('RGB')

            data_lowlight = TF.to_tensor(data_lowlight)
            data_highlight = TF.to_tensor(data_highlight)

            filename = os.path.splitext(os.path.split(data_lowlight_path)[-1])[0]
            #print(filename)
            return data_lowlight, data_highlight, filename
            
    def __len__(self):
        return len(self.data_list)