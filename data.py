import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import loadDataCL, loadDataClassify

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F

dimension = 256

transform=A.Compose(
[
    A.Resize(dimension, dimension),
    A.HorizontalFlip(p=0.5),
    A.CLAHE(p=0.5),
    A.AdvancedBlur(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.7, contrast_limit=0.7, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

to_tensor=A.Compose([A.Resize(dimension, dimension),
                     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                     ToTensorV2()
                    ])

class DriveDataset(Dataset):
    
    def __init__(self, paths, scale_dataset):

        self.paths = paths
        self.n_samples = len(paths)*scale_dataset
        self.transform = transform
        self.scale = scale_dataset

    def __getitem__(self, idx):
        """ Reading image """

        original = idx % self.scale == 0
        index = idx//self.scale

        # print(self.images_path[index], self.masks_path[index])
        image = cv2.imread(self.paths[index][0], cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.float32(image)
        # image = image/255.0 ## (512, 512, 3)
        # image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        # image = image.astype(np.float32)

        """ Reading mask """
        mask = cv2.imread(self.paths[index][1], cv2.IMREAD_COLOR)
        # mask = mask[:,:,1]
        # mask = np.float32(mask)
        # mask = mask/255.0 ## (512, 512, 3)
        # mask = mask.astype(np.float32)

        if not original:
            transformed = transform(image=image, mask=mask)
        else:
            transformed = to_tensor(image=image, mask=mask)
            
        return transformed['image'].type(torch.FloatTensor), transformed['mask'].type(torch.FloatTensor)

    def __len__(self):
        return self.n_samples

class DallysonDriveDataset(Dataset):

    def __init__(self, paths, scale_dataset):

        self.paths = paths
        self.n_samples = len(paths)*scale_dataset
        self.transform = transform
        self.scale = scale_dataset

    def __getitem__(self, idx):
        """ Reading image """

        original = idx % self.scale == 0
        index = idx//self.scale

        # print(self.images_path[index], self.masks_path[index])
        image = cv2.imread(self.paths[index][0], cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.float32(image)
        # image = image/255.0 ## (512, 512, 3)
        # image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        # image = image.astype(np.float32)

        """ Reading mask """
        disc = cv2.imread(self.paths[index][1], cv2.IMREAD_GRAYSCALE)

        _, disc = cv2.threshold(disc,127,255,cv2.THRESH_BINARY)

        disc = disc.astype(np.float64)

        disc /= 255
        
        cup = cv2.imread(self.paths[index][2], cv2.IMREAD_GRAYSCALE)

        _, cup = cv2.threshold(cup,127,255,cv2.THRESH_BINARY)

        cup = cup.astype(np.float64)

        cup /= 255

        # if disc.shape == cup.shape:
        #     fundo = np.zeros(disc.shape)
        mask = cup + disc
        # mask = np.float32(mask)
        # mask = mask/255.0 ## (512, 512, 3)
        # disc = np.expand_dims(disc, axis=2)
        # cup = np.expand_dims(cup, axis=2)
        # fundo = np.expand_dims(fundo, axis=2)
        # mask = np.concatenate([fundo, disc, cup], axis=2)

        # mask = mask/255

        # print(f"Image shape = {image.shape}\tMask shape = {mask.shape}")

        if not original:
            transformed = transform(image=image, mask=mask)
        else:
            transformed = to_tensor(image=image, mask=mask)
            
        return transformed['image'].type(torch.FloatTensor), transformed['mask'].type(torch.FloatTensor)


    def __len__(self):
        return self.n_samples

class ClassifyDataset(Dataset):

    def __init__(self, paths, scale_dataset):

        self.paths = paths
        self.n_samples = len(paths)*scale_dataset
        self.transform = transform
        self.scale = scale_dataset

    def __getitem__(self, idx):
        """ Reading image """

        original = idx % self.scale == 0
        index = idx//self.scale

        filename = f"E:/SGMD{self.paths[index][0]}"

        image = cv2.imread(filename, cv2.IMREAD_COLOR)

        label = float(self.paths[index][1])

        if not original:
            transformed = transform(image=image)
        else:
            transformed = to_tensor(image=image)
            
        return transformed['image'].type(torch.FloatTensor), torch.tensor(label)

    def __len__(self):
        return self.n_samples

class SegClassifyDataset(Dataset):

    def __init__(self, paths, scale_dataset):

        self.paths = paths
        self.n_samples = len(paths)*scale_dataset
        self.transform = transform
        self.scale = scale_dataset

    def __getitem__(self, idx):
        
        original = idx % self.scale == 0
        index = idx//self.scale

        """ Reading image """
        image = cv2.imread(self.paths[index][0], cv2.IMREAD_COLOR)

        """ Reading mask """
        mask = cv2.imread(self.paths[index][1], cv2.IMREAD_COLOR)

        """ Reading label """
        label = torch.tensor(float(self.paths[index][2]))

        if not original:
            transformed = transform(image=image, mask=mask)
        else:
            transformed = to_tensor(image=image, mask=mask)
            
        return transformed['image'].type(torch.FloatTensor), transformed['mask'].type(torch.FloatTensor), label

    def __len__(self):
        return self.n_samples

if __name__ == '__main__':

    base = 'C:/Projetos/Datasets/thiago.freire/'
    
    refuge_path = f"{base}REFUGE/"
    origa_path = f"{base}ORIGA/"

    origa, refuge = loadDataCL(origa_path, refuge_path)

    train = np.concatenate([origa, refuge], axis=0)

    dataset = SegClassifyDataset(train, 1)

    # base = "E:/SGMD"
        
    # sgmd = loadDataClassify(base)

    # print(len(sgmd))
 
    # dataset = ClassifyDataset(sgmd, 5)

    train_loader = DataLoader(
                dataset=dataset,
                batch_size=10,
                shuffle=True,
                num_workers=2
            )
    
    for x, y, l in train_loader:

        print(f"X com shape = {x.shape}")
        print(f"Y com shape = {y.shape}")
        print(f"L = {l}")
        break

    # for x, y in train_loader:
    #     print(f"X com shape = {x.shape}")
    #     print(f"Y com shape = {y}")


    #     image, mask = x[0], y[0]
    #     image = image.cpu().numpy()
    #     image = image.transpose(1, 2, 0)
    #     print(image.max(), mask)
    #     break