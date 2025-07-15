import json
import os
import random
import numpy as np
import pandas as pd
import torch
from glob import glob

def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

""" Calculate the time taken """
def fold_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_hor = int(elapsed_mins / 60)
    elapsed_mins = int(elapsed_mins - (elapsed_hor * 60))
    elapsed_secs = int(elapsed_time - ((elapsed_mins * 60)+(elapsed_hor * 360)))
    return elapsed_hor, elapsed_mins, elapsed_secs

""" Load Data """
def loadData(Origa_path, Refuge_path):

    origa_images = sorted(glob(os.path.join(Origa_path, 'Images/*.jpg')))
    origa_masks = sorted(glob(os.path.join(Origa_path, 'Masks_Proc/*.png')))
    origa_paths = list(zip(origa_images, origa_masks))

    train_x = sorted(glob(os.path.join(Refuge_path, 'train/Images/*.jpg')))
    valid_x = sorted(glob(os.path.join(Refuge_path, 'val/Images/*.jpg')))
    test_x = sorted(glob(os.path.join(Refuge_path, 'test/Images/*.jpg')))

    train_y = sorted(glob(os.path.join(Refuge_path, 'train/Masks_Proc/*.png')))
    valid_y = sorted(glob(os.path.join(Refuge_path, 'val/Masks_Proc/*.png')))
    test_y = sorted(glob(os.path.join(Refuge_path, 'test/Masks_Proc/*.png')))

    refuge_paths = list(zip(train_x, train_y))+list(zip(valid_x, valid_y))+list(zip(test_x, test_y))

    return np.array(origa_paths), np.array(refuge_paths)

def loadDataCL(Origa_path, Refuge_path):

    origa = pd.read_csv(f"{Origa_path}OrigaList.csv")

    origa['imagem'] = origa['Filename'].apply(lambda x: Origa_path + 'Images/' + x)

    origa['mask'] = origa['Filename'].apply(lambda x: Origa_path + 'Masks_Proc/' + x).str.replace('.jpg', '.png')

    origa_labels = list(zip(origa["imagem"],  origa['mask'], origa["Glaucoma"]))

    Refuge_train = Refuge_path + "other/train/"

    Refuge_val = Refuge_path + "other/val/"

    refuge_labels = []

    with open(Refuge_train+'index.json', 'r') as f:
        refuge = json.load(f)

    for i in refuge:
        refuge_labels.append((Refuge_train+'Images/'+refuge[i]['ImgName'], Refuge_train+'Masks_Proc/'+refuge[i]['ImgName'].replace('.jpg','.png'), refuge[i]['Label']))

    with open(Refuge_val+'index.json', 'r') as f:
        refuge = json.load(f)

    for i in refuge:
        refuge_labels.append((Refuge_val+'Images/'+refuge[i]['ImgName'], Refuge_val+'Masks_Proc/'+refuge[i]['ImgName'].replace('.jpg','.png'), refuge[i]['Label']))

    return np.array(origa_labels), np.array(refuge_labels)

def loadDataClassify(SGMD_path: str):

    sgmd = pd.read_csv(f"{SGMD_path}/metadata - standardized.csv")

    labels = list(zip(sgmd["fundus"], sgmd["types"]))

    return labels

def loadDallysonData(base_path, all):

    if all :
        glaucoma_images_left = sorted(glob(os.path.join(base_path, 'Dallyson/glaucoma/left_eye/*.png')))
        glaucoma_mask_disc_left = sorted(glob(os.path.join(base_path, 'Dallyson/glaucoma/left_eye/Mask_disc/*.png')))
        glaucoma_mask_cup_left = sorted(glob(os.path.join(base_path, 'Dallyson/glaucoma/left_eye/Mask_cup/*.png')))
        glaucoma_left = list(zip(glaucoma_images_left, glaucoma_mask_disc_left, glaucoma_mask_cup_left))

        glaucoma_images_right= sorted(glob(os.path.join(base_path, 'Dallyson/glaucoma/right_eye/*.png')))
        glaucoma_mask_disc_right= sorted(glob(os.path.join(base_path, 'Dallyson/glaucoma/right_eye/Mask_disc/*.png')))
        glaucoma_mask_cup_right= sorted(glob(os.path.join(base_path, 'Dallyson/glaucoma/right_eye/Mask_cup/*.png')))
        glaucoma_right = list(zip(glaucoma_images_right, glaucoma_mask_disc_right, glaucoma_mask_cup_right))

        normal_images_left = sorted(glob(os.path.join(base_path, 'Dallyson/normal/left_eye/*.png')))
        normal_mask_disc_left = sorted(glob(os.path.join(base_path, 'Dallyson/normal/left_eye/Mask_disc/*.png')))
        normal_mask_cup_left = sorted(glob(os.path.join(base_path, 'Dallyson/normal/left_eye/Mask_cup/*.png')))
        normal_left = list(zip(normal_images_left, normal_mask_disc_left, normal_mask_cup_left))

        normal_images_right= sorted(glob(os.path.join(base_path, 'Dallyson/normal/right_eye/*.png')))
        normal_mask_disc_right= sorted(glob(os.path.join(base_path, 'Dallyson/normal/right_eye/Mask_disc/*.png')))
        normal_mask_cup_right= sorted(glob(os.path.join(base_path, 'Dallyson/normal/right_eye/Mask_cup/*.png')))
        normal_right = list(zip(normal_images_right, normal_mask_disc_right, normal_mask_cup_right))

        print(len(glaucoma_left), len(glaucoma_right), len(normal_left), len(normal_right))

        return_paths = glaucoma_left + glaucoma_right + normal_left + normal_right

    else:
        cont = 1
        return_paths = []
        vet = []
        with open("cup.txt", "r") as file:
            contents = file.read()
            lista = contents.split('\n')
            for i in range(0, 4883, 3):
                return_paths.append([lista[i], lista[i+1], lista[i+2]])
    
        print(f"Tamanho: {len(return_paths)}\n1º: {return_paths[0]}")

    return np.array(return_paths)

#def printMat(mat):

 # df_cm = pd.DataFrame(data=mat)
 # sn.heatmap(df_cm, annot=True)
 # plt.show()

if __name__ == '__main__':

    base = 'C:/Projetos/Datasets/thiago.freire/'
    
    refuge_path = f"{base}REFUGE/"
    origa_path = f"{base}ORIGA/"

    origa, refuge = loadDataCL(origa_path, refuge_path)

    print(origa[0], refuge[0])

    # sgmd = loadDataClassify('E:\\SGMD')

    # print(sgmd[0][0], sgmd[0][1])
