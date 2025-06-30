import os, time
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

from data import DriveDataset
from model_double_decoder import ResUNetAtt


dimension = 256

transform=A.Compose([A.Resize(dimension, dimension),
                     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                     ToTensorV2()])

from utils import loadData, seeding

class Tester():

    def __init__(self, test, fold, base_results, modelDict, checkpoint_path=None):
        self.dataset = DriveDataset(test, 1)
        self.fold = fold
        self.base_results = base_results
        self.modelDict = modelDict

        """ Hyperparameters """
        if checkpoint_path == None:
            self.checkpoint_path = f"{self.base_results}fold/fold_{self.fold}/pretrained_ckpt/res_unet.pth"
        else:
            self.checkpoint_path = checkpoint_path

        if not os.path.isdir(f"{self.base_results}fold/fold_{self.fold}/output"):
            os.mkdir(f"{self.base_results}fold/fold_{self.fold}/output")

        if not os.path.isdir(f"{self.base_results}fold/fold_{self.fold}/results/"):
            os.mkdir(f"{self.base_results}fold/fold_{self.fold}/results/")

        if not os.path.isdir(f"{self.base_results}fold/fold_{self.fold}/grad_cam/"):
            os.mkdir(f"{self.base_results}fold/fold_{self.fold}/grad_cam/")

        if not os.path.isdir(f"{self.base_results}fold/fold_{self.fold}/results/"):
            os.mkdir(f"{self.base_results}fold/fold_{self.fold}/results/")

        if not os.path.isdir(f"{self.base_results}fold/fold_{self.fold}/grad_cam/"):
            os.mkdir(f"{self.base_results}fold/fold_{self.fold}/grad_cam/")

    def calculate_metrics_3(self, y_true: torch.Tensor, 
                          y_pred_disc: torch.Tensor, 
                          y_pred_cup: torch.Tensor):

        y_pred_cup = y_pred_cup.cpu().numpy()
        y_pred_disc = y_pred_disc.cpu().numpy()
        y_true = y_true.cpu().numpy()

        y_pred_cup = (y_pred_cup > 0.5) * 1
        y_pred_disc = (y_pred_disc > 0.5) * 1

        y_pred_cup = y_pred_cup.astype(dtype=np.uint8)
        y_pred_disc = y_pred_disc.astype(dtype=np.uint8)
        y_true = y_true.astype(dtype=np.uint8)

        y_true_disc = y_true[:,:,1]
        y_true_cup = y_true[:,:,2]

        y_pred_cup = y_pred_cup.flatten()
        y_true_cup = y_true_cup.flatten()
        y_pred_disc = y_pred_disc.flatten()
        y_true_disc = y_true_disc.flatten()


        score_jaccard = {
            'disc':jaccard_score(y_true=y_true_disc, y_pred=y_pred_disc), 
            'cup':jaccard_score(y_true=y_true_cup, y_pred=y_pred_cup)
            }
        score_f1 = {
            'disc':f1_score(y_true=y_true_disc, y_pred=y_pred_disc), 
            'cup':f1_score(y_true=y_true_cup, y_pred=y_pred_cup)
            }
        score_recall = {
            'disc':recall_score(y_true=y_true_disc, y_pred=y_pred_disc), 
            'cup':recall_score(y_true=y_true_cup, y_pred=y_pred_cup)
            }
        score_precision = {
            'disc':precision_score(y_true=y_true_disc, y_pred=y_pred_disc), 
            'cup':precision_score(y_true=y_true_cup, y_pred=y_pred_cup)
            }
        score_acc = {
            'disc':accuracy_score(y_true=y_true_disc, y_pred=y_pred_disc), 
            'cup':accuracy_score(y_true=y_true_cup, y_pred=y_pred_cup)
            }

        return {'jaccard': score_jaccard, 'f1': score_f1, 'recall': score_recall, 'precision': score_precision, 'accuracy': score_acc}

    def calculate_metrics_2(self, y_true: torch.Tensor, 
                          y_pred: torch.Tensor):

        y_pred= y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()

        y_pred = (y_pred > 0.5) * 1

        y_pred = y_pred.astype(dtype=np.uint8)
        y_true = y_true.astype(dtype=np.uint8)

        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        score_jaccard = jaccard_score(y_true=y_true, y_pred=y_pred)
        score_f1 = f1_score(y_true=y_true, y_pred=y_pred)
        score_recall = recall_score(y_true=y_true, y_pred=y_pred)
        score_precision =precision_score(y_true=y_true, y_pred=y_pred)
        score_acc = accuracy_score(y_true=y_true, y_pred=y_pred)

        return {'jaccard': score_jaccard, 'f1': score_f1, 'recall': score_recall, 'precision': score_precision, 'accuracy': score_acc}

    def soma(self, all_score, score):

        for m_key, metric in score.items():
            for tag in metric.keys():
                all_score[m_key][tag] += score[m_key][tag]
    
    def mask_parse(self, mask: torch.Tensor):

        mask = mask.cpu().numpy()
        mask = mask * 254
        mask = np.expand_dims(a=mask, axis=2)

        return np.concatenate([mask,mask,mask], axis=2)

    def binary_converter(self, y_pred: torch.Tensor):

        z = y_pred.cpu().numpy()

        for i in range(z.shape[0]):
            idx = np.argmax(z[i], axis=-1)
            y = np.zeros( z[i].shape )
            y[ np.arange(y.shape[0]), idx] = 1

            z[i] = y
        
        y_pred = torch.from_numpy(z)
            
        return y_pred

    def run(self):
        """ Seeding """
        seeding(42)

        """ Load the checkpoint """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = ResUNetAtt(blocks=self.modelDict['blocks'],
                           layers=self.modelDict['layers'],
                           skips=self.modelDict['skip'])
        model = model.to(device)
        model.load_state_dict(torch.load(self.checkpoint_path, map_location=device, weights_only=False))
        model.eval()

        metrics_score = {'jaccard': {'disc': 0.0, 'cup': 0.0}, 
                         'f1': {'disc': 0.0, 'cup': 0.0}, 
                         'recall': {'disc': 0.0, 'cup': 0.0}, 
                         'precision': {'disc': 0.0, 'cup': 0.0}, 
                         'accuracy': {'disc': 0.0, 'cup': 0.0}}
        
        time_taken = []

        with open(f"{self.base_results}fold/fold_{self.fold}/output/out_metrics_double_back_decoder.csv", "w") as file:
            file.write("arquivo;jaccard-disc;jaccard-cup;")
            file.write("f1-disc;f1-cup;recall-disc;recall-cup;")
            file.write("precision-disc;precision-cup;")
            file.write("accuracy-disc;accuracy-cup;\n")
        #     file.write("arquivo;jaccard-disc;")
        #     file.write("f1-disc;recall-disc;")
        #     file.write("precision-disc;")
        #     file.write("accuracy-disc;\n")
        loader = DataLoader(
            dataset = self.dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = 1
        )
        
        cont = 1

        for image, mask in tqdm(loader, desc="Processando double_back_decoder"):

            image = image.to(device=device)
            mask = mask.to(device=device).squeeze()
            
            with torch.no_grad():
                """ Prediction and Calculating FPS """
                start_time = time.time()
                y_pred_disc, y_pred_cup = model(image)

                # print(y_pred.shape)

                y_disc_pred = y_pred_disc.squeeze()
                y_cup_pred = y_pred_cup.squeeze()

                y_disc_pred = torch.sigmoid(y_disc_pred)
                y_cup_pred = torch.sigmoid(y_cup_pred)

                total_time = time.time() - start_time
                time_taken.append(total_time)

                score = self.calculate_metrics_3(mask, y_disc_pred, y_cup_pred)
                self.soma(metrics_score, score)
            
            with open(f"{self.base_results}fold/fold_{self.fold}/output/out_metrics_double_back_decoder.csv", "a") as file:
                file.write(f"{cont:03}.png;")
                file.write(f"{score['jaccard']['disc']:1.4f};{score['jaccard']['cup']:1.4f};"+
                        f"{score['f1']['disc']:1.4f};{score['f1']['cup']:1.4f};"+
                        f"{score['recall']['disc']:1.4f};{score['recall']['cup']:1.4f};"+
                        f"{score['precision']['disc']:1.4f};{score['precision']['cup']:1.4f};"+
                        f"{score['accuracy']['disc']:1.4f};{score['accuracy']['cup']:1.4f}\n")
            #     file.write(f"{score['jaccard']['disc']:1.4f};"+
            #             f"{score['f1']['disc']:1.4f};"+
            #             f"{score['recall']['disc']:1.4f};"+
            #             f"{score['precision']['disc']:1.4f};"+
            #             f"{score['accuracy']['disc']:1.4f}\n")

            """ Saving masks """
            ori_mask_disc = self.mask_parse(mask[:,:,1])
            ori_mask_cup = self.mask_parse(mask[:,:,2])
            y_disc_pred = self.mask_parse(y_disc_pred)
            y_cup_pred = self.mask_parse(y_cup_pred)
            line = np.ones((dimension, 5, 3)) * 128

            image = image.squeeze().cpu().numpy()
            image = np.transpose(image, (1, 2, 0))

            cat_images_disc = np.concatenate(
                [image, line, ori_mask_disc, line, y_disc_pred], axis=1
            )

            cat_images_cup = np.concatenate(
                [image, line, ori_mask_cup, line, y_cup_pred], axis=1
            )
            
            cv2.imwrite(f"{self.base_results}fold/fold_{self.fold}/results/{cont:03}_disc.png", cat_images_disc)
            cv2.imwrite(f"{self.base_results}fold/fold_{self.fold}/results/{cont:03}_cup.png", cat_images_cup)
            
            cont += 1

        jaccard = {'disc':metrics_score['jaccard']['disc']/len(loader), 
                   'cup':metrics_score['jaccard']['cup']/len(loader)
                   }
        f1 = {'disc':metrics_score['f1']['disc']/len(loader), 
              'cup':metrics_score['f1']['cup']/len(loader)
            }
        recall = {'disc':metrics_score['recall']['disc']/len(loader), 
                  'cup':metrics_score['recall']['cup']/len(loader)
                }
        precision = {'disc':metrics_score['precision']['disc']/len(loader), 
                     'cup':metrics_score['precision']['cup']/len(loader)
                    }
        accuracy = {'disc':metrics_score['accuracy']['disc']/len(loader), 
                    'cup':metrics_score['accuracy']['cup']/len(loader)
                    }
        
        with open(f"{self.base_results}fold/fold_{self.fold}/output/out.txt", "w") as file:
            file.write(f"Jaccard:\n\tDisc - {jaccard['disc']:1.4f}\n\tCup - {jaccard['cup']:1.4f}\n"+
                        f"F1:\n\tDisc - {f1['disc']:1.4f}\n\tCup - {f1['cup']:1.4f}\n"+
                        f"Recall:\n\tDisc - {recall['disc']:1.4f}\n\tCup - {recall['cup']:1.4f}\n"+
                        f"Precision:\n\tDisc - {precision['disc']:1.4f}\n\tCup - {precision['cup']:1.4f}\n"+
                        f"Acc:\n\tDisc - {accuracy['disc']:1.4f}\n\tCup - {accuracy['cup']:1.4f}\n")
        #     file.write(f"Jaccard:\n\tDisc - {jaccard['disc']:1.4f}\n"+
        #                 f"F1:\n\tDisc - {f1['disc']:1.4f}\n"+
        #                 f"Recall:\n\tDisc - {recall['disc']:1.4f}\n"+
        #                 f"Precision:\n\tDisc - {precision['disc']:1.4f}\n"+
        #                 f"Acc:\n\tDisc - {accuracy['disc']:1.4f}\n")
            
            fps = 1/np.mean(time_taken)
            file.write(f"FPS: {fps}")
        
        return f1['disc'], f1['cup']

if __name__ == "__main__":

    base = "C:/Projetos/Datasets/thiago.freire"
    origa = f"{base}/ORIGA/"
    refuge = f"{base}/REFUGE/other"
    origa_paths, refuge_paths = loadData(Origa_path=origa, Refuge_path=refuge)

    train, test = train_test_split(origa_paths, test_size=0.2)

    batch_size = 35
    epochs = 1
    lr = 1e-4
    base_results = "C:\\Projetos\\Timm_Testes\\"

    tester = Tester(test, 0, base_results)#, {'dice':0, 'boundary':1})
    tester.run()