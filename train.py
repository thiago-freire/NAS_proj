import time
import os

import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from data import DriveDataset
from model import ResUNetAtt
from loss import DiceBCELossSigmoid
from utils import seeding, epoch_time
import matplotlib.pyplot as plt
# from tqdm import tqdm

class Trainer():

    def __init__(self, train, validation, fold, scale, base_results, modelDict):

        self.num_classes = 3
        self.fold = fold
        self.base_results = base_results
        self.model = ResUNetAtt(blocks=modelDict['blocks'], 
                                layers=modelDict['layers'], 
                                skips=modelDict['skip'])
        
        """ Seeding """
        seeding(42)

        data_str = f"Dataset Size:\nTrain: {len(train)} - Valid: {len(validation)}\n"
        print(data_str)

        print("Select device cuda")
        self.device = torch.device('cuda')   ## RTX 4060 Ti 16GB

        self.model = self.model.to(self.device)

        if not os.path.isdir(f"{self.base_results}fold/fold_{self.fold}/pretrained_ckpt/"):
            os.mkdir(f"{self.base_results}fold/fold_{self.fold}/pretrained_ckpt/")

        self.checkpoint_path = f"{self.base_results}fold/fold_{self.fold}/pretrained_ckpt/res_unet.pth"

        """ Dataset and loader """
        self.train_dataset = DriveDataset(train, scale)
        self.valid_dataset = DriveDataset(validation, scale)

    def setHyperParam(self, lr, batch_size, epochs, alfa_loss, alfa_class):

        """ Hyperparameters """
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.alfa_loss = alfa_loss
        self.alfa_class = alfa_class

        summary(self.model, (self.batch_size, 3, 256, 256), device=self.device, depth=2)

    def train(self, loader: DataLoader, optimizer: torch.optim.Adam, 
              loss_fn: DiceBCELossSigmoid, device: torch.device):
        
        epoch_loss = 0.0

        self.model.train()
        print(f"Treinando com {len(loader)} mini-batchs")

        for x, y in loader: #tqdm(loader):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_pred = self.model(x)

            loss = loss_fn(y_pred, y[:,:,:,1])
            # disc_loss = loss_fn(y_pred[:,0,:,:], y[:,:,:,1])
            # cup_loss = loss_fn(y_pred[:,1,:,:], y[:,:,:,2])

            # loss = self.alfa_class * disc_loss + (1-self.alfa_class) * cup_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
            
        return epoch_loss

    def evaluate(self, loader: DataLoader, loss_fn: DiceBCELossSigmoid, device: torch.device):
        epoch_loss = 0.0

        self.model.eval()

        with torch.no_grad():
            for x, y in loader:

                x = x.to(device)
                y = y.to(device)

                y_pred = self.model(x)
                loss = loss_fn(y_pred, y[:,:,:,1])
                # disc_loss = loss_fn(y_pred[:,0,:,:], y[:,:,:,1])
                # cup_loss = loss_fn(y_pred[:,1,:,:], y[:,:,:,2])

                # loss = self.alfa_class * disc_loss + (1-self.alfa_class) * cup_loss

                epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
                
        return epoch_loss

    def run(self):

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12
        )

        valid_loader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12
        )


        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = DiceBCELossSigmoid(self.alfa_loss)

        """ Training the model """
        best_valid_loss = float("inf")

        loss_values = []
        valid_loss_values = []
        epocas = range(self.num_epochs)

        print("Iniciando Treinamento...")
        for epoch in epocas:
            start_time = time.time()
            print(f"Iniciando época {epoch+1} de {self.num_epochs}...")
            train_loss = self.train(train_loader, optimizer, loss_fn, self.device)
            valid_loss = self.evaluate(valid_loader, loss_fn, self.device)

            loss_values.append(train_loss)
            valid_loss_values.append(valid_loss)

            """ Saving the model """
            if valid_loss < best_valid_loss:
                data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {self.checkpoint_path}"
                print(data_str)

                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.checkpoint_path)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
            data_str += f'\tTrain Loss: {train_loss:2.4f}\n'
            data_str += f'\t Val. Loss: {valid_loss:2.4f}\n'
            print(data_str)

        figura = plt.figure()
        figura.add_subplot(111)
        plt.plot(epocas, loss_values, color='r', label='train loss')
        plt.plot(epocas, valid_loss_values, color='b', label='test loss')
        plt.title('Loss Comparison')
        plt.ylabel('Loss')
        plt.xlabel('Epocas')
        plt.legend()

        if not os.path.isdir(f"{self.base_results}fold/fold_{self.fold}/lossGraf/"):
            os.mkdir(f"{self.base_results}fold/fold_{self.fold}/lossGraf/")

        plt.savefig(f'{self.base_results}fold/fold_{self.fold}/lossGraf/grafico.png', dpi=200)