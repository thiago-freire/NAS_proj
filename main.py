import os
# from train_loss import Trainer
from test import Tester
from train import Trainer
from utils import loadData
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import optuna
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def model(blocks, layers, skips, alfa_loss = 0.5,
          epochs = 20, n_folds = 5, albu_scale=3):

    lr = 1e-3
    batch_size = 40

    modelDict = {'blocks':blocks,
                 'layers':layers,
                 'skip':skips}


    base = ""
    base_results = ""
    if os.name == 'posix':
        base = "/backup/thiago.freire/Dataset/"
        base_results = "/backup/thiago.freire/results/cascade/"
    else:
        base = "C:/Projetos/Datasets/thiago.freire/"
        base_results = "E:/results/NAS/"
        
    refuge = f"{base}REFUGE/other"
    origa = f"{base}ORIGA/"
    
    if not os.path.isdir(f"{base_results}fold/"):
        os.mkdir(f"{base_results}fold/")

    origa_paths, refuge_paths = loadData(Refuge_path=refuge, Origa_path=origa)

    if n_folds != 1:

        kFold=KFold(n_splits=n_folds,random_state=42,shuffle=True)

        origa_folds = []
        for train_index, test_index in kFold.split(origa_paths):
            origa_folds.append([train_index, test_index])

        refuge_folds = []
        for train_index, test_index in kFold.split(refuge_paths):
            refuge_folds.append([train_index, test_index])

        for i in range(kFold.get_n_splits()):
            if not os.path.isdir(f"{base_results}fold/fold_{i}"):
                os.mkdir(f"{base_results}fold/fold_{i}")
                
            origa_train, origa_test = origa_paths[origa_folds[i][0]], origa_paths[origa_folds[i][1]]
            refuge_train, refuge_test = refuge_paths[refuge_folds[i][0]], refuge_paths[refuge_folds[i][1]]

            train = np.concatenate([origa_train, refuge_train], axis=0)
            test = np.concatenate([origa_test, refuge_test], axis=0)

            validation_size = len(test)/len(train)
            train, validation = train_test_split(train, test_size=validation_size)
            print(f"Treino: {len(train)}\tValidação: {len(validation)}\tTeste: {len(test)}" )

            trainer = Trainer(train, validation, fold=i, scale=albu_scale, base_results=base_results, modelDict=modelDict)
            trainer.setHyperParam(lr=lr, batch_size=batch_size, epochs=epochs, 
                                  alfa_loss=alfa_loss)
            trainer.run()

            tester = Tester(test, fold=i, base_results=base_results, modelDict=modelDict)
            tester.run()

        # for i in range(kFold.get_n_splits()):
                
        #     origa_train, origa_test = origa_paths[origa_folds[i][0]], origa_paths[origa_folds[i][1]]
        #     refuge_train, refuge_test = refuge_paths[refuge_folds[i][0]], refuge_paths[refuge_folds[i][1]]

        #     train = np.concatenate([origa_train, refuge_train], axis=0)
        #     test = np.concatenate([origa_test, refuge_test], axis=0)

        #     validation_size = len(test)/len(train)
        #     train, validation = train_test_split(train, test_size=validation_size)
        #     print(f"Treino: {len(train)}\tValidação: {len(validation)}\tTeste: {len(test)}" )

        #     old = "/backup/thiago.freire/Dataset/"
        #     new = f"{base_results}disc_dataset/"

        #     new_train = [[line[0].replace(old, new),line[1].replace(old, new)] for line in train]
        #     new_valid = [[line[0].replace(old, new),line[1].replace(old, new)] for line in validation]
        #     new_test = [[line[0].replace(old, new),line[1].replace(old, new)] for line in test]

        #     trainer = Trainer(new_train, new_valid, fold=i, scale=albu_scale, base_results=base_results, k=k, tipo='cup')
        #     trainer.setHyperParam(lr=lr, batch_size=batch_size, epochs=epochs, 
        #                           alfa_loss=alfa_loss)
        #     trainer.run()

        #     tester = Tester(new_test, fold=i, base_results=base_results, k=k, tipo='cup')
        #     tester.run()
        
    else: 

        if not os.path.isdir(f"{base_results}fold/fold_7"):
                os.mkdir(f"{base_results}fold/fold_7")
        train = np.concatenate([origa_paths, refuge_paths], axis=0)
        train, validation = train_test_split(train, test_size=0.15)
        train, test = train_test_split(train, test_size=0.15)

        print(f"Treino: {len(train)}\tValidação: {len(validation)}\tTeste: {len(test)}" )

        trainer = Trainer(train, validation, 7, scale=albu_scale, base_results=base_results, modelDict=modelDict)
        trainer.setHyperParam(lr=lr, batch_size=batch_size, 
                              epochs=epochs, alfa_loss=alfa_loss, alfa_class=0.5)
        trainer.run()

        tester = Tester(test, 7, base_results, modelDict)
        return tester.run()

def objective(trial: optuna.Trial) -> float:

    epochs = 20  
    n_folds = 1

    albu_scale = trial.suggest_int("albumentations", 1, 4)

    alfa_loss = trial.suggest_float("alfa_loss", 0.4, 0.7)

    blocks = []
    for i in range(8):
        cate = trial.suggest_categorical(f"step_{i+1}", ["AT", "NT"])
        blocks.append(cate)
    
    layers = []
    for i in range(4):
        layer = trial.suggest_int(f"layer_{i+1}", 2, 5)
        layers.append(layer)
    
    skips = []
    for i in range(4):
        skip = trial.suggest_categorical(f"skip_{i+1}", [True, False])
        skips.append(skip)

    dice_disc, dice_cup = model(blocks, layers, skips, alfa_loss, 
                                n_folds = 1, albu_scale = albu_scale)
    
    return dice_cup, dice_disc

def runOptuna():

    create = True

    if create:
        study = optuna.create_study(
            # storage="postgresql://postgres:postgres@192.168.200.169/optuna_thiago",  # Specify the storage URL here.
            storage="postgresql://postgres:123456@localhost/optuna",  # Specify the storage URL here.
            study_name="Optimização do Modelo",
            directions=["maximize", "maximize"])
    else:
        study = optuna.load_study(
            storage="postgresql://postgres:123456@localhost/optuna",  # Specify the storage URL here.
            study_name="Optimização do Modelo")
    
    study.optimize(objective, n_trials=12)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    runOptuna()
    # model(k = 2, batch_size = 20, epochs = 50,  
    #       n_folds = 5, albu_scale = 2, 
    #       alfa_loss = 0.61)
            
    # iou_disc, iou_cup, dice_disc, dice_cup = model(n_folds=1)
    