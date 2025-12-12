import os
import time
# from train_loss import Trainer
from test_double_decoder import Tester
from train_double_decoder import Trainer
from utils import fold_time, loadData, openHoldout
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import optuna
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def model(blocks, layers, skips, alfa_loss = 0.5, alfa_class = 0.5,
          epochs = 35, n_folds = 5, albu_scale=1):

    lr = 1e-3
    batch_size = 40

    modelDict = {'blocks':blocks,
                 'layers':layers,
                 'skip':skips}


    base = ""
    base_results = ""
    if os.name == 'posix':
        base = "/backup/thiago.freire/Dataset/"
        base_results = "/backup/thiago.freire/results/NAS/"
    else:
        base = "C:\\Projetos\\Datasets\\thiago.freire\\"
        base_results = "E:\\results\\NAS\\"
        
    refuge = f"{base}REFUGE/other"
    origa = f"{base}ORIGA/"
    
    if not os.path.isdir(f"{base_results}fold/"):
        os.mkdir(f"{base_results}fold/")

    if n_folds != 1:

        dices = []
        for i in range(5):

            start_time = time.time()

            train = openHoldout(f"data_set/train_{i}.txt")
            validation = openHoldout(f"data_set/val_{i}.txt")
            test = openHoldout(f"data_set/test_{i}.txt")

            if not os.path.isdir(f"{base_results}fold/fold_{i}/"):
                os.mkdir(f"{base_results}fold/fold_{i}/")

            trainer = Trainer(train, validation, fold=i, scale=albu_scale, base_results=base_results, modelDict=modelDict)
            trainer.setHyperParam(lr=lr, batch_size=batch_size, epochs=epochs, 
                                  alfa_loss=alfa_loss, alfa_class=alfa_class)
            trainer.run()

            tester = Tester(test, fold=i, base_results=base_results, modelDict=modelDict)
            dices.append(tester.run())

            end_time = time.time()
            fold_hor, fold_mins, fold_secs = fold_time(start_time, end_time)
            data_str = f'**************************************\nFold {i} Time: {fold_hor}h {fold_mins}m {fold_secs}s\n**************************************'
            print(data_str)
        
        return np.array(dices)
    
    else: 

        origa_paths, refuge_paths = loadData(Refuge_path=refuge, Origa_path=origa)

        if not os.path.isdir(f"{base_results}fold/fold_7"):
                os.mkdir(f"{base_results}fold/fold_7")
        train = np.concatenate([origa_paths, refuge_paths], axis=0)
        train, validation = train_test_split(train, test_size=0.15)
        train, test = train_test_split(train, test_size=0.15)

        print(f"Treino: {len(train)}\tValidação: {len(validation)}\tTeste: {len(test)}" )

        trainer = Trainer(train, validation, 7, scale=albu_scale, base_results=base_results, modelDict=modelDict)
        trainer.setHyperParam(lr=lr, batch_size=batch_size, 
                              epochs=epochs, alfa_loss=alfa_loss, alfa_class=alfa_class)
        trainer.run()

        tester = Tester(test, 7, base_results, modelDict)
        return tester.run()

def objective(trial: optuna.Trial) -> float:

    albu_scale = 4 #trial.suggest_int("albumentations", 1, 4)

    alfa_loss = 0.549962875 #trial.suggest_float("alfa_loss", 0.3, 0.7)

    alfa_class = 0.474814334 #trial.suggest_float("alfa_class", 0.3, 0.7)

    # blocks = ["AT", "AT", "C", "NT", "NT", "C", "AT", "AT"] # Obtido pelo Optuna.
    blocks = []
    for i in range(4):
        cate = trial.suggest_categorical(f"step_{i+1}", ["AT", "NT", "C"])
        blocks.append(cate)
    blocks = np.concatenate((blocks, blocks[::-1]))

    # 0.9538957916035747, 0.8777849589995625
    # layer_1 3
    # layer_2 5
    # layer_3 3
    # layer_4 4
    layers = [3,4,3,5]
    # for i in range(4):
    #     layer = trial.suggest_int(f"layer_{i+1}", 2, 5)
    #     layers.append(layer)
    
    skips = [True, True, True, False]
    # for i in range(4):
    #     skip = trial.suggest_categorical(f"skip_{i+1}", [True, False])
    #     skips.append(skip)

    dices = model(blocks, layers, skips, alfa_loss, alfa_class, epochs=35,
                  n_folds = 5, albu_scale = albu_scale)
    
    dice_oc = dices[:,0]

    dice_od = dices[:,1]

    oc_mean = np.mean(dice_oc)
    oc_std = np.std(dice_oc)

    od_mean = np.mean(dice_od)
    od_std = np.std(dice_od)

    # with open(f"/backup/thiago.freire/results/NAS/trial_{trial.number}.txt","w") as file:
    #     file.write(f"Dice OC Mean = {oc_mean}\tDice OC Std = {oc_std}\nDice OD Mean = {od_mean}\tDice OD Std = {od_std}")

    # dice_cup = oc_mean - oc_std

    # dice_disc = od_mean - od_std
    
    return oc_mean, oc_std, od_mean, od_std

def runOptuna():

    study = optuna.create_study(
        storage="postgresql://postgres:postgres@192.168.200.169/optuna",  # Specify the storage URL here.
        load_if_exists=True,
        study_name="Optimização do Modelo",
        directions=["maximize", "minimize", "maximize", "minimize"]
    )

    study.optimize(objective, n_trials=10)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

    runOptuna()
    # model(k = 2, batch_size = 20, epochs = 50,  
    #       n_folds = 5, albu_scale = 2, 
    #       alfa_loss = 0.61)


    # blocks = ["AT", "AT", "C", "NT", "NT", "C", "AT", "AT"]
    # layers = [3,4,3,5]
    # skips = [True, True, True, False]
            
    # model(blocks, layers, skips, n_folds=5, albu_scale=4, alfa_loss=0.549962875, alfa_class=0.474814334, epochs=100)
    