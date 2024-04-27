from collections import OrderedDict, namedtuple
from itertools import product
from torch.utils.tensorboard import SummaryWriter
import torchvision
import pandas as pd
import time
import torch
from pathlib import Path
from EarlyStopping import EarlyStopping

class RunBuilder:
    def __init__(self, params) -> None:
        self.runs = self.get_runs(params)
    def __len__(self):
        return len(self.runs)

    def get_runs(self, params):
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            print(f'value {v} and {Run(*v)}')
            runs.append(Run(*v))
        return runs


params = OrderedDict(
    lr=[.01],
    batch_size=[1000,2000],
    num_workers=[0,1],
    device=['cuda', 'cpu']
)

#TODO: Add the number of epoch ran until got early stopped and the run number into the summary writer

class RunManager():
    def writeError(self, msg=''):
        with open(self.errorPath, 'a') as f:
            f.write(f'Error at runs: {self.run_count}\nParameters: {self.run_params}\nAdditional msg: {msg}')

    def writeToCSV(self):
        oldStatsDF = None
        try:
            with open(self.statsFileCSV, 'r') as f:
                oldStatsDF = pd.read_csv(f)
                oldStatsDF = pd.concat(oldStatsDF, pd.DataFrame.from_records(self.run_data[-1]))
        except:
            oldStatsDF = pd.DataFrame.from_dict(self.run_data)

        with open(self.statsFileCSV, 'w') as f:
            oldStatsDF.to_csv(f)

    def __init__(self, statsFolderPath, statsFileName, earlyStop=True):
        self.epoch_count = 0
        self.epoch_train_loss = 0
        self.epoch_valid_loss = 0
        self.epoch_numTrain_correct = 0
        self.epoch_numValid_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.model = None
        self.train_loader = None
        self.valid_loader = None
        self.tb = None
        
        self.useEarlyStop = earlyStop
        self.earlyStop = None
        self.stop = False

        self.statsFolderPath = Path(statsFolderPath)
        Path.mkdir(self.statsFolderPath, exist_ok=True, parents=True)
        self.statsFileCSV = Path(self.statsFolderPath / f'{statsFileName}.csv')
        self.errorPath = Path(statsFolderPath, 'error.txt')
        open(self.errorPath, 'w')   # restart the error file

    def begin_run(self, run, model, trainLoader, validLoader):
        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.model = model
        self.train_loader = trainLoader
        self.valid_loader = validLoader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.train_loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)   # Add images and graph when begin one run
        # self.tb.add_graph(
        #     self.model,
        #     images.to(getattr(run, 'device', 'cpu')))   # Default to cpu if the key `device` in dictionary 'run' does not exist
        self.earlyStop = EarlyStopping(patience=15)
        self.stop = False


    def end_run(self, savePath='', save=False):
        self.tb.close()
        self.epoch_count = 0
        self.writeToCSV()
        if save:
            self.saveModel(savePath, result=self.run_data[-1])

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_train_loss = 0
        self.epoch_numTrain_correct = 0
        self.epoch_valid_loss = 0
        self.epoch_numValid_correct = 0

    def end_epoch(self, checkptFolderPath):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        train_loss = self.epoch_train_loss / len(self.train_loader.dataset)
        train_accuracy = self.epoch_numTrain_correct / len(self.train_loader.dataset)

        valid_loss = self.epoch_valid_loss / len(self.valid_loader.dataset)
        valid_accuracy = self.epoch_numValid_correct / len(self.valid_loader.dataset)

        self.tb.add_scalars('Loss', {'trainLoss': train_loss, 'validLoss': valid_loss}, self.epoch_count)
        self.tb.add_scalars('Accuracy', {'trainAcc': train_accuracy, 'validAcc': valid_accuracy}, self.epoch_count)  # Add scalar is use when at the end of epoch

        for name, param in self.model.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            if param.grad != None:
                self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results['run'] = self.run_count
        results['modelName'] = self.model.name
        results['epoch'] = self.epoch_count
        results['train loss'] = train_loss
        results['valid loss'] = valid_loss
        results['train accuracy'] = train_accuracy
        results['valid accuracy'] = valid_accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items(): 
            results[k] = v  # Add the hyperparameter to words to easy report
        self.run_data.append(results)
        # df = pd.DataFrame.from_dict(self.run_data, orient='columns')    # Technically you can put this outside the function and call this after the final run
                    # but we will just gona re-create the dataframe everytime cause we lazy
        if self.useEarlyStop:
            self.checkEarlyStop(valid_loss, self.model, checkptFolderPath, self.run_count)
            if self.earlyStop.early_stop:
                self.stop = True

    def track_train_loss(self, loss):
        self.epoch_train_loss += loss.item() * self.train_loader.batch_size
    
    def track_valid_loss(self, loss):
        self.epoch_valid_loss += loss.item() * self.valid_loader.batch_size

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def track_numTrain_correct(self,  preds, labels):
        self.epoch_numTrain_correct += self._get_num_correct(preds, labels)
    def track_numValid_correct(self,  preds, labels):
        self.epoch_numValid_correct += self._get_num_correct(preds, labels)

    def checkEarlyStop(self, valLoss, model, folderPath, run):
        fPath = Path(folderPath)
        Path.mkdir(fPath, exist_ok=True, parents=True)
        filePath = Path(fPath / f'earlyStop_run_{run}.pt')
        
        self.earlyStop(valLoss, model, filePath)


    def saveModel(self, pathName, result):
        moduleName = ''
        for k, v in result.items():
            moduleName += f'_{k}:{v}'
        folderPath = Path(pathName)
        Path.mkdir(folderPath, exist_ok=True, parents=True)
        filePath = Path(folderPath / f'{moduleName}.pt')
        
        torch.save(self.model.state_dict(), filePath)