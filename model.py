import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision
from torch.utils.data import random_split
from collections import OrderedDict
from runBuilder import RunBuilder, RunManager
from tqdm import trange
import time

DATASET_PATH = './Datasets'
SAVE_MODEL_PATH = './savedModels'
STATISTIC_PATH = './savedStatistics'
TIMESTAMP = time.ctime()

# Load the data
imgTrans = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

trainValSet = ImageFolder("./Datasets/TrainVal/", transform=imgTrans)
trainValSplitPercent = [0.7, 0.3]
trainSet, valSet = random_split(trainValSet, [int(p * len(trainValSet)) for p in trainValSplitPercent])


def makeModel():
    # model = torchvision.models.resnet50(weights='DEFAULT')
    model = torchvision.models.resnet18(weights='DEFAULT')
    model.name = 'ResNet18'
    # Froze other layers
    for layer in model.parameters():
        layer.requires_grad = False

    inFeaturesLast = model.fc.in_features
    model.fc = torch.nn.Sequential(OrderedDict(
        [
            ('lin1', nn.Linear(in_features=inFeaturesLast, out_features=6, bias=True)),
        ]
    ))
    # unfroze the classification
    for linear in model.fc.parameters():
        linear.requires_grad = True

    return model


cuda = torch.device('cuda')
criterion = nn.CrossEntropyLoss()
numEpoch = 400
optim = 'ADAM'

paramsADAM = OrderedDict(
    lr=[.01, .001, 0.0001],
    batch_size=[100, 200, 400],
    num_workers=[1, 2, 4],
    device=['cuda']
)
paramsSGD = OrderedDict(
    lr=[.01, .001],
    momentum=[0.9],
    stepSize=[7],
    gamma = [0.1],
    batch_size=[100, 200, 400],
    num_workers=[1, 2, 4],
    device=['cuda']
)
manager = RunManager(STATISTIC_PATH, TIMESTAMP)
runsBuilder = RunBuilder(paramsSGD) if optim == 'SGD' else RunBuilder(paramsADAM)
for k, run in enumerate(runsBuilder.runs):
    try:
        device = torch.device(run.device)
        model = makeModel()
        model = model.to(device)
        model.train()
        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=run.batch_size, num_workers=run.num_workers)
        valLoader = torch.utils.data.DataLoader(valSet, batch_size=run.batch_size, num_workers=run.num_workers)

        if optim == 'SGD':
            optimizer = torch.optim.SGD(model.fc.parameters(), lr=run.lr, momentum=run.momentum)
            expLRScheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=run.stepSize, gamma=run.gamma)
        else:
            optimizer = torch.optim.Adam(model.fc.parameters(), lr=run.lr)

        manager.begin_run(run, model, trainLoader, valLoader)
        for epoch in trange(numEpoch, desc=f'Run {k + 1}/{len(runsBuilder)} epoch progress'):
            manager.begin_epoch()

            for images, labels in trainLoader:
                images, labels = images.to(run.device), labels.to(run.device)
                preds = model(images)
                loss = criterion(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if optim == 'SGD':
                    expLRScheduler.step()

                manager.track_train_loss(loss)
                manager.track_numTrain_correct(preds, labels)
            
            model.eval()
            for valImages, valLabels in valLoader:
                valImages, valLabels = valImages.to(run.device), valLabels.to(run.device)
                with torch.cuda.amp.autocast(enabled=True):
                    valOut = model(valImages)
                    valLoss = criterion(valOut, valLabels)

                manager.track_valid_loss(valLoss)
                manager.track_numValid_correct(valOut, valLabels)

            manager.end_epoch(f'{SAVE_MODEL_PATH}/{TIMESTAMP}')
            if manager.stop:    #? Skip running the rest of epochs if early stopped
                break

        manager.end_run(f'{SAVE_MODEL_PATH}/{TIMESTAMP}')
    except Exception as e:
        manager.writeError(msg=str(e))
        if 'CUDA' in str(e):  # Record cuda problem for later
            continue
        else:
            raise
# manager.save(f'{TIMESTAMP}', STATISTIC_PATH)