import argparse
import torch
from pathlib import Path
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision
from collections import OrderedDict
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

def singleImgPredict(testModel, dataset):
    singlePath = Path(args.image)
    imgTrans = transforms.Compose([transforms.ToTensor()])
    img = Image.open(singlePath)
    imgTensor = imgTrans(img).unsqueeze(dim=0).cuda()
    print(imgTensor)
    print(imgTensor.shape)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            output = testModel(imgTensor)
    percentageConfidence = torch.nn.functional.softmax(output.cpu().max(dim=0)[0], dim=0)
    print(f'Confidence: {torch.max(percentageConfidence, dim=0)[0]}')
    predictClass = torch.argmax(output.cpu().max(dim=0)[0])
    confidence = torch.max(percentageConfidence, dim=0)[0]
    return {'classIdx': predictClass.item(), 'confidence': confidence.item()}

def groupPredict(testModel, dataset):
    predList = []
    trueList = []
    testLoader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=4)

    with torch.no_grad():
        for img, label in iter(testLoader):
            img = img.cuda()
            label = label.cuda()
            
            with torch.cuda.amp.autocast(enabled=False):
                output = testModel(img)

            predList.extend(output.cpu().max(dim=1)[1])
            trueList.extend(label.cpu())
        
        acc = accuracy_score(trueList, predList)
        # print(f'condfidence of the first img: {output.cpu().max(dim=0)}')
        percentageConfidence = torch.nn.functional.softmax(output.cpu().max(dim=0)[0], dim=0)
        print(f'Confidence: {torch.max(percentageConfidence, dim=0)[0]}')
        print(f'Accuracy: {acc}')
        print(classification_report(trueList, predList))

def predict(args):
    fullPath = Path(pathName, modelName)
    dataPath = Path(args.data)
    imgTrans = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    dataset = ImageFolder(dataPath, transform=imgTrans)
    testModel = torchvision.models.resnet18(weights='DEFAULT')
    inFeature = testModel.fc.in_features
    testModel.fc = torch.nn.Sequential(OrderedDict(
            [
            ('lin1', nn.Linear(in_features=inFeature, out_features=6, bias=True)),
            ]
        ))
    testModel.load_state_dict(torch.load(fullPath))
    testModel.eval()
    testModel.cuda()
    res = None
    if args.use_single:
        singleRes = singleImgPredict(testModel, dataset)
        res = {'species': dataset.classes[singleRes['classIdx']], 'confidence': singleRes['confidence']}
    else:
        groupPredict(testModel, dataset)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the species class using a prediction model')
    parser.add_argument('--name', '-N',
                        help='The name of the saved model',
                        default='_run:26_epoch:500_loss:0.0313185274953375_accuracy:1.0_epoch duration:11.824613094329834_run duration:10446.869837760925_lr:0.0001_batch_size:400_num_workers:4_device:cuda.pt'
                        )
    parser.add_argument('--path', '-P',
                        help='The folder path to all saved model',
                        default='savedModels/Thu May  4 04:20:06 2023'
                        )
    parser.add_argument('--use-single', '-U',
                        help='Enable predicting only one image',
                        default=False
                        )
    parser.add_argument('--image', '-I',
                        help='The single image path to predict'
                        )
    parser.add_argument('--data', '-D',
                        help='The root image folder to test. Ex: root/classes of images/xxx.png',
                        default='Datasets/Test'
                        )
    args = parser.parse_args()
    # print(args)
    modelName = args.name
    pathName = args.path
    if args.use_single and not args.image:
        raise Exception('There is no image path to use. Use option -I')
    predict(args)