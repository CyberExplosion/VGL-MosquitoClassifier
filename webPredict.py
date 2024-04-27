import torch
from pathlib import Path
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
from collections import OrderedDict
from PIL import Image
import json
from io import BytesIO

DATAPATH = Path('Datasets/Test')

def predict(image,
            modelPath='savedModels/Mon May 15 21:26:20 2023/earlyStop_run_19.pt'):
    """
    Parameters
    ----------
    image : BytesIO
        The bytes representing the mosquito image
    modelPath : str
        The string for relative path to the saved model to use for prediction
    """
    imgTrans = transforms.Compose([transforms.ToTensor()])
    img = Image.open(image)
    imgTensor = imgTrans(img).unsqueeze(dim=0).cuda()
    predictModelPath = Path(modelPath)
    dataset = ImageFolder(DATAPATH)
    testModel = torchvision.models.resnet18(weights='DEFAULT')
    inFeatures = testModel.fc.in_features
    testModel.fc = torch.nn.Sequential(OrderedDict(
            [
                ('lin1', nn.Linear(in_features=inFeatures, out_features=6, bias=True)),
            ]
        ))
    testModel.load_state_dict(torch.load(predictModelPath))
    testModel.eval()
    testModel.cuda()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            output = testModel(imgTensor)
    percentageConfidence = torch.nn.functional.softmax(output.cpu().max(dim=0)[0], dim=0)
    print(f'Class: {dataset.classes[torch.argmax(output.cpu().max(dim=0)[0]).item()]} - Confidence: {torch.max(percentageConfidence, dim=0)[0]}')
    predictClass = torch.argmax(output.cpu().max(dim=0)[0])
    confidence = torch.max(percentageConfidence, dim=0)[0]
    ret = {'class': dataset.classes[predictClass.item()], 'confidence': confidence.item()}
    json_obj = json.dumps(ret, indent=4)
    return json_obj

if __name__ == '__main__':
    tempImgPath = Path('Datasets/Test/Anopheles sinensis/0_1_As_1.JPG')
    with open(tempImgPath, 'rb') as f:
        data = f.read()
        inImg = BytesIO(data)
    predict(inImg)