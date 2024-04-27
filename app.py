import os
import json
from fastapi import FastAPI, Form, Request,  File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles  
from io import BytesIO
from PIL import Image
import base64

from webPredict import predict

app = FastAPI()
templates = Jinja2Templates(directory='templates')
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/predict', response_class=JSONResponse)
async def predict_(request: Request, image: UploadFile = Form(...)):
    # Read the image file contents
    contents = await image.read()
    pil_image = Image.open(BytesIO(contents))

    # Process the image and get the prediction
    prediction = predict(BytesIO(contents))
    print(type(prediction))
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    buffered.seek(0)
    
    prediction = json.loads(prediction)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
    # Render the HTML response with the image and prediction
    return templates.TemplateResponse('index.html', {'request': request, 'image': img_base64, 'prediction': prediction})

