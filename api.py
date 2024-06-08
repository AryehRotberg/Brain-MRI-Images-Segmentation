from typing import Tuple
from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

import numpy as np

import cv2

import torch
from torchvision import transforms

from PIL import Image, ImageChops

import segmentation_models_pytorch as smp


constants: dict = {
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    
    'sigmoid_threshold': 0.55,

    'model_path': 'models/production/unetplusplus_resnet34.pth'
}

def load_model() -> smp.UnetPlusPlus:
    '''
    Returns:
        model: smp.UnetPlusPlus
    '''
    global model

    if model is None:
        model = smp.UnetPlusPlus(encoder_name=constants['encoder_name'],
                                 encoder_weights=constants['encoder_weights'],
                                 in_channels=3,
                                 classes=1).to(device)
        
        model.load_state_dict(torch.load(constants['model_path'], map_location=device))

    return model

def draw_bounding_boxes(mask: np.array) -> Tuple[np.array, float]:
    '''
    Arguments:
        mask: np.array (numpy)
    
    Returns:
        Tuple[np.array, float]
    '''
    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return mask, 0.0

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mean_area = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        max_x = x + w
        max_y = y + h

        cv2.rectangle(mask_bgr, (x, y), (max_x, max_y), (255, 0, 0), 1)
        mean_area.append(abs(max_x - x) * abs(max_y - y))
    
    return mask_bgr, sum(mean_area) / len(mean_area)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256, 256), antialias=True)])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = None

app = FastAPI(title='Brain MRI Medical Images Segmentation')

app.add_middleware(CORSMiddleware,
                   allow_headers=['*'],
                   expose_headers=['has_tumor', 'average_pixel_area'])

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    input_image = Image.open(BytesIO(await file.read())).convert('RGB')
    array_image = np.array(input_image)
    transformed_image = transform(array_image)
    transformed_image = transformed_image.unsqueeze(0)
    transformed_image = transformed_image.to(device)

    model = load_model()
    model.eval()

    with torch.no_grad():
        prediction = model(transformed_image)
    
    prediction = torch.sigmoid(prediction)
    prediction = (prediction > constants['sigmoid_threshold']).float()
    prediction = prediction.squeeze()
    prediction = prediction.cpu()
    prediction = prediction.numpy()
    prediction = prediction * 255
    prediction, mean_area = draw_bounding_boxes(prediction)

    transformed_image = transformed_image.cpu()
    transformed_image = transformed_image[0].permute(1, 2, 0)
    transformed_image = transformed_image.numpy() * 255
    transformed_image = transformed_image.astype(np.uint8)

    input_image = Image.fromarray(transformed_image).convert('RGBA')
    predicted_mask = Image.fromarray(prediction).convert('RGBA')
    result = ImageChops.screen(input_image, predicted_mask)

    bytes_io = BytesIO()
    result.save(bytes_io, format='PNG')
    bytes_io.seek(0)
    
    return StreamingResponse(bytes_io, media_type='image/png', headers={'has_tumor': f'{prediction.max() > 0}', 'average_pixel_area': f'{round(mean_area, 2)}'})
