from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from starlette.responses import StreamingResponse

import numpy as np

import torch
from torchvision import transforms

from PIL import Image

import segmentation_models_pytorch as smp


constants: dict = {
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    
    'sigmoid_threshold': 0.55,
}

MODEL_PATH = 'models/production/unetplusplus_resnet34.pth'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256, 256), antialias=True)])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = smp.UnetPlusPlus(encoder_name=constants['encoder_name'],
                         encoder_weights=constants['encoder_weights'],
                         in_channels=3,
                         classes=1).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

app = FastAPI(title='Brain MRI Medical Images Segmentation')

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    input_image = Image.open(BytesIO(await file.read())).convert('RGB')
    array_image = np.array(input_image)
    transformed_image = transform(array_image)
    transformed_image = transformed_image.unsqueeze(0)
    transformed_image = transformed_image.to(device)

    model.eval()

    with torch.no_grad():
        prediction = model(transformed_image)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > constants['sigmoid_threshold']).float()
        prediction = prediction.squeeze()
        prediction = prediction.cpu()
        prediction = prediction.numpy()
        prediction = prediction * 255
    
    transformed_image = transformed_image.cpu()
    transformed_image = transformed_image[0].permute(1, 2, 0)
    transformed_image = transformed_image.numpy() * 255
    transformed_image = transformed_image.astype(np.uint8)

    input_image = Image.fromarray(transformed_image).convert('RGB')
    predicted_mask = Image.fromarray(prediction).convert('L')
    input_image.paste(predicted_mask, (0, 0), mask=predicted_mask)

    bytes_io = BytesIO()
    input_image.save(bytes_io, format='PNG')
    bytes_io.seek(0)
    
    return StreamingResponse(bytes_io, media_type='image/png', headers={'has_tumor': f'{prediction.max() > 0}'})
