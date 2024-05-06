from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from starlette.responses import StreamingResponse

import numpy as np

import torch
from torchvision import transforms

from PIL import Image

from unet_for_api import UNET


SIGMOID_THRESHOLD = 0.55
transform = transforms.Compose([transforms.ToTensor(),
                                # transforms.RandomRotation(degrees=60),
                                transforms.Resize((256, 256), antialias=True)])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = UNET(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load('models/model.pth'))

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
        prediction = (prediction > SIGMOID_THRESHOLD).float()
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
