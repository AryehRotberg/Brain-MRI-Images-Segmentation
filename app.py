import streamlit as st

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from unet_for_api import UNET


SIGMOID_THRESHOLD = 0.55

# Image Transformer initialization
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256, 256), antialias=True)])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model Initialization
model = UNET(in_channels=3, out_channels=1)
model.load_state_dict(torch.load('models/model.pth', map_location=torch.device(device)))

st.title('Brain MRI Medical Images Segmentation')

uploaded_image = st.file_uploader('Choose an image.')

if uploaded_image is not None and st.button('Apply'):
    array_image = np.array(Image.open(uploaded_image).convert('RGB'))
    transformed_image = transform(array_image)
    transformed_image = transformed_image.unsqueeze(0)

    model.eval()

    with torch.no_grad():
        prediction = model(transformed_image)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > SIGMOID_THRESHOLD).float()
        prediction = prediction.squeeze()
        prediction = prediction.numpy()
        prediction = prediction * 255
    
    transformed_image = transformed_image[0].permute(1, 2, 0)
    transformed_image = transformed_image.numpy() * 255
    transformed_image = transformed_image.astype(np.uint8)

    input_image = Image.fromarray(transformed_image).convert('RGB')
    predicted_mask = Image.fromarray(prediction).convert('L')
    input_image.paste(predicted_mask, (0, 0), mask=predicted_mask)

    st.image(input_image)

    if prediction.max() > 0:
        st.info('The program has found a tumor in this MRI scan.')
    else:
        st.info('The program has not found any tumor in this MRI scan.')
