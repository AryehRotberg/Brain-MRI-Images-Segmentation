import streamlit as st

import numpy as np
from PIL import Image, ImageColor

import torch
from torchvision import transforms

import segmentation_models_pytorch as smp


constants: dict = {
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    
    'sigmoid_threshold': 0.55,
}

MODEL_PATH = 'models/production/unetplusplus_resnet34.pth'

# Image Transformer initialization
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256, 256), antialias=True)])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model Initialization
model = smp.UnetPlusPlus(encoder_name=constants['encoder_name'],
                         encoder_weights=constants['encoder_weights'],
                         in_channels=3,
                         classes=1).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Streamlit
st.title('Brain MRI Medical Images Segmentation')

uploaded_image = st.file_uploader('Choose an image.')

if uploaded_image is not None:
    color_picked = st.color_picker('**Pick a mask color**', '#FFFFFF')
    alpha = st.slider('**Choose alpha**', 0., 1.0, value=0.5)

    array_image = np.array(Image.open(uploaded_image).convert('RGB'))
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
    
    colored_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    colored_mask[prediction == 255] = ImageColor.getcolor(color_picked, 'RGB')

    transformed_image = transformed_image.cpu()
    transformed_image = transformed_image[0].permute(1, 2, 0)
    transformed_image = transformed_image.numpy() * 255
    transformed_image = transformed_image.astype(np.uint8)

    blended_image = transformed_image * (1 - alpha) + colored_mask * alpha
    blended_image = blended_image.astype(np.uint8)
    blended_image = Image.fromarray(blended_image).convert('RGB')

    st.image(blended_image)

    if prediction.max() > 0:
        st.info('The program has found a tumor in this MRI scan.')
    else:
        st.info('The program has not found any tumor in this MRI scan.')
