import streamlit as st

import numpy as np
from PIL import Image, ImageColor, ImageChops

import cv2

import torch
from torchvision import transforms

import segmentation_models_pytorch as smp


constants: dict = {
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    
    'sigmoid_threshold': 0.55,

    'model_path': 'models/production/unetplusplus_resnet34.pth'
}

# Image Transformer initialization
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256, 256), antialias=True)])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model Initialization
@st.cache_resource
def load_model() -> smp.UnetPlusPlus:
    '''
    Returns:
        model: smp.UnetPlusPlus
    '''
    model = smp.UnetPlusPlus(encoder_name=constants['encoder_name'],
                             encoder_weights=constants['encoder_weights'],
                             in_channels=3,
                             classes=1).to(device)

    model.load_state_dict(torch.load(constants['model_path'], map_location=device))

    return model

def draw_bounding_boxes(mask: np.array, color: tuple) -> np.array:
    '''
    Arguments:
        mask: np.array (numpy)
        color: tuple
    
    Returns:
        np.array (numpy)
    '''
    mask = mask.astype(np.uint8)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(mask_bgr, (x, y), (x + w, y + h), color, 1)
    
    return mask_bgr

# Streamlit
st.title('Brain MRI Medical Images Segmentation')

uploaded_image = st.file_uploader('Choose an image.')

if uploaded_image is not None:
    color_picked = st.color_picker('**Pick a bounding box color**', '#FF0000')

    array_image = np.array(Image.open(uploaded_image).convert('RGB'))
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
    prediction = draw_bounding_boxes(prediction, ImageColor.getcolor(color_picked, 'RGB'))

    transformed_image = transformed_image.cpu()
    transformed_image = transformed_image[0].permute(1, 2, 0)
    transformed_image = transformed_image.numpy() * 255
    transformed_image = transformed_image.astype(np.uint8)

    input_image = Image.fromarray(transformed_image).convert('RGBA')
    predicted_mask = Image.fromarray(prediction).convert('RGBA')

    st.image(ImageChops.screen(input_image, predicted_mask), width=400)

    if prediction.max() > 0:
        st.info('The program has found a tumor in this MRI scan.')
    else:
        st.info('The program has not found any tumor in this MRI scan.')
