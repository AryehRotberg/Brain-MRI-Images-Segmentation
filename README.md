# Brain MRI Images Segmentation using U-NET++ Architecture

![project_1_image_2](https://github.com/AryehRotberg/Brain-MRI-Images-Segmentation/assets/49247848/a1446e2e-1225-4b5a-9fae-463f490485e7)

## Overview

This project focuses on the automated segmentation of brain tumors from MRI scans using deep learning techniques. The model is built on the U-Net++ architecture, implemented in PyTorch, and enhanced with various advanced techniques to improve accuracy and robustness. The ultimate goal is to aid in the diagnosis and treatment planning for brain tumor patients.

## Technologies Used

- Programming Languages: Python
- Deep Learning Frameworks: PyTorch, segmentation_models_pytorch
- Web Frameworks: FastAPI
- Data Processing: NumPy, pandas, scikit-learn
- Image Processing: PIL
- Visualization: Matplotlib
- Model Tracking: MLFlow

## Getting Started

### Installation

1. Clone this repository: git clone https://github.com/AryehRotberg/Brain-MRI-Images-Segmentation.git
2. Create and activate a virtual environment:
 - python -m venv venv
 - source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install the required packages: pip install -r requirements.txt

## Usage

1. Train the model: python src/training_pipeline.py --raw_data_path data/raw --images_path data/images --masked_images data/masked_images --model_path models/production/model.pth --sorted_data_available ""
2. Evaluate the model: python src/evaluation_pipeline.py --images_path data/images --masked_images data/masked_images --model_path models/production/unetplusplus_resnet34.pth
3. Run Streamlit website: streamlit run app.py
4. Run the web interface: uvicorn api:app --host 0.0.0.0 --port 8000
   Access the web interface at http://0.0.0.0:8000/
   Alternatively, you can access the FastAPI page at http://0.0.0.0:8000/docs/

## Contribution

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.
