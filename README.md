# Brain MRI Images Segmentation using U-NET++ Architecture

![Figure_1](https://github.com/AryehRotberg/Brain-MRI-Images-Segmentation/assets/49247848/728459d3-f7c5-49d2-826d-39d2ec60cb31)

## Overview

This project focuses on the automated segmentation of brain tumors from MRI scans using deep learning techniques. The model is built on the U-Net++ architecture, implemented in PyTorch, and enhanced with various advanced techniques to improve accuracy and robustness. The ultimate goal is to aid in the diagnosis and treatment planning for brain tumor patients.

## Technologies Used

- Programming Languages: Python
- Deep Learning Frameworks: PyTorch, Segmentation Models (segmentation-models-pytorch)
- Web Frameworks: FastAPI, Streamlit
- Data Processing: NumPy, pandas, scikit-learn
- Image Processing: PIL, OpenCV
- Visualization: Matplotlib
- Model Tracking: MLFlow
- Containerization: Docker

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
5. Docker Setup:
   - Build the Docker image: docker build -t brain-mri-images-segmentation .
   - Run the Docker container: docker run -p 8000:8000 brain-mri-images-segmentation

## Contribution

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.
