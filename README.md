# Satellite Image Classification

## Problem Description

This project addresses the problem of **satellite image classification**, where the goal is to automatically classify satellite images into predefined classes (**cloudy**, **desert**, **green_area**, **water**) based on visual content.

Satellite image classification is a fundamental task in remote sensing and is widely used in:
- environmental monitoring
- urban planning
- agriculture analysis
- disaster assessment

**Real-world usage example:**  
The model can be used by government agencies and private companies to automatically analyze satellite imagery for monitoring land use changes, detecting urban expansion, and assessing environmental impact.

The task is formulated as a **multi-class image classification** problem. A convolutional neural network (CNN) is trained to learn discriminative visual patterns from satellite imagery and predict the correct class label for unseen images.

---

## Dataset Description with Fields and Classes

The project uses the **[Satellite Image Classification](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification)** dataset from Kaggle (published by Mahmoud Reda).  
The dataset consists of satellite images organized in a **folder-based structure**, where each subdirectory represents a single class. This structure is fully compatible with `torchvision.datasets.ImageFolder`.

Directory structure:

```text
data/
├── cloudy/
├── desert/
├── green_area/
└── water/
```
### Sample Images from the Dataset

Below are example satellite images for each class in the dataset.

| Cloudy | Desert |
|--------|--------|
| ![Cloudy sample](images/cloudy_example.jpg) | ![Desert sample](images/desert_example.jpg) |

| Green Area | Water |
|------------|-------|
| ![Green area sample](images/green_area_example.jpg) | ![Water sample](images/water_example.jpg) |

### Classes

The dataset contains four classes:

| Class name | Description |
|-----------|-------------|
| **cloudy** | Satellite images dominated by cloud cover. This class represents scenes where clouds obscure the Earth’s surface, which is a common challenge in satellite image analysis and remote sensing. |
| **desert** | Images of arid and semi-arid regions with sandy terrain and minimal vegetation. These images are important for identifying barren land and monitoring desertification. |
| **green_area** | Images characterized by dense vegetation such as forests, grasslands, and agricultural fields. This class is essential for environmental monitoring, agriculture, and ecosystem analysis. |
| **water** | Images containing large water bodies such as rivers, lakes, or seas. This class supports applications related to hydrology, flood monitoring, and water resource management. |

### Target Variable

- **Target name:** `target`  
- **Type:** Categorical  
- **Task:** Multi-class image classification  
- **Description:** The model predicts the category of a satellite image, selecting one of the four classes listed above.

### Class Distribution

The dataset contains approximately **5,600 images** in total and is moderately imbalanced:

- **cloudy:** ~1,500 images  
- **desert:** ~1,100 images  
- **green_area:** ~1,500 images  
- **water:** ~1,500 images  

---

## Directory Structure

```text
.
├── README.md
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Preprocessed data (if applicable)
├── notebooks/
│   └── notebook.ipynb    # EDA and experiments
├── src/
│   ├── train.py          # Model training script
│   ├── predict.py        # Inference logic
│   └── utils.py          # Helper functions
├── app/
│   └── main.py           # FastAPI service
├── model/
│   └── model.pth         # Saved trained model
├── requirements.txt
├── Dockerfile
├── aws/
│   └── deploy_ec2.md     # AWS deployment notes
└── screenshots/
```

---

## Exploratory Data Analysis (EDA)

EDA is performed in `notebooks/notebook.ipynb` and includes:

- Dataset size and class distribution
- Visualization of sample images per class
- Image resolution and channel analysis
- Basic data augmentation inspection

For image-specific analysis:
- Visual inspection of representative images
- Checking class imbalance

**TODO:**
- Add EDA plots (class distribution)
- Add example images per class
- Add explanation of observed patterns

---

## Model Training

Several deep learning approaches were explored.

### Models
- Custom **ResNet9** architecture (CNN with residual connections)

### Training details
- Optimizer: Adam
- Loss function: Cross-Entropy Loss
- Learning rate: TODO
- Number of epochs: TODO
- Batch size: TODO

The final model is trained using the script:

```bash
python src/train.py
```

The trained model is saved to:

```text
model/model.pth
```

**TODO:**
- Add training/validation accuracy table
- Add loss/accuracy curves
- Mention experiments with alternative architectures (if added later)

---

## Reproducibility

The project is fully reproducible.

To reproduce results:

```bash
git clone <REPO_URL>
cd satellite-image-classification
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

- Random seeds are fixed
- Dataset is either included or has clear download instructions
- Training and inference scripts can be executed without modification

**TODO:**
- Add dataset download link (if not included)

---

## Model Deployment (FastAPI)

The trained model is deployed as a REST API using **FastAPI**.

To run the service locally:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Example request

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"image": "<base64-encoded-image>"}'
```

**TODO:**
- Add request/response JSON schema
- Add screenshot of API testing

---

## Dependency and Environment Management

Dependencies are managed via `requirements.txt`.

Main libraries:
- torch
- torchvision
- fastapi
- uvicorn
- numpy

Environment setup:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Containerization

The application is fully containerized using Docker.

### Build Docker image

```bash
docker build -t satellite-classification .
```

### Run container

```bash
docker run -p 8000:8000 satellite-classification
```

**TODO:**
- Add explanation of exposed ports

---

## Cloud Deployment (AWS Deployment)

The service is deployed on **AWS EC2** using Docker.

Deployment steps:
1. Launch EC2 instance
2. Install Docker
3. Clone repository
4. Build Docker image
5. Run container

The service is accessible at:

```text
http://<EC2_PUBLIC_IP>:8000
```

**TODO:**
- Add EC2 instance type
- Add security group configuration
- Add screenshots or video of deployed service

---

## Conclusion

This project demonstrates an end-to-end deep learning pipeline for satellite image classification, including:
- Data exploration
- Model training
- API deployment
- Containerization
- Cloud deployment

The solution is designed to be reproducible, extensible, and production-ready.

