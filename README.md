# Skin Cancer Detection using Attention-Based Deep Learning

An AI-powered skin cancer detection system developed using Deep Learning and Explainable AI techniques.
This project began as a mini project and is being continuously enhanced toward a final year major project.

---

## Project Overview

This system detects different categories of skin cancer lesions using an attention-based deep learning model.
The application also includes Grad-CAM heatmap visualization to improve model interpretability and explain the AI's predictions visually.

---

## Features

- Skin lesion image classification
- Attention-based deep learning architecture
- Grad-CAM heatmap visualization
- Interactive Streamlit web application
- Explainable AI (XAI) support
- GPU acceleration with CUDA support

---

## Technologies Used

- Python
- PyTorch
- Streamlit
- OpenCV
- Matplotlib
- Pandas
- Scikit-learn

---

## Project Structure

```bash
Skin_Cancer_Attention_Project/
│
├── app.py
├── train_final.py
├── test_visual.py
├── model_attention.py
├── attention_blocks.py
├── explainability.py
├── data_loader.py
├── skin_attention_model.pth
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone Repository

```bash
git clone YOUR_GITHUB_REPO_LINK
cd Skin_Cancer_Attention_Project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
streamlit run app.py
```

---

## Explainable AI Output

The project generates Grad-CAM heatmaps showing the regions focused on by the AI model during prediction.

---

## Current Status

This project is currently under active development and will be further expanded as a final year project with additional features and improved model performance.

---

## Future Enhancements

- Improved model accuracy
- Real-time image prediction
- Cloud deployment
- Patient history management
- Enhanced medical report generation
- Multi-model comparison

---

## Author

Developed by Rohith
