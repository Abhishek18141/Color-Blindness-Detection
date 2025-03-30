# Diabetic Retinopathy Classification App

This project is a web-based application built using **Streamlit** to classify diabetic retinopathy using pre-trained machine learning models. The app supports both binary classification (No DR vs. DR) and multiclass classification (No DR, Mild Non-Proliferative DR, and Proliferative DR).
**Live Demo**: https://color-blindness-detection.streamlit.app/
---

## Features
- **Image Upload**: Users can upload retina images in PNG, JPG, or JPEG formats.
- **Feature Extraction**: Utilizes the `InceptionResNetV2` model to extract features from retina images.
- **Classification**:
  - Binary classification using SVM, MLP, and XGBoost models.
  - Multiclass classification using SVM, MLP, and XGBoost models.
- **Interactive UI**: Buttons for predictions, displaying results in real-time.

---

## Technologies Used
- **Python Libraries**:
  - [Streamlit](https://streamlit.io/) for the user interface.
  - [OpenCV](https://opencv.org/) for image processing.
  - [TensorFlow](https://www.tensorflow.org/) for feature extraction with `InceptionResNetV2`.
  - [NumPy](https://numpy.org/) for numerical operations.
  - [joblib](https://joblib.readthedocs.io/) for loading SVM and MLP models.
  - [XGBoost](https://xgboost.ai/) for loading and using XGBoost models.

- **Machine Learning Models**:
  - Binary classification: `svm_model_bin_U7.pkl`, `mlp_model_bin_U7.pkl`, `xgb_model_bin_U7.json`.
  - Multiclass classification: `svm_model_multi_U7.pkl`, `mlp_model_multi_U7.pkl`, `xgb_model_multi_U7.json`.

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Required libraries (listed in `requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetic-retinopathy-classification.git
   cd diabetic-retinopathy-classification
