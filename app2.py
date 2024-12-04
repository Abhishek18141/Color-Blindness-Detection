import streamlit as st
import cv2
import numpy as np
import joblib
import xgboost as xgb
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load models
binary_model_path = 'svm_model_bin_U7.pkl'
multi_model_path = 'svm_model_multi_U7.pkl'
mlp_bin_model_path = 'mlp_model_bin_U7.pkl'
mlp_multi_model_path = 'mlp_model_multi_U7.pkl'
xgb_bin_model_path = 'xgb_model_bin_U7.json'
xgb_multi_model_path = 'xgb_model_multi_U7.json'

# Load SVM models
svm_model_bin = joblib.load(binary_model_path)
svm_model_multi = joblib.load(multi_model_path)

# Load MLP models
mlp_model_bin = joblib.load(mlp_bin_model_path)
mlp_model_multi = joblib.load(mlp_multi_model_path)

# Load XGBoost models
xgb_model_bin = xgb.XGBClassifier()  # Binary classification
xgb_model_bin.load_model(xgb_bin_model_path)

xgb_model_multi = xgb.XGBClassifier()  # Multiclass classification
xgb_model_multi.load_model(xgb_multi_model_path)

# Manually set the number of classes for the multiclass model
xgb_model_multi.n_classes_ = 3  # Adjust based on your dataset

# Load the InceptionResNetV2 model
IMG_SIZE = 299
inception_resnet_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x_inception_resnet = GlobalAveragePooling2D()(inception_resnet_base.output)
inception_resnet_model = Model(inputs=inception_resnet_base.input, outputs=x_inception_resnet)

# Function to process new images
def preprocess_new_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Expand dimensions to fit model input
    features = inception_resnet_model.predict(img)
    return features

# Streamlit app
st.title("Diabetic Retinopathy Classification App")
st.write("Upload an image for classification")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, channels="BGR")

    # Process and extract features
    features = preprocess_new_image(image)

    # Create buttons for predictions
    if st.button("Predict Binary Class"):
        # Make binary predictions
        binary_pred_svm = svm_model_bin.predict(features)
        binary_pred_mlp = mlp_model_bin.predict(features)
        binary_pred_xgb = xgb_model_bin.predict(features)

        # Map binary predictions to categories
        binary_category_svm = "No DR" if binary_pred_svm[0] == 0 else "DR"
        binary_category_mlp = "No DR" if binary_pred_mlp[0] == 0 else "DR"

        # Display binary predictions
        st.write(f"SVM Prediction (Binary): {binary_category_svm}")
        st.write(f"MLP Prediction (Binary): {binary_category_mlp}")
        st.write(f"XGBoost Prediction (Binary): {binary_category_svm if binary_pred_xgb[0] == 1 else 'No DR'}")

    if st.button("Predict Multiclass"):
        # Make multiclass predictions
        multi_pred_svm = svm_model_multi.predict(features)
        multi_pred_mlp = mlp_model_multi.predict(features)
        multi_pred_xgb = xgb_model_multi.predict(features)

        # Map multiclass predictions to categories
        multi_categories = {
            0: "No DR",
            1: "Mild Non-Proliferative DR",
            2: "Proliferative DR"
        }

        # Display multiclass predictions
        st.write(f"SVM Prediction (Multiclass): {multi_categories[multi_pred_svm[0]]}")
        st.write(f"MLP Prediction (Multiclass): {multi_categories[multi_pred_mlp[0]]}")
        st.write(f"XGBoost Prediction (Multiclass): {multi_categories[multi_pred_xgb[0]]}")
