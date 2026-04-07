import os, zipfile

# Unzip the model if .h5 file is not already extracted
if not os.path.exists("cifar10_model.h5") and os.path.exists("cifar10_model.zip"):
    with zipfile.ZipFile("cifar10_model.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
        
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
import tensorflow as tf
import streamlit as st

st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🐾", layout="wide")

import tensorflow as tf
import numpy as np
from PIL import Image
import os
# CIFAR-10 Classes
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load or train model
@st.cache_resource
def load_model():
    if os.path.exists("cifar10_model.h5"):
        model = tf.keras.models.load_model("cifar10_model.h5")
    else:
        st.warning("⚠️ Model not found! Training a new model (this may take 1-2 mins)...")

        # Load CIFAR-10 dataset
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = x_train/255.0, x_test/255.0
        y_train, y_test = y_train.flatten(), y_test.flatten()

        from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
        from tensorflow.keras.models import Model

        K = len(set(y_train))
        i = Input(shape=x_train[0].shape)
        x = Conv2D(32, (3,3), activation='relu', padding='same')(i)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(K, activation='softmax')(x)

        model = Model(i, x)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Train with fewer epochs for quick demo
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=64)
        model.save("cifar10_model.h5")

    return model

model = tf.keras.models.load_model("cifar10_model.h5")


# Custom CSS for a modern, attractive UI
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #4CAF50 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Card styling for columns */
    div[data-testid="stVerticalBlock"] > div {
        background-color: #1e2127;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        background-color: #262930 !important;
        padding: 20px;
        transition: 0.3s;
    }
    .stFileUploader:hover {
        border-color: #45a049;
        background-color: #2d313a !important;
    }
    
    /* Image caption */
    .stImage caption {
        font-size: 1.1rem;
        color: #a0aab5;
    }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.title("🐾 CIFAR-10 Image Classifier")
st.markdown(
    "Upload an image of one of the categories below, and our **Deep Learning Model** will classify it instantly! "
    "*(Categories: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)*"
)
st.write("---")

# Main Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📂 Upload an Image")
    uploaded_file = st.file_uploader("Choose a JPG or PNG image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display original user image
        image = Image.open(uploaded_file)
        # Convert RGBA to RGB if needed to avoid shape issues
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        st.image(image, caption="📸 Your Image", use_container_width=True)

with col2:
    if uploaded_file is not None:
        st.subheader("🤖 Model Analysis")
        
        with st.spinner("Analyzing image..."):
            # Prepare image for the model (32x32 size for CIFAR-10)
            img_resized = image.resize((32, 32))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = model.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Results
            st.success(f"### 🎉 Prediction: **{CLASS_NAMES[class_index].capitalize()}**")
            st.info(f"**Confidence Score:** {confidence * 100:.1f}%")
            
            st.write("---")
            st.write("### 🔎 Top 3 Predictions")
            
            # Show top 3 predictions with progress bars
            top_indices = prediction[0].argsort()[-3:][::-1]
            for i in top_indices:
                class_name = CLASS_NAMES[i].capitalize()
                pred_prob = float(prediction[0][i])
                
                # Format progress bar color based on confidence
                st.write(f"**{class_name}** - {pred_prob * 100:.1f}%")
                st.progress(pred_prob)
    else:
        st.info("👈 Please upload an image on the left to see the classification results.")
        st.image("https://images.unsplash.com/photo-1549692520-acc6669e2f0c?q=80&w=600&auto=format&fit=crop", 
                 caption="Waiting for your input...", use_container_width=True)
