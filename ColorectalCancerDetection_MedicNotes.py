import os
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy.stats import zscore
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pydicom import dcmread

def load_text_data(filepath):
    data = pd.read_csv(filepath)
    return data.dropna(subset=["clinical_notes", "diagnosis"])

def prepare_text(tokenizer, text_data, max_length=128):
    encodings = tokenizer(
        list(text_data),
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="tf"
    )
    return encodings
def preprocess_images(image_paths, image_size=224):
    images = []
    for img_path in image_paths:
        dicom_image = dcmread(img_path).pixel_array
        image = Image.fromarray(dicom_image).convert("RGB").resize((image_size, image_size))
        images.append(np.array(image) / 255.0)  
    return np.array(images)

def build_image_model(input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=input_shape, pooling="avg", weights="imagenet")
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu")
    ])
    return model

def combine_features(text_features, image_features):
    text_features = zscore(text_features)
    image_features = zscore(image_features)
    combined = np.concatenate([text_features, image_features], axis=1)
    pca = PCA(n_components=100)
    return pca.fit_transform(combined)

def train_final_model(features, labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)
    
    model = models.Sequential([
        layers.Dense(256, activation="relu", input_shape=(features.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(len(set(encoded_labels)), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return model


if __name__ == "__main__":
    # MIMIC-III Dataset for Clinical Notes
    text_filepath = "/path/to/MIMIC-III/clinical_notes.csv"
    text_data = load_text_data(text_filepath)
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    encodings = prepare_text(tokenizer, text_data["clinical_notes"])
    
    text_model = TFAutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=256)
    text_features = text_model(encodings["input_ids"]).last_hidden_state.numpy().mean(axis=1)

    # TCIA Dataset for Imaging
    image_paths = [
        "/path/to/TCIA/colorectal_images/image1.dcm",
        "/path/to/TCIA/colorectal_images/image2.dcm"
    ]
    images = preprocess_images(image_paths)
    image_model = build_image_model(input_shape=(224, 224, 3))
    image_features = image_model.predict(images)

    # Combine Text and Image Features
    combined_features = combine_features(text_features, image_features)

    # Train Final Model
    train_final_model(combined_features, text_data["diagnosis"].tolist())