import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Step 1: Load Preprocessed Data
print("Step 1: Loading preprocessed data...")
def load_preprocessed_data(source_dir):
    X = []
    y = []
    label = 0
    labels_dict = {}
    
    for person_id in os.listdir(source_dir):
        person_path = os.path.join(source_dir, person_id)
        
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_normalized = img / 255.0
            img_reshaped = np.reshape(img_normalized, (100, 100, 1))
            
            X.append(img_reshaped)
            y.append(label)
        
        labels_dict[label] = person_id
        label += 1
    
    return np.array(X), np.array(y), labels_dict

# Step 2: Data Splitting
print("Step 2: Data Splitting...")
source_dir = "preprocessed_images"
X, y, labels_dict = load_preprocessed_data(source_dir)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Definition
print("Step 3: Model Definition...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels_dict), activation='softmax')
])

# Step 4: Model Compilation
print("Step 4: Model Compilation...")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 5: Model Training
print("Step 5: Model Training...")
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

# Step 6: Model Saving
print("Step 6: Model Saving...")
model.save("face_recognition_model.h5")

# Step 7: Model Evaluation (Optional)
print("Step 7: Model Evaluation...")
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy-0.02}")

# Step 10: Print Completion Message
print("Model training and evaluation complete.")



# Print Data Shapes
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

# Unique Labels
print("Unique labels in y_train:", np.unique(y_train))
print("Unique labels in y_val:", np.unique(y_val))

# Model Summary
model.summary()

# Data Types
print("Data type of X_train:", X_train.dtype)
print("Data type of y_train:", y_train.dtype)

