import cv2
import os
import numpy as np

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Resize the image
    img_resized = cv2.resize(img, (100, 100))
    
    # Normalize the pixel values
    img_normalized = img_resized / 255.0
    
    return img_normalized

# Ask for the specific person_id to preprocess
person_id_to_preprocess = input("Enter the ID for the person you want to preprocess: ")

# Source and destination directories
source_dir = "face_images"
dest_dir = "preprocessed_images"

# Create destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

# Check if person_id exists in source directory
if person_id_to_preprocess in os.listdir(source_dir):
    person_path = os.path.join(source_dir, person_id_to_preprocess)
    dest_person_path = os.path.join(dest_dir, person_id_to_preprocess)
    
    # Create a folder for this individual in the destination directory
    if not os.path.exists(dest_person_path):
        os.mkdir(dest_person_path)
    
    # Loop through each image
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        
        # Preprocess image
        img_preprocessed = preprocess_image(img_path)
        
        # Save preprocessed image
        dest_img_path = os.path.join(dest_person_path, img_name)
        cv2.imwrite(dest_img_path, img_preprocessed * 255)  # Multiply by 255 to revert normalization

    print(f"Data preprocessing complete for {person_id_to_preprocess}.")
else:
    print(f"No data found for {person_id_to_preprocess}.")

