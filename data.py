import pandas as pd
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_annotations(csv_file):
    return pd.read_csv(csv_file)


def create_mask(row, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for _, r in row.iterrows():
        xmin, ymin, xmax, ymax = r['xmin'], r['ymin'], r['xmax'], r['ymax']
        label = 1 if r['label'] == 'rbc' else 2  # 1 for RBC, 2 for WBC
        mask[int(ymin):int(ymax), int(xmin):int(xmax)] = label
    return mask


def preprocess_data(annotations, base_path='resources/images'):
    images = []
    masks = []
    for image_file in annotations['image'].unique():
        full_path = os.path.join(base_path, image_file)

        if not os.path.exists(full_path):
            print(f"Image file {full_path} not found.")
            continue

        img = load_img(full_path, color_mode='rgb')
        img = img_to_array(img)
        img = cv2.resize(img, (256, 256))
        images.append(img)

        mask = create_mask(annotations[annotations['image'] == image_file], img.shape)
        masks.append(mask)

    images = np.array(images, dtype='float32') / 255.0
    masks = np.array(masks)
    masks = to_categorical(masks, num_classes=3)  # 3 classes: background, RBC, WBC

    return train_test_split(images, masks, test_size=0.2, random_state=42)

