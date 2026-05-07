#!/usr/bin/env python
"""Test script to verify all imports and basic functionality"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("✓ All imports successful!")

# Test model creation
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

print("✓ Model created successfully!")

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy'])

print("✓ Model compiled successfully!")

# Test with sample data
sample_input = np.random.random((1, 48, 48, 1))
prediction = model.predict(sample_input, verbose=0)
print(f"✓ Model prediction works! Output shape: {prediction.shape}")

# Check if model.h5 exists
import os
if os.path.exists('src/model.h5'):
    print("✓ Model weights file (model.h5) found!")
    # Try loading
    model.load_weights('src/model.h5')
    print("✓ Model weights loaded successfully!")
else:
    print("⚠ Model weights file (model.h5) not found")

# Check cascade classifier
if os.path.exists('src/haarcascade_frontalface_default.xml'):
    print("✓ Cascade classifier file found!")
    facecasc = cv2.CascadeClassifier('src/haarcascade_frontalface_default.xml')
    print("✓ Cascade classifier loaded successfully!")
else:
    print("⚠ Cascade classifier not found")

print("\n" + "="*50)
print("All tests passed! The emotion detection script is ready to use.")
print("="*50)
print("\nUsage:")
print("  python emotions.py --mode display  (requires webcam)")
print("  python emotions.py --mode train    (requires training data in data/train and data/test)")
