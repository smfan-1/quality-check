# Importing libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import serial
import time

# Load the pre-trained model
model = load_model("product_quality.h5")

# Dataset path
data_dir = r"D:\DSMP\PROJECTS\dl"

# Image parameters
img_height, img_width = 128, 128
batch_size = 15

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


def predict_live_with_trigger():
    # Setup serial communication with Arduino
    # Replace 'COM3' with the port your Arduino is connected to.
    COM_PORT = 'COM4'
    ser = serial.Serial(COM_PORT, 9600, timeout=1)
    time.sleep(3)  # Allow time for Arduino to reset after opening the port

    # Load class label names (make sure these match your training configuration)
    labels = list(train_generator.class_indices.keys())  # e.g., ["Defective", "Original", ...]

    # For this scenario, assume "Original" means non-defective.
    # Any category that is not "Original" will be treated as defective.

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera")
        return

    print("Press 'c' to capture the bottle image for classification, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Display the live camera feed
        cv2.imshow("Camera Feed", frame)

        # Listen for a key press:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Capture the frame when the bottle stops in front of the camera
            resized_frame = cv2.resize(frame, (img_width, img_height))
            normalized_frame = resized_frame / 255.0
            img_array = np.expand_dims(normalized_frame, axis=0)

            # Make prediction using the trained model
            prediction = model.predict(img_array)
            class_idx = np.argmax(prediction)
            class_label = labels[class_idx]
            confidence = np.max(prediction)

            print(f"Prediction: {class_label} ({confidence * 100:.1f}%)")

            # Define the command to send:
            # If the bottle is "Original" (i.e., non-defective), let it pass ('N')
            # Otherwise, mark as defective and trigger servo action ('D')
            if class_label == "Original":
                ser.write(b'N')
                print("Correct detected -> Command 'N' sent to Arduino.")
            else:
                ser.write(b'D')
                print("Defective detected -> Command 'D' sent to Arduino.")

            # Optionally, display the prediction on the frame
            display_text = f"{class_label} ({confidence * 100:.1f}%)"
            color = (0, 255, 0) if class_label == "Original" else (0, 0, 255)
            cv2.putText(frame, f"Prediction: {display_text}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Camera Feed", frame)
            # Allow time for the operator to see the results, then continue.
            cv2.waitKey(2000)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ser.close()


predict_live_with_trigger()
