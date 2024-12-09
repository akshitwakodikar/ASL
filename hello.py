import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow as tf


# Custom model loader to handle the incompatible 'groups' parameter
def load_compatible_model(model_path):
    try:
        # First attempt: Try loading with custom object scope
        with tf.keras.utils.custom_object_scope({'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D}):
            model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except:
        try:
            # Second attempt: Load and save without problematic configs
            model = tf.keras.models.load_model(model_path, compile=False)
            temp_path = 'temp_model.h5'
            model.save(temp_path, include_optimizer=False)
            return tf.keras.models.load_model(temp_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


# Modified Classifier class to use our custom loader
class ModifiedClassifier(Classifier):
    def __init__(self, model_path, labels_path):
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = load_compatible_model(self.model_path)

        # Read labels
        self.f = open(self.labels_path, 'r')
        self.labels = self.f.read().split('\n')
        self.f.close()


# Main code
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = ModifiedClassifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
folder = "Data/C"
counter = 0
labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        try:
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Drawing
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26),
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        except Exception as e:
            print(f"Error processing hand: {e}")
            continue

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    # Add a quit condition
    if key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()