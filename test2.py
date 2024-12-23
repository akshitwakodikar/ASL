import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]  

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from the webcam. Exiting...")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure cropping coordinates are within image dimensions
        y_start = max(0, y - offset)
        y_end = min(img.shape[0], y + h + offset)
        x_start = max(0, x - offset)
        x_end = min(img.shape[1], x + w + offset)

        imgCrop = img[y_start:y_end, x_start:x_end]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCropShape = imgCrop.shape

        # Check if the cropped image has a valid shape
        if imgCropShape[0] > 0 and imgCropShape[1] > 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Get predictions
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(f"Prediction: {prediction}, Index: {index}")

            # Ensure index is within the range of labels
            if index < len(labels):
                label = labels[index]
            else:
                label = "Unknown"
                print(f"Warning: Index {index} out of range for labels.")

            # Draw bounding boxes and prediction text
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, label, (x, y - 26),
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

            # Show intermediate images
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        else:
            print("Invalid cropped image. Skipping this frame.")

    # Show the output
    cv2.imshow("Image", imgOutput)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
