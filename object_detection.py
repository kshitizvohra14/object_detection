import cv2
import numpy as np

# Load pre-trained model (MobileNet SSD)
net = cv2.dnn.readNetFromCaffe(
    "D:\AI_ML_DS\opencv\MobileNetSSD_deploy.prototxt",
    "D:\AI_ML_DS\opencv\MobileNetSSD_deploy.caffemodel"
)

# Class labels MobileNet SSD can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Start video capture
cap = cv2.VideoCapture(0)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
 



while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Convert frame to blob for DNN input
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter weak detections
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = f"{CLASSES[idx]}: {confidence*100:.2f}%"
            color = COLORS[idx]
            # Draw bounding box
            label = f"{CLASSES[idx]}: {confidence*100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the frame
    cv2.imshow("Object Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
