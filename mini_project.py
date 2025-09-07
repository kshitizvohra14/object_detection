import cv2
import mediapipe as mp
import time
import posemodule as pm
import numpy as np

# Initialize video capture and pose detector
cap = cv2.VideoCapture(0)
detector = pm.PoseModule()
count_left = 0
count_right = 0
dir_left = 0
dir_right = 0
pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 720))
    img = detector.findPose(img)
    lmList = detector.getPosition(img, draw=False)

    if len(lmList) != 0:
        # LEFT ARM: shoulder (11), elbow (13), wrist (15)
        angle_left = detector.findAngle(img, 11, 13, 15)
        per_left = np.interp(angle_left, (40, 160), (100, 0))
        bar_left = np.interp(angle_left, (40, 160), (650, 100))

        if per_left >= 100:
            if dir_left == 0:
                count_left += 0.5
                dir_left = 1
        if per_left <= 0:
            if dir_left == 1:
                count_left += 0.5
                dir_left = 0

        # RIGHT ARM: shoulder (12), elbow (14), wrist (16)
        angle_right = detector.findAngle(img, 12, 14, 16)
        per_right = np.interp(angle_right, (40, 160), (100, 0))
        bar_right = np.interp(angle_right, (40, 160), (650, 100))

        if per_right >= 100:
            if dir_right == 0:
                count_right += 0.5
                dir_right = 1
        if per_right <= 0:
            if dir_right == 1:
                count_right += 0.5
                dir_right = 0

        # LEFT arm progress bar
        cv2.rectangle(img, (1100, 100), (1175, 600), (0, 255, 0), 3)
        cv2.rectangle(img, (1100, int(bar_left)), (1175, 600), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per_left)}%', (1100, 75),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # RIGHT arm progress bar
        cv2.rectangle(img, (1000, 100), (1075, 600), (255, 0, 0), 3)
        cv2.rectangle(img, (1000, int(bar_right)), (1075, 600), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per_right)}%', (1000, 75),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Display rep counts
        cv2.putText(img, f'Left Reps: {int(count_left)}', (50, 100),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.putText(img, f'Right Reps: {int(count_right)}', (50, 150),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

    cv2.imshow("AI Trainer - Dual Arm", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()