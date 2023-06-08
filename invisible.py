from asyncore import read
import cv2
import time
import numpy as np
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))
cap = cv2.VideoCapture(0)
time.sleep(2)
bg = 0
for i in range(60):
    ret, bg = cap.read()
bg = np.flip(bg, axis=1)
while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    img = np.flip(img, axis = 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lb = np.array([0, 120, 50])
    ub = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lb, ub)
    lb = np.array([104, 153, 70])
    ub = np.array([30, 30, 0])
    mask2 = cv2.inRange(hsv, lb, ub)
    mask1 = mask1+mask2
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))
    mask2 = cv2.bitwise_not(mask1)
    result1 = cv2.bitwise_and(img, img, mask = mask2)
    result2 = cv2.bitwise_and(bg, bg, mask = mask1)
    finalResult = cv2.addWeighted(result1, 1, result2, 1, 0)
    output.write(finalResult)
    cv2.imshow("Video", finalResult)
    cv2.waitKey(3)
cap.release()
output.release()
cv2.destroyAllWindows()