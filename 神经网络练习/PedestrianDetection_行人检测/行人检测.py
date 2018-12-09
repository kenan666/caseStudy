#行人检测  简单
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np

img=cv2.imread("1.jpg")
orig = img.copy()
# 定义HOG对象，采用默认参数，或者按照下面的格式自己设置
defaultHog=cv2.HOGDescriptor()

# 设置SVM分类器，用默认分类器
defaultHog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 对图像进行多尺度行人检测，返回结果为矩形框
# people=defaultHog.detectMultiScale(img, 0,(8,8),(32,32),1.05,2)

# detect people in the image
(rects, weights) = defaultHog.detectMultiScale(img, winStride=(4, 4),
     padding=(8, 8), scale=1.05)

# 画长方形，框出行人
# for i in range(len(people)):
#     r=people[0][i]
#     cv2.rectangle(img,(r[0],r[1]),(r[2],r[3]),(0,255,0),2,cv2.LINE_AA)

for (x, y, w, h) in rects:
    cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)

cv2.imshow("Before NMS", orig)
cv2.imshow("After NMS", img)
cv2.waitKey(0)