import cv2
import numpy as np

img = cv2.imread(r"dataset\HTP\01\00\01_00_01_0014.jpg")

# 三个通道都 > 200 的像素，直接设为纯白
mask = np.all(img > 220, axis=2)
img[mask] = [255, 255, 255]

cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", img)

cv2.waitKey(0)
cv2.destroyAllWindows()