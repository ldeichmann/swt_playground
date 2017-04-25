import cv2
import imutils

img = cv2.imread('/home/cru/Pictures/IMG_20170408_142603.jpg',0)
resized = imutils.resize(img, width=300)
img_filt = cv2.medianBlur(resized, 5)
img_th = cv2.adaptiveThreshold(img_filt,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
contours, hierarchy, _ = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow("Over the Clouds", contours)
# cv2.imshow("Over the Clouds - gray", hierarchy)

while cv2.waitKey(0) != ord("q"):
    pass
cv2.destroyAllWindows()