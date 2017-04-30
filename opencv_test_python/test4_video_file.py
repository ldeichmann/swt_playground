import cv2
import sys
import numpy as np
from math import sqrt, pow, ceil

# video_path = "http://192.168.0.101:4747/mjpegfeed"

def nothing(_):
    pass

# Check if filename is passed
if len(sys.argv) <= 1:
    print("Usage: python hsvThresholder.py <ImageFilePath>")
    exit()

# Create a black image, a window
cap = cv2.VideoCapture(sys.argv[1])
_, img = cap.read()
cv2.namedWindow('image')
cv2.namedWindow('hsv_image')
cv2.namedWindow('controls')

# create trackbars for color change
cv2.createTrackbar('HMin', 'controls', 0, 179, nothing)  # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin', 'controls', 0, 255, nothing)
cv2.createTrackbar('VMin', 'controls', 0, 255, nothing)
cv2.createTrackbar('HMax', 'controls', 0, 179, nothing)
cv2.createTrackbar('SMax', 'controls', 0, 255, nothing)
cv2.createTrackbar('VMax', 'controls', 0, 255, nothing)
cv2.createTrackbar('Border_Width', 'controls', 0, 3, nothing)
cv2.createTrackbar('NoiseFilter', 'controls', 0, 50, nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'controls', 179)
cv2.setTrackbarPos('SMax', 'controls', 255)
cv2.setTrackbarPos('VMax', 'controls', 255)

# Set default value for MIN.
cv2.setTrackbarPos('HMin', 'controls', 0)
cv2.setTrackbarPos('SMin', 'controls', 0)
cv2.setTrackbarPos('VMin', 'controls', 32)
cv2.setTrackbarPos('NoiseFilter', 'controls', 10)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# Output Image to display
while 1:
    ret, img = cap.read()
    if img is None:
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
        ret, img = cap.read()
    output = img
    temp = img.copy()

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'controls')
    sMin = cv2.getTrackbarPos('SMin', 'controls')
    vMin = cv2.getTrackbarPos('VMin', 'controls')

    hMax = cv2.getTrackbarPos('HMax', 'controls')
    sMax = cv2.getTrackbarPos('SMax', 'controls')
    vMax = cv2.getTrackbarPos('VMax', 'controls')

    min_size = cv2.getTrackbarPos('NoiseFilter', 'controls')
    bWidth = cv2.getTrackbarPos('Border_Width', 'controls')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

    h, s, v = cv2.split(output)
    im2, contours, hierarchy = cv2.findContours(v, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # find all pollers and blocks and collect them in these lists
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        size = rect[1]  # size
        # WARNING
        # arbitrary minimal size to remove noise
        if size[0] > min_size and size[1] > min_size:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            im = cv2.drawContours(temp, [box], 0, (0, 0, 255), bWidth)
    # cv2.drawContours(temp, contours, -1, (0, 255, 0), bWidth)


    # Print if there is a change in HSV value
    if (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin, sMin, vMin, hMax, sMax,
                                                                                          vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display output image
    cv2.imshow('image', temp)
    cv2.imshow('hsv_image', output)

    WAIT = 200
    # Wait for 33 milliseconds: 30FPS
    k = cv2.waitKey(WAIT) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
