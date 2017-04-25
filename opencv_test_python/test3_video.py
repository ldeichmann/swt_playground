import cv2
import sys
import numpy as np
from math import sqrt, pow, ceil

video_path = "http://192.168.0.101:4747/mjpegfeed"

def nothing(_):
    pass


class Rectangle:

    def __init__(self, top_left_x, top_left_y, bottom_right_x, bottom_right_y, occ=False):
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y
        self.height = self.bottom_right_y - self.top_left_y
        self.width = self.bottom_right_x - self.top_left_x
        # print("new rect, size {},{}".format(self.width, self.height))
        self.occupied = occ

    def set_occupied(self, occ):
        self.occupied = occ

    def intersects(self, rect):
        return ((self.top_left_x <= rect.top_left_x + rect.width) and
                (self.top_left_x + self.width) >= rect.top_left_x and
                self.top_left_y <= (rect.top_left_y + rect.height) and
                (self.top_left_y + self.height) >= rect.height)

    def union(self, b):
        tx = min(self.top_left_x, b.top_left_x)
        ty = min(self.top_left_y, b.top_left_y)
        w = max(self.top_left_x + self.width, b.top_left_x + b.width) - tx
        h = max(self.top_left_y + self.height, b.top_left_y + b.height) - ty
        bx = tx + w
        by = ty + h
        return tx, ty, bx, by

    def intersection(self, b):
        tx = min(self.top_left_x, b.top_left_x)
        ty = min(self.top_left_y, b.top_left_y)
        w = max(self.top_left_x + self.width, b.top_left_x + b.width) - tx
        h = max(self.top_left_y + self.height, b.top_left_y + b.height) - ty
        bx = tx + w
        by = ty + h
        if w < 0 or h < 0:
            return None  # or (0,0,0,0) ?
        return tx, ty, bx, by

    def __str__(self):
        return "x: {} - y: {} - w: {} - h: {}".format(self.top_left_x, self.top_left_y, self.width, self.height)


class Grid:

    def __init__(self, top_left_x, top_left_y, bottom_right_x, bottom_right_y, grid_size=7):
        # root
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        # max value
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y

        # assuming top_left is 0
        self.width = bottom_right_x - top_left_x
        self.height = bottom_right_y - top_left_y

        self.grid_size = grid_size

        extra_space = 2  # add number of fields on every side
        rows = int(abs(self.width / grid_size) + extra_space * 2)
        columns = int(abs(self.height / grid_size) + extra_space * 2)
        self.offset = int(extra_space * grid_size)
        # print("TOP_LEFT_X: {} - OFFSET: {}".format(self.top_left_x, self.offset))
        self._initialize_grid(columns, rows, grid_size)

    def _initialize_grid(self, columns, rows, grid_size):
        self._grid = []
        for i in range(columns):
            self._grid.insert(i, [])
            for j in range(rows):
                new_rect_top_left_x = (self.top_left_x - self.offset) + (j * grid_size)
                new_rect_top_left_y = (self.top_left_y - self.offset) + (i * grid_size)
                new_rect = Rectangle(new_rect_top_left_x, new_rect_top_left_y, new_rect_top_left_x + grid_size,
                                     new_rect_top_left_y + grid_size)
                self._grid[i].insert(j, new_rect)

    def add_obstacle(self, rect, poller=True):
        idx1, idx2 = self.get_index_from_position(rect.top_left_x, rect.top_left_y)
        slot = None
        check_rows = int(idx1 + ceil(rect.height / 7)) + 1
        check_columns = int(idx2 + ceil(rect.width / 7)) + 1
        for i in range(idx2, check_columns):
            for j in range(idx1, check_rows):
                if i < len(self._grid):
                    row = self._grid[i]
                    if j < len(row):
                        slot = row[j]
                        if slot.intersects(rect):
                            if poller:
                                slot.set_occupied(1)
                            else:
                                slot.set_occupied(2)
                        else:
                            print("Non Intersection detected.")

    def get_index_from_position(self, pos_x, pos_y):
        x = int((pos_x - (self.top_left_x - self.offset)) / self.grid_size)
        y = int((pos_y - (self.top_left_y - self.offset)) / self.grid_size)
        return x, y

    def print_grid(self):
        for column in self._grid:
            for slot in column:
                if slot.occupied:
                    print("X", end='')
                else:
                    print("O", end='')
            print("")

    def draw_grid(self, img):
        for column in self._grid:
            for slot in column:
                if slot.occupied == 1:
                    cv2.rectangle(img, (slot.top_left_x, slot.top_left_y), (slot.bottom_right_x, slot.bottom_right_y),
                                  (0, 255, 255), -1)
                elif slot.occupied == 2:
                    cv2.rectangle(img, (slot.top_left_x, slot.top_left_y), (slot.bottom_right_x, slot.bottom_right_y),
                                  (0, 0, 255), -1)
                else:
                    cv2.rectangle(img, (slot.top_left_x, slot.top_left_y), (slot.bottom_right_x, slot.bottom_right_y),
                                  (255, 255, 255), bWidth)

# Create a black image, a window
cap = cv2.VideoCapture(video_path)
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
cv2.createTrackbar('Erode', 'controls', 0, 24, nothing)
cv2.createTrackbar('Dilate', 'controls', 0, 24, nothing)
cv2.createTrackbar('Border_Width', 'controls', 0, 3, nothing)
cv2.createTrackbar('DistMax', 'controls', 1, 20, nothing)
cv2.createTrackbar('DistMin', 'controls', 1, 20, nothing)
cv2.createTrackbar('DrawGrid', 'controls', 0, 1, nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'controls', 179)
cv2.setTrackbarPos('SMax', 'controls', 255)
cv2.setTrackbarPos('VMax', 'controls', 255)
cv2.setTrackbarPos('DistMax', 'controls', 7)
cv2.setTrackbarPos('DistMin', 'controls', 5)

# Set default value for MIN.
cv2.setTrackbarPos('HMin', 'controls', 0)
cv2.setTrackbarPos('SMin', 'controls', 81)
cv2.setTrackbarPos('VMin', 'controls', 95)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# Output Image to display
while 1:
    _, img = cap.read()
    output = img
    temp = img.copy()

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'controls')
    sMin = cv2.getTrackbarPos('SMin', 'controls')
    vMin = cv2.getTrackbarPos('VMin', 'controls')

    hMax = cv2.getTrackbarPos('HMax', 'controls')
    sMax = cv2.getTrackbarPos('SMax', 'controls')
    vMax = cv2.getTrackbarPos('VMax', 'controls')

    dilSize = cv2.getTrackbarPos('Dilate', 'controls')
    eroSize = cv2.getTrackbarPos('Erode', 'controls')
    bWidth = cv2.getTrackbarPos('Border_Width', 'controls')
    dist_max = cv2.getTrackbarPos('DistMax', 'controls')
    dist_min = cv2.getTrackbarPos('DistMin', 'controls')
    draw_grid = cv2.getTrackbarPos('DrawGrid', 'controls') == 1

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

    dil_kernel = np.ones((dilSize, dilSize), np.uint8)
    ero_kernel = np.ones((eroSize, eroSize), np.uint8)

    output = cv2.erode(output, dil_kernel, iterations=1)
    output = cv2.dilate(output, ero_kernel, iterations=1)

    h, s, v = cv2.split(output)
    im2, contours, hierarchy = cv2.findContours(v, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # find all pollers and blocks and collect them in these lists
    pollers = []
    poller_contours = []
    used_pollers = []
    blockse = []
    blockse_contours = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        size = rect[1]  # size
        # WARNING
        # arbitrary minimal size to remove noise
        if size[0] > 10 and size[1] > 10:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            if size[0] - 5 < size[1] < size[0] + 5:
                pollers.append(rect)
                poller_contours.append(cnt)
                # im = cv2.drawContours(temp, [box], 0, (255, 0, 0), bWidth)
            else:
                # draw blocks instantly, we don't use them now
                blockse.append(rect)
                blockse_contours.append(cnt)
                im = cv2.drawContours(temp, [box], 0, (0, 0, 255), bWidth)
    # cv2.drawContours(temp, contours, -1, (0, 255, 0), bWidth)

    # calculate the distance between two pollers
    def poller_dist(poller1, poller2):
        x_dist = poller1[0][0] - poller2[0][0]
        y_dist = poller1[0][1] - poller2[0][1]
        dist = sqrt(pow(x_dist, 2) + pow(y_dist, 2))
        # print("poller1: {}\npoller2: {}\ndist: {}".format(poller1[0], poller2[0], dist))
        # print("x_dist: {}\ny_dist: {}".format(x_dist, y_dist))
        return dist

    # calculate average poller size
    avg_poller_size = 0
    if len(pollers) > 0:
        for poller in pollers:
            avg_poller_size += (poller[1][0] + poller[1][1]) / 2
        avg_poller_size = avg_poller_size / len(pollers)
    else:
        avg_poller_size = 15

    # print("avg_poller_size {}".format(avg_poller_size))

    MAX_POLLER_DIST = dist_max*avg_poller_size
    MIN_POLLER_DIST = dist_min*avg_poller_size


    # check if distance makes pollers a valid pair
    def valid_dist(poller1, poller2):
        dist = poller_dist(poller1, poller2)
        return MIN_POLLER_DIST <= dist <= MAX_POLLER_DIST


    def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        return cnts, boundingBoxes


    # compare every unused poller with every other unused poller for pairing
    for poller in pollers:
        box = cv2.boxPoints(poller)
        box = np.int0(box)
        im = cv2.drawContours(temp, [box], 0, (255, 0, 0), bWidth)
        # if not is_arr_in_list(poller, used_pollers):
        if poller not in used_pollers:
            best_poller = None
            best_dist = None
            for second_poller in pollers:
                # if they are a valid pair or a better pair than the previously found pair
                if poller is not second_poller and second_poller not in used_pollers:
                    if (best_poller is None and valid_dist(poller, second_poller)) or\
                            (best_poller is not None and poller_dist(poller, second_poller) < best_dist):
                        best_poller = second_poller
                        best_dist = poller_dist(poller, second_poller)
            if best_poller is not None:
                # print("line from {} to {} with length {}".format(poller[0], best_poller[0], best_dist))
                start_x = int(poller[0][0])
                start_y = int(poller[0][1])
                finish_x = int(best_poller[0][0])
                finish_y = int(best_poller[0][1])
                # we have to draw a line here!
                cv2.line(temp, (start_x, start_y), (finish_x, finish_y), (255, 0, 0), bWidth)
                used_pollers.append(poller)
                used_pollers.append(best_poller)
            else:
                im = cv2.drawContours(temp, [box], 0, (0, 255, 255), bWidth)

    if len(poller_contours) > 0 or len(blockse_contours) > 0:
        cnts = poller_contours + blockse_contours
        contours, boxes = sort_contours(cnts)
        box_left = boxes[0]
        box_right = boxes[len(boxes)-1]
        contours, boxes = sort_contours(cnts, "top-to-bottom")
        box_top = boxes[0]
        box_bottom = boxes[len(boxes)-1]

        left = box_left[0]
        right = box_right[0] + box_right[2]
        top = box_top[1]
        bottom = box_bottom[1] + box_bottom[3]

        if draw_grid:

            grid = Grid(left, top, right, bottom, int(avg_poller_size / 2))

            for poller in poller_contours:
                bound = cv2.boundingRect(poller)
                new_rect = Rectangle(bound[0], bound[1], bound[0] + bound[2], bound[1] + bound[3])
                grid.add_obstacle(new_rect)

            for poller in blockse_contours:
                bound = cv2.boundingRect(poller)
                new_rect = Rectangle(bound[0], bound[1], bound[0] + bound[2], bound[1] + bound[3])
                grid.add_obstacle(new_rect, poller=False)

            # grid.print_grid()
            grid.draw_grid(temp)

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
