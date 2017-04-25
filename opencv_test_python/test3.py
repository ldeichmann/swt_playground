import cv2
import sys
import numpy as np
from math import sqrt, pow, ceil, floor


def nothing(_):
    pass


# measurements in mm

CONE_LENGTH = 17  # TODO
CONE_WIDTH = 17  # TODO

JENGA_LENGTH = 75
JENGA_WIDTH = 24

CAR_LENGTH = 70  # TODO
CAR_WIDTH = 50  # TODO

CAR_TURN_RADIUS = 52  # TODO: at medium speed?


class Rectangle:

    RECT_FREE = 0
    RECT_OCCUPIED = 1
    RECT_WAYPOINT = 2

    def __init__(self, top_left_x, top_left_y, width, height, rotation=0, occ=RECT_FREE, contour=None):
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.height = height
        self.width = width
        # print("new rect, size {},{}".format(self.width, self.height))
        self.occupied = occ
        self.contour = contour
        self.rotation = rotation

    def set_occupied(self, occ):
        self.occupied = occ

    def get_center(self):
        x = self.top_left_x + (self.width / 2)
        y = self.top_left_y + (self.height / 2)
        return x, y

    def get_center_int(self):
        x = ceil(self.top_left_x + (self.width / 2))
        y = ceil(self.top_left_y + (self.height / 2))
        return x, y

    def distance(self, rectangle):
        """
        Distance from center to center
        :param rectangle: 
        :return: 
        """
        x1, y1 = self.get_center()
        x2, y2 = rectangle.get_center()

        a = x1 - x2
        b = y1 - y2

        return sqrt(pow(a, 2) + pow(b, 2))

    def cv_intersects(self, rect):
        c1 = self.get_center()
        r1 = ((c1[0], c1[1]), (self.width, self.height), self.rotation)
        c2 = rect.get_center()
        r2 = ((c2[0], c2[1]), (rect.width, rect.height), rect.rotation)
        return cv2.rotatedRectangleIntersection(r1, r2)

    def intersects(self, rect):
        return self.cv_intersects(rect)
        # return ((self.top_left_x <= rect.top_left_x + rect.width) and
        #         (self.top_left_x + self.width) >= rect.top_left_x and
        #         self.top_left_y <= (rect.top_left_y + rect.height) and
        #         (self.top_left_y + self.height) >= rect.height)

    def passable(self):
        return self.occupied == Rectangle.RECT_FREE or self.occupied == Rectangle.RECT

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

    def __init__(self, top_left_x, top_left_y, width, height, grid_size=7):
        """
        assuming top_left is 0, because opencv does it this way
        
        :param top_left_x: image coordinate
        :param top_left_y: image coordinate
        :param width: image width
        :param height: image height
        :param grid_size: 
        """

        # root
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y

        self.width = width
        self.height = height

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
                new_rect = Rectangle(new_rect_top_left_x, new_rect_top_left_y, grid_size, grid_size)
                self._grid[i].insert(j, new_rect)


    def add_obstacle(self, rect, type=Rectangle.RECT_OCCUPIED, max_failed_steps=15):
        """
        This is currently assuming something like the bounding rect is used, so the obstacle is grid aligned and not
        rotated
        :param rect: 
        :return: 
        """

        _column, _row = self.get_index_from_position(rect.top_left_x, rect.top_left_y)
        starting_row = _row - 5
        starting_col = _column - 5
        if starting_row < 0:
            starting_row = 0
        if starting_col < 0:
            starting_col = 0
        slot = None
        check_rows = int(_row + ceil(rect.height / 7)) + 3
        check_columns = int(_column + ceil(rect.width / 7)) + 3
        found = False
        row = starting_row

        rows_started = False
        row_was_empty = False

        while ((not row_was_empty and rows_started) or not rows_started) and row < len(self._grid):
            col = starting_col
            row_started = False
            row_ended = False
            while ((not row_started and not row_ended and (col - starting_col) < max_failed_steps) or
                   (row_started and not row_ended)) and col < len(self._grid[row]):
                slot = self._grid[row][col]
                intersects = slot.cv_intersects(rect)[0] > 0
                print("Checking {} {}".format(row, col))
                if intersects:
                    slot.set_occupied(type)
                    print("Occupying {} {}".format(row, col))
                    row_started = True
                    rows_started = True
                    row_was_empty = False
                elif not intersects and row_started:
                    row_ended = True
                col += 1
            if not row_started:
                row_was_empty = True
            row += 1
            print("{} {} {} {} {}".format(row, col, row_was_empty, rows_started, row < len(self._grid)))

    def add_cone(self, rect):
        self.add_obstacle(rect, Rectangle.RECT_WAYPOINT)

    def get_index_from_position(self, pos_x, pos_y):
        """
        Return indexes for given position
        :param pos_x: 
        :param pos_y: 
        :return: 
        """
        row = int((pos_x - (self.top_left_x - self.offset)) / self.grid_size)
        column = int((pos_y - (self.top_left_y - self.offset)) / self.grid_size)
        return row, column

    def draw_grid(self, canvas_image):
        for column in self._grid:
            for slot in column:
                if slot.occupied == Rectangle.RECT_WAYPOINT:
                    cv2.rectangle(canvas_image, (slot.top_left_x, slot.top_left_y), (slot.top_left_x + slot.width,
                                                                                     slot.top_left_y + slot.height),
                                  (0, 0, 0), -1)
                elif slot.occupied:
                    cv2.rectangle(canvas_image, (slot.top_left_x, slot.top_left_y), (slot.top_left_x + slot.width,
                                                                                     slot.top_left_y + slot.height),
                                  (0, 255, 255), -1)
                else:
                    cv2.rectangle(canvas_image, (slot.top_left_x, slot.top_left_y), (slot.top_left_x + slot.width,
                                                                                     slot.top_left_y + slot.height),
                                  (255, 255, 255), bWidth)

# Check if filename is passed
if len(sys.argv) <= 1:
    print("Usage: python hsvThresholder.py <ImageFilePath>")
    exit()

# Create a black image, a window
img = cv2.imread(sys.argv[1])


def create_windows():
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
    cv2.createTrackbar('Border_Mode', 'controls', 0, 1, nothing)  # default, bounding
    cv2.createTrackbar('DistMax', 'controls', 1, 20, nothing)
    cv2.createTrackbar('DistMin', 'controls', 1, 20, nothing)
    cv2.createTrackbar('DrawGrid', 'controls', 0, 1, nothing)
    cv2.createTrackbar('NoiseFilter', 'controls', 0, 50, nothing)

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

create_windows()
# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0


def detect_cones_and_obstacles(contours):
    min_size = cv2.getTrackbarPos('NoiseFilter', 'controls')
    cones = []
    obstacles = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        size = rect[1]  # size
        # arbitrary minimal size to remove noise
        if size[0] > min_size and size[1] > min_size:
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # if height and width are about the same, it's likely a cone
            top_left_x = rect[0][0] - (rect[1][0] / 2)
            top_left_y = rect[0][1] - (rect[1][1] / 2)
            rectangle = Rectangle(top_left_x, top_left_y, rect[1][0], rect[1][1], rotation=rect[2], contour=cnt)
            if size[0] - 5 < size[1] < size[0] + 5:
                cones.append(rectangle)
            else:
                # otherwise it's just an obstacle
                obstacles.append(rectangle)
    return cones, obstacles

# check if distance makes cones a valid gate
def is_valid_gate_distance(cone_1, cone_2, min, max):
    """
    Expects rects
    :param MIN: 
    :param MAX: 
    :param cone_1: 
    :param cone_2: 
    :return: 
    """
    dist = cone_1.distance(cone_2)
    return min <= dist <= max

def get_cone_pairs(cones, min, max):
    used_cones = []
    pairs = []
    for cone in cones:
        if cone not in used_cones:
            best_cone = None
            best_dist = None
            for second_cone in cones:
                # if they are a valid pair or a better pair than the previously found pair
                if cone is not second_cone and second_cone not in used_cones:
                    if (best_cone is None and is_valid_gate_distance(cone, second_cone, min, max)) or\
                            (best_cone is not None and cone.distance(second_cone) < best_dist):
                        best_cone = second_cone
                        best_dist = cone.distance(second_cone)
            if best_cone is not None:
                # we have to draw a line here!
                used_cones.append(cone)
                used_cones.append(best_cone)
                pairs.append((cone, best_cone))
    return pairs

# Output Image to display
output = img
while 1:
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
    border_mode = cv2.getTrackbarPos('Border_Mode', 'controls')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

    dil_kernel = np.ones((dilSize, dilSize), np.uint8)
    ero_kernel = np.ones((eroSize, eroSize), np.uint8)

    if eroSize > 0:
        output = cv2.erode(output, dil_kernel, iterations=1)
    if dilSize > 0:
        output = cv2.dilate(output, ero_kernel, iterations=1)

    h, s, v = cv2.split(output)
    im2, contours, hierarchy = cv2.findContours(v, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    cones, obstacles = detect_cones_and_obstacles(contours)

    # calculate average poller size
    average_cone_size = 0
    if len(cones) > 0:
        for cone in cones:
            cone_rect = cv2.minAreaRect(cone.contour)
            average_cone_size += (cone.width + cone.height) / 2
        average_cone_size = average_cone_size / len(cones)

    MAX_POLLER_DIST = dist_max*average_cone_size
    MIN_POLLER_DIST = dist_min*average_cone_size


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
        boundingBoxes = [cv2.boundingRect(c.contour) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        return cnts, boundingBoxes



    # compare every unused cone with every other unused cone for gates
    pairs = get_cone_pairs(cones, MIN_POLLER_DIST, MAX_POLLER_DIST)

    def draw_border(rectangle, color):
        x, y = rectangle.get_center_int()
        box = cv2.boxPoints(((x, y), (rectangle.width, rectangle.height), rectangle.rotation))
        box = np.int0(box)
        im = cv2.drawContours(temp, [box], 0, color, bWidth)

    # draw all the things!
    for cone in cones:
        draw_border(cone, (0, 255, 0))

    for obstacle in obstacles:
        draw_border(obstacle, (0, 0, 255))

    for pair in pairs:
        draw_border(pair[0], (255, 255, 0))
        draw_border(pair[1], (255, 255, 0))
        cv2.line(temp, pair[0].get_center_int(), pair[1].get_center_int(), (255, 255, 0), bWidth)

    if len(cones) > 0 or len(obstacles) > 0:
        objects = cones + obstacles
        contours, boxes = sort_contours(cones)
        box_left = boxes[0]
        box_right = boxes[len(boxes)-1]
        contours, boxes = sort_contours(objects, "top-to-bottom")
        box_top = boxes[0]
        box_bottom = boxes[len(boxes)-1]

        left = box_left[0]
        right = box_right[0] + box_right[2]
        top = box_top[1]
        bottom = box_bottom[1] + box_bottom[3]

        if draw_grid:

            grid = Grid(left, top, right-left, bottom-top, int(average_cone_size / 2))

            for cone in cones:
                grid.add_cone(cone)

            for obstacle in obstacles:
                grid.add_obstacle(obstacle)
            # for poller in poller_contours:
            #     bound = cv2.boundingRect(poller)
            #     new_rect = Rectangle(bound[0], bound[1], bound[2], bound[3])
            #     grid.add_obstacle(new_rect)

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
