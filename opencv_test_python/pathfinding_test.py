import cv2
import numpy as np
import logging
import sys
import heapq
import copy
from math import sqrt

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class Rectangle:

    def __init__(self, center_x, center_y, width, height, rotation=0, contour=None):
        self.x = center_x
        self.y = center_y
        self.height = height
        self.width = width
        self.contour = contour
        self.rotation = rotation

    def distance(self, rectangle):
        """
        Distance from center to center
        :param rectangle: 
        :return: 
        """
        a = self.x - rectangle.x
        b = self.y - rectangle.y

        return sqrt(pow(a, 2) + pow(b, 2))

    def intersects(self, rect):
        r1 = ((self.x, self.y), (self.width, self.height), self.rotation)
        r2 = ((rect.x, rect.y), (rect.width, rect.height), rect.rotation)
        return cv2.rotatedRectangleIntersection(r1, r2)[0] > 0

    def __str__(self):
        return "center_x: {} - center_y: {} - w: {} - h: {}".format(self.x, self.y, self.width, self.height)


class GridRectangle(Rectangle):

    FREE = 0
    CONE = 10
    OBSTACLE = 20
    AVOID = 30
    WAYPOINT = 40

    N = 200
    NE = 210
    E = 220
    SE = 230
    S = 240
    SW = 250
    W = 260
    NW = 270

    def __init__(self, center_x, center_y, width, height, x_coordinate, y_coordinate, occupied=FREE):
        super().__init__(center_x, center_y, width, height)
        self.coordinates = (x_coordinate, y_coordinate)
        self.occupied = occupied
        self.direction = None

    def passable(self):
        return self.occupied == GridRectangle.FREE or self.occupied == GridRectangle.WAYPOINT

    def set_occupied(self, new_occupied):
        self.occupied = new_occupied

    def __gt__(self, rect2):
        return self.coordinates[0] > rect2.coordinates[0] and self.coordinates[1] > rect2.coordinates[1]

    def same(self, rect2):
        if self.direction is not None and rect2.direction is not None:
            return self.coordinates == rect2.coordinates and self.direction == rect2.direction
        else:
            return self.coordinates == rect2.coordinates

    def __str__(self):
        return "coordinates: {} - occupied: {}".format(self.coordinates, self.occupied)


class Grid:
    """
    Grid for use on OpenCV Images
    """

    def __init__(self, top_left_x, top_left_y, width, height, grid_size=7, spacing=2):
        """
        assuming top_left is 0, because opencv does it this way

        :param top_left_x: image coordinate
        :param top_left_y: image coordinate
        :param width: image width
        :param height: image height
        :param grid_size: 
        :param spacing: extra grid elements around the image grid
        """

        self.x = top_left_x
        self.y = top_left_y
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.spacing = spacing

        # calculate grid
        self.rows = int(abs(self.width / grid_size) + spacing * 2)
        self.columns = int(abs(self.height / grid_size) + spacing * 2)
        self.offset = int(spacing * grid_size)

        self.grid = []
        for i in range(self.columns):
            self.grid.insert(i, [])
            for j in range(self.rows):
                # calculate center variables please
                new_rect_center_x = (self.x - self.offset) + (j * grid_size) + (grid_size / 2)
                new_rect_center_y = (self.y - self.offset) + (i * grid_size) + (grid_size / 2)
                new_rect = GridRectangle(new_rect_center_x, new_rect_center_y, grid_size, grid_size, j, i)
                self.grid[i].insert(j, new_rect)

    def add_obstacle(self, obstacle, obstacle_type=GridRectangle.OBSTACLE, max_failed_steps=15):
        estimated_col, estimated_row = self.get_index_from_position(obstacle.x, obstacle.y)
        starting_row = estimated_row - int(max_failed_steps / 2)
        starting_col = estimated_col - int(max_failed_steps / 2)
        if starting_row < 0:
            starting_row = 0
        if starting_col < 0:
            starting_col = 0

        current_row = starting_row

        rows_started = False
        row_was_empty = False

        while ((not row_was_empty and rows_started) or not rows_started) and current_row < self.columns:
            current_col = starting_col
            row_started = False
            row_ended = False
            while ((not row_started and not row_ended and (current_col - starting_col) < max_failed_steps) or
                   (row_started and not row_ended)) and current_col < self.rows:
                slot_r = self.grid[current_row]
                slot = slot_r[current_col]
                intersects = slot.intersects(obstacle)
                log.debug("Checking {} {}".format(current_row, current_col))
                if intersects:
                    slot.set_occupied(obstacle_type)
                    if obstacle_type == GridRectangle.OBSTACLE:
                        for neighbor in self.neighbors(slot, directional=False):
                            if neighbor.occupied == GridRectangle.FREE or \
                                            neighbor.occupied == GridRectangle.AVOID:
                                neighbor.set_occupied(GridRectangle.AVOID)
                            for second_neighbor in self.neighbors(neighbor, directional=False):
                                if second_neighbor.occupied == GridRectangle.FREE or \
                                                second_neighbor.occupied == GridRectangle.AVOID:
                                    second_neighbor.set_occupied(GridRectangle.AVOID)
                                for third_neighbor in self.neighbors(second_neighbor, directional=False):
                                    if third_neighbor.occupied == GridRectangle.FREE or \
                                                    third_neighbor.occupied == GridRectangle.AVOID:
                                        third_neighbor.set_occupied(GridRectangle.AVOID)

                    if obstacle_type == GridRectangle.CONE:
                        for neighbor in self.neighbors(slot, directional=False):
                            if neighbor.occupied == GridRectangle.FREE or \
                                            neighbor.occupied == GridRectangle.AVOID:
                                neighbor.set_occupied(GridRectangle.AVOID)

                    log.debug("Occupying {} {}".format(current_row, current_col))
                    row_started = True
                    rows_started = True
                    row_was_empty = False
                elif not intersects and row_started:
                    row_ended = True
                current_col += 1
            if not row_started:
                row_was_empty = True
            current_row += 1

    def add_cone(self, obstacle):
        self.add_obstacle(obstacle, obstacle_type=GridRectangle.CONE)

    def add_waypoint(self, waypoint):
        self.add_obstacle(waypoint, obstacle_type=GridRectangle.WAYPOINT)

    def get_index_from_position(self, x, y):
        row = int((x - (self.x - self.offset)) / self.grid_size)
        column = int((y - (self.y - self.offset)) / self.grid_size)
        return row, column

    def rect_in_bounds(self, coordinates):
        log.debug("rect_in_bounds {}".format(coordinates))
        x, y, direction = coordinates
        log.debug("{}".format(0 <= y < self.columns and 0 <= x < self.rows))
        return 0 <= y < self.columns and 0 <= x < self.rows

    @staticmethod
    def rect_passable(rect):
        return rect.passable()

    def neighbors(self, rect, directional=True):
        try:
            (x, y) = rect.coordinates
            direction = rect.direction
        except AttributeError:
            y, x = self.get_index_from_position(rect.x, rect.y)
            direction = None
        result_rects = []

        N = (x, y - 1, GridRectangle.N)
        NE = (x + 1, y - 1, GridRectangle.NE)
        E = (x + 1, y, GridRectangle.E)
        SE = (x + 1, y + 1, GridRectangle.SE)
        S = (x, y + 1, GridRectangle.S)
        SW = (x - 1, y + 1, GridRectangle.SW)
        W = (x - 1, y, GridRectangle.W)
        NW = (x - 1, y - 1, GridRectangle.NW)

        if directional:
            if direction == GridRectangle.N:
                results = [NW, N, NE]
            elif direction == GridRectangle.NE:
                results = [N, NE, E]
            elif direction == GridRectangle.E:
                results = [NE, E, SE]
            elif direction == GridRectangle.SE:
                results = [E, SE, S]
            elif direction == GridRectangle.S:
                results = [SE, S, SW]
            elif direction == GridRectangle.SW:
                results = [S, SW, W]
            elif direction == GridRectangle.W:
                results = [SW, W, NW]
            elif direction == GridRectangle.NW:
                results = [W, NW, N]
            else:
                results = [N, NE, E, SE, S, SW, W, NW]
        else:
            results = [N, NE, E, SE, S, SW, W, NW]

        if (x + y) % 2 == 0: results.reverse()  # aesthetics
        results = list(filter(self.rect_in_bounds, results))
        for result in results:
            if directional:
                rect = copy.copy(self.grid[result[1]][result[0]])
                rect.direction = result[2]
                result_rects.append(rect)
            else:
                rect = self.grid[result[1]][result[0]]
                result_rects.append(rect)
        result_rects = list(filter(self.rect_passable, result_rects))
        log.debug("for rect {} - results {} - result_rects {}".format(rect, list(results), result_rects))
        return result_rects

    def cost(self, from_node, to_node):
        if to_node.occupied == GridRectangle.AVOID:
            return 50000
        if from_node.direction != to_node.direction:
            return 100
        return 0


class Pathfinding:

    class PriorityQueue:
        def __init__(self):
            self.elements = []

        def empty(self):
            return len(self.elements) == 0

        def put(self, item, priority):
            heapq.heappush(self.elements, (priority, item))

        def get(self):
            return heapq.heappop(self.elements)[1]

    def __init__(self, grid, waypoints):
        self.grid = grid
        self.waypoints = waypoints
        # TODO: Sort by distance
        # self.waypoints = sorted(waypoints, key=lambda waypoint: waypoint.)

    def heuristic(self, start, goal, node):
        # max_dist = start.distance(goal)
        # # return abs(goal.distance(node) - max_dist)
        # return abs((goal.distance(node)/max_dist)*100)
        return abs((goal.distance(node)))

    def search(self, start, goal):
        log.info("Finding path from {} to {}".format(start, goal))
        frontier = self.PriorityQueue()
        frontier.put(start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0
        current = None

        while not frontier.empty():
            current = frontier.get()

            if current.same(goal):
                log.debug("Found a path, breaking")
                # log.debug("Path is: {}".format(came_from))
                break

            log.debug("Checking neighbours for {}".format(current))
            for next in self.grid.neighbors(current):
                new_cost = cost_so_far[current] + self.grid.cost(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(start, goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current

        return came_from, cost_so_far, current

    def test_path(self):
        if len(self.waypoints) > 1:
            paths = []
            last_finish = None
            for i in range(len(self.waypoints)-1):
                if not last_finish:
                    start_wp = self.waypoints[i]
                    start_x, start_y = self.grid.get_index_from_position(start_wp.x, start_wp.y)
                    start = self.grid.grid[start_y][start_x]
                else:
                    start = last_finish
                goal_wp = self.waypoints[i+1]
                goal_x, goal_y = self.grid.get_index_from_position(goal_wp.x, goal_wp.y)

                finish = self.grid.grid[goal_y][goal_x]

                result, cost, last_finish = self.search(start, finish)
                paths.append((result, last_finish))

            return paths
        return None, None


class CV:

    def __init__(self, image_path):
        self.original_img = cv2.imread(image_path)

        self.hMin = self.sMin = self.vMin = self.hMax = self.sMax = self.vMax = 0
        self.phMin = self.psMin = self.pvMin = self.phMax = self.psMax = self.pvMax = 0
        self.dilSize = self.eroSize = self.bWidth = self.dist_min = self.dist_max = 0
        self.draw_grid = self.border_mode = self.min_size = self.pathfinding = 0
        self.grid = None
        self.initialized = False
        self.create_windows()
        self.start()

    def detect_cones_and_obstacles(self, contours):
        cones = []
        obstacles = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            size = rect[1]  # size
            # arbitrary minimal size to remove noise
            if size[0] > self.min_size and size[1] > self.min_size:
                # box = cv2.boxPoints(rect)
                # box = np.int0(box)
                # if height and width are about the same, it's likely a cone
                top_left_x = rect[0][0]
                top_left_y = rect[0][1]
                rectangle = Rectangle(top_left_x, top_left_y, rect[1][0], rect[1][1], rotation=rect[2], contour=cnt)
                if size[0] - 5 < size[1] < size[0] + 5:
                    cones.append(rectangle)
                else:
                    # otherwise it's just an obstacle
                    obstacles.append(rectangle)
        return cones, obstacles

    # check if distance makes cones a valid gate
    def is_valid_gate_distance(self, cone_1, cone_2, min, max):
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

    def get_cone_pairs(self, cones, min, max):
        used_cones = []
        pairs = []
        for cone in cones:
            if cone not in used_cones:
                best_cone = None
                best_dist = None
                for second_cone in cones:
                    # if they are a valid pair or a better pair than the previously found pair
                    if cone is not second_cone and second_cone not in used_cones:
                        if (best_cone is None and self.is_valid_gate_distance(cone, second_cone, min, max)) or \
                                (best_cone is not None and cone.distance(second_cone) < best_dist):
                            best_cone = second_cone
                            best_dist = cone.distance(second_cone)
                if best_cone is not None:
                    # we have to draw a line here!
                    used_cones.append(cone)
                    used_cones.append(best_cone)
                    pairs.append((cone, best_cone))
        return pairs

    def get_gate_waypoints(self, cones):
        waypoints = []
        for pair in cones:
            vector = (pair[1].x - pair[0].x, pair[1].y - pair[0].y)
            half_vector = (vector[0] / 2, vector[1] / 2)
            x = pair[0].x + half_vector[0]
            y = pair[0].y + half_vector[1]
            waypoint = Rectangle(x-1, y-1, 2, 2)
            waypoints.append(waypoint)
        return waypoints

    def sort_contours(self, cnts, method="left-to-right"):
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

    def update(self, opt=None):
        tmp = self.original_img.copy()
        output = tmp.copy()

        # Set minimum and max HSV values to display
        lower = np.array([self.hMin, self.sMin, self.vMin])
        upper = np.array([self.hMax, self.sMax, self.vMax])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(tmp, tmp, mask=mask)

        if self.eroSize > 0:
            ero_kernel = np.ones((self.eroSize, self.eroSize), np.uint8)
            output = cv2.erode(output, ero_kernel, iterations=1)
        if self.dilSize > 0:
            dil_kernel = np.ones((self.dilSize, self.dilSize), np.uint8)
            output = cv2.dilate(output, dil_kernel, iterations=1)

        h, s, v = cv2.split(output)
        im2, contours, hierarchy = cv2.findContours(v, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        cones, obstacles = self.detect_cones_and_obstacles(contours)

        # calculate average poller size
        average_cone_size = 0
        if len(cones) > 0:
            for cone in cones:
                cone_rect = cv2.minAreaRect(cone.contour)
                average_cone_size += (cone.width + cone.height) / 2
            average_cone_size = average_cone_size / len(cones)

        MAX_POLLER_DIST = self.dist_max*average_cone_size
        MIN_POLLER_DIST = self.dist_min*average_cone_size

        # compare every unused cone with every other unused cone for gates
        pairs = self.get_cone_pairs(cones, MIN_POLLER_DIST, MAX_POLLER_DIST)
        waypoints = self.get_gate_waypoints(pairs)


        # create grid
        if len(cones) > 0 or len(obstacles) > 0:
            objects = cones + obstacles
            contours, boxes = self.sort_contours(cones)
            box_left = boxes[0]
            box_right = boxes[len(boxes) - 1]
            contours, boxes = self.sort_contours(objects, "top-to-bottom")
            box_top = boxes[0]
            box_bottom = boxes[len(boxes) - 1]

            left = box_left[0]
            right = box_right[0] + box_right[2]
            top = box_top[1]
            bottom = box_bottom[1] + box_bottom[3]

            self.grid = Grid(left, top, right-left, bottom-top, int(average_cone_size / 2))
            for cone in cones:
                self.grid.add_cone(cone)

            for obstacle in obstacles:
                self.grid.add_obstacle(obstacle)

            for waypoint in waypoints:
                self.grid.add_waypoint(waypoint)
        else:
            self.grid = None

        # draw all the things!
        for cone in cones:
            self.draw_border(cone, tmp, (0, 255, 0))

        for obstacle in obstacles:
            self.draw_border(obstacle, tmp, (0, 0, 255))

        for pair in pairs:
            self.draw_border(pair[0], tmp, (255, 255, 0))
            self.draw_border(pair[1], tmp, (255, 255, 0))
            cv2.line(tmp, (int(pair[0].x), int(pair[0].y)), (int(pair[1].x), int(pair[1].y)), (255, 255, 0), self.bWidth)

        if self.draw_grid:
            self.draw_cv_grid(tmp)

        for waypoint in waypoints:
            self.draw_border(waypoint, tmp, (200, 0, 0))

        if self.grid and self.pathfinding:
            pf = Pathfinding(self.grid, waypoints)
            paths = pf.test_path()
            if paths:
                for path, finish in paths:
                    rect = path.get(finish)
                    log.debug("Drawing {}".format(rect))
                    while rect:
                        log.debug("Drawing2 {}".format(rect))
                        self.draw_border(rect, tmp, (0,252,124))
                        rect = path.get(rect)

                # for rect in test_path:
                #     self.draw_border(rect, tmp, (0, 123, 123))

        # Display output image
        cv2.imshow('image', tmp)
        cv2.imshow('hsv_image', output)

    def draw_cv_grid(self, image):
        if self.grid:
            for column in self.grid.grid:
                for slot in column:
                    x = int(slot.x)
                    y = int(slot.y)
                    height = slot.height
                    width = slot.width
                    x = int(x - (width / 2))
                    y = int(y - (height / 2))
                    x2 = int(x + width)
                    y2 = int(y + height)

                    if slot.occupied == GridRectangle.WAYPOINT:
                        cv2.rectangle(image, (x, y), (x2, y2), (250,206,135), -1)
                    elif slot.occupied == GridRectangle.OBSTACLE:
                        cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 0), -1)
                    elif slot.occupied == GridRectangle.CONE:
                        cv2.rectangle(image, (x, y), (x2, y2), (0, 165, 255), -1)
                    elif slot.occupied == GridRectangle.AVOID:
                        cv2.rectangle(image, (x, y), (x2, y2), (100, 100, 100), -1)
                    else:
                        cv2.rectangle(image, (x, y), (x2, y2), (255, 255, 255), self.bWidth)

    def draw_border(self, rectangle, image, color):
        x, y = rectangle.x, rectangle.y
        box = cv2.boxPoints(((x, y), (rectangle.width, rectangle.height), rectangle.rotation))
        box = np.int0(box)
        im = cv2.drawContours(image, [box], 0, color, self.bWidth)

    def _update_trackbar_values(self):
        self.hMin = cv2.getTrackbarPos('HMin', 'controls')
        self.sMin = cv2.getTrackbarPos('SMin', 'controls')
        self.vMin = cv2.getTrackbarPos('VMin', 'controls')

        self.hMax = cv2.getTrackbarPos('HMax', 'controls')
        self.sMax = cv2.getTrackbarPos('SMax', 'controls')
        self.vMax = cv2.getTrackbarPos('VMax', 'controls')

        self.dilSize = cv2.getTrackbarPos('Dilate', 'controls')
        self.eroSize = cv2.getTrackbarPos('Erode', 'controls')
        self.bWidth = cv2.getTrackbarPos('Border_Width', 'controls')
        self.dist_max = cv2.getTrackbarPos('DistMax', 'controls')
        self.dist_min = cv2.getTrackbarPos('DistMin', 'controls')
        self.draw_grid = cv2.getTrackbarPos('DrawGrid', 'controls') == 1
        self.border_mode = cv2.getTrackbarPos('Border_Mode', 'controls')
        self.min_size = cv2.getTrackbarPos('NoiseFilter', 'controls')
        self.pathfinding = cv2.getTrackbarPos('Path', 'controls') == 1

    def trackbar_value_changed(self, trackbar):
        if self.initialized:
            self._update_trackbar_values()
            self.update()

    def create_windows(self):
        cv2.namedWindow('image')
        cv2.namedWindow('hsv_image')
        cv2.namedWindow('controls')

        # create trackbars for color change
        cv2.createTrackbar('HMin', 'controls', 0, 179, self.trackbar_value_changed)  # Hue is from 0-179 for Opencv
        cv2.createTrackbar('SMin', 'controls', 0, 255, self.trackbar_value_changed)
        cv2.createTrackbar('VMin', 'controls', 0, 255, self.trackbar_value_changed)
        cv2.createTrackbar('HMax', 'controls', 0, 179, self.trackbar_value_changed)
        cv2.createTrackbar('SMax', 'controls', 0, 255, self.trackbar_value_changed)
        cv2.createTrackbar('VMax', 'controls', 0, 255, self.trackbar_value_changed)
        cv2.createTrackbar('Erode', 'controls', 0, 24, self.trackbar_value_changed)
        cv2.createTrackbar('Dilate', 'controls', 0, 24, self.trackbar_value_changed)
        cv2.createTrackbar('Border_Width', 'controls', 0, 3, self.trackbar_value_changed)
        cv2.createTrackbar('Border_Mode', 'controls', 0, 1, self.trackbar_value_changed)  # default, bounding
        cv2.createTrackbar('DistMax', 'controls', 1, 20, self.trackbar_value_changed)
        cv2.createTrackbar('DistMin', 'controls', 1, 20, self.trackbar_value_changed)
        cv2.createTrackbar('DrawGrid', 'controls', 0, 1, self.trackbar_value_changed)
        cv2.createTrackbar('NoiseFilter', 'controls', 0, 50, self.trackbar_value_changed)
        cv2.createTrackbar('Path', 'controls', 0, 1, self.trackbar_value_changed)

        # Set default value for MAX HSV trackbars.
        cv2.setTrackbarPos('HMax', 'controls', 179)
        cv2.setTrackbarPos('SMax', 'controls', 255)
        cv2.setTrackbarPos('VMax', 'controls', 255)
        cv2.setTrackbarPos('DistMax', 'controls', 7)
        cv2.setTrackbarPos('DistMin', 'controls', 4)

        # Set default value for MIN.
        cv2.setTrackbarPos('HMin', 'controls', 0)
        cv2.setTrackbarPos('SMin', 'controls', 81)
        cv2.setTrackbarPos('VMin', 'controls', 95)
        cv2.setTrackbarPos('Path', 'controls', 1)

        self.initialized = True

    def start(self):
        while 1:
            self._update_trackbar_values()
            self.update()
            WAIT = 2000
            k = cv2.waitKey(WAIT) & 0xFF
            if k == 27:
                break


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage: python THIS_FILE.py <ImageFilePath>")
        exit()
    CV(sys.argv[1])
