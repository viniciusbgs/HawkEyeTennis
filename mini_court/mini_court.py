import cv2
import sys
sys.path.append('../')
import constants
from utils import convert_meters_to_pixel_distance, convert_pixel_distance_to_meters

class MiniCourt():
    def __init__(self, frame):
        self.drawning_rectangle_width = 250
        self.drawning_rectangle_height = 450
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)

    def convert_meters_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters, constants.COURT_HEIGHT, self.court_drawning_width)

    def set_court_drawning_keypoints(self, keypoints):
        drawning_keypoints = []*28

        #point 0
        drawning_keypoints[0], drawning_keypoints[1] = int(self.court_start_x), int(self.court_start_y)
        #point 1
        drawning_keypoints[2], drawning_keypoints[3] = int(self.court_end_x), int(self.court_start_y)
        #point 2
        drawning_keypoints[4] = int(self.court_start_x)
        drawning_keypoints[5] = self.court_start_y + convert_meters_to_pixel_distance(constants.COURT_HEIGHT,
                                                                                      constants.DOUBLE_LINE_WIDTH,
                                                                                        self.court_drawning_width)




    def set_mini_court_position(self, frame):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawning_width = self.court_end_x - self.court_start_x 

    def set_canvas_background_box_position(self, frame):
        frames = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = frame.shape[0] + self.drawning_rectangle_height
        self.start_x = self.end_x - self.drawning_rectangle_width
        self.start_y = self.end_y - self.drawning_rectangle_height
    