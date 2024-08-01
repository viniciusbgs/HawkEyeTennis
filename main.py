from utils import (read_video,
                   save_video)
from trackers import(PlayerTracker,
                     BallTracker)

from ultralytics import YOLO
import os

def main():
        
    input_video_path = 'input_videos/input_video.mp4'

    #Read Video
    video_frames = read_video(input_video_path)

    #Create Player and Ball Trackers
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/last.pt')

    #Detect Players 
    player_detections = player_tracker.detect_frames(video_frames, 
                                                        read_from_stub=True,
                                                        stub_path='tracker_stubs/player_dict.pkl')
    #Detect Ball
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                    read_from_stub=True,
                                                    stub_path='tracker_stubs/ball_dict.pkl')

    #Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_boxes(video_frames, player_detections)
    #Draw Ball Bounding Boxes
    output_video_frames = ball_tracker.draw_boxes(output_video_frames, ball_detections)

    #Save Video
    ##OBS: QUERO SALVAR EM MP4 DPS
    save_video(output_video_frames, 'output_videos/output_video.avi')


    return True

if __name__ == "__main__":
    main() 
