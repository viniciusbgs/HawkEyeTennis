from utils import (read_video,
                   save_video)
from trackers import(PlayerTracker,
                     BallTracker)
import cv2
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt


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
    
    #Interpolate Ball Positions
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections) 

    #Court Line Detector
    court_model_path = 'models/keypoints_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    #Choose and Filter Players
    player_detections = player_tracker.choose_and_filter_players(player_detections, court_keypoints)

    #Get Ball Shot Frames
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    print(ball_shot_frames)

    #Create Mini Court
    mini_court = MiniCourt(video_frames[0])

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                          ball_detections,
                                                                                                          court_keypoints)

    #Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_boxes(video_frames, player_detections) 
    #Draw Ball Bounding Boxes
    output_video_frames = ball_tracker.draw_boxes(output_video_frames, ball_detections)
    #Draw Court Line Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    

    #Draw Foot Positions
    output_video_frames = mini_court.draw_foot_positions(output_video_frames, player_detections)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                          ball_detections,
                                                                                                          court_keypoints)

    #Write number of frame:
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"frame: {i}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    #Save Video
    ##OBS: QUERO SALVAR EM MP4 DPS
    save_video(output_video_frames, 'output_videos/output_video.avi')

    return True

if __name__ == "__main__":
    main() 
