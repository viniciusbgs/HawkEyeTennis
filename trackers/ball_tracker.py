from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_list):
        ball_positions = [x.get(1, []) for x in ball_list]

        #convert to pandas dataframe
        ball_positions_df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        #interpolate missing values
        ball_positions_df = ball_positions_df.interpolate()
        ball_positions_df = ball_positions_df.bfill()

        #convert back to list 
        ball_positions = [{1:x} for x in ball_positions_df.to_numpy().tolist()]

        return ball_positions


    def detect_frames(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_dict_list = pickle.load(f)
            return ball_dict_list

        ball_dict_list = []

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_dict_list.append(ball_dict)

        if stub_path is not None:
            with open('ball_dict.pkl', 'wb') as f:
                pickle.dump(ball_dict_list, f)

        return ball_dict_list


    def detect_frame(self, frame):

        results = self.model.predict(frame, conf=0.15)[0]
        ball_dict = {}

        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
                
        return ball_dict
    
    def draw_boxes(self, video_frames, ball_dict_list):

        output_video_frames = []

        for frame, ball_dict in zip(video_frames, ball_dict_list):
            #Draw bounding boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f'Ball: {track_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            output_video_frames.append(frame)

        return output_video_frames

