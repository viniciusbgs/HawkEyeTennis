from ultralytics import YOLO
import cv2
import pickle
import sys 
sys.path.append('../')
from utils import get_center_of_bb, get_distance

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    
    def choose_and_filter_players(self, player_dict_list, court_keypoints):
        player_detections_first_frame = player_dict_list[0]
        choosen_players = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_dict_list = []
        for player_dict in player_dict_list:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in choosen_players}
            filtered_player_dict_list.append(filtered_player_dict) 
        return filtered_player_dict_list


    def choose_players(self, keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bb(bbox)
            min_distance = float('inf')
            
            for i in range(0, len(keypoints), 2):
                court_keypoint = (keypoints[i], keypoints[i+1])
                distance = get_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        #sort by distance
        distances.sort(key=lambda x: x[1])
        #choose the two closest players
        chosen_players = [distances[0][0], distances[1][0]]
        
        return chosen_players


    def detect_frames(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_dict_list = pickle.load(f)
            return player_dict_list

        player_dict_list = []

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_dict_list.append(player_dict)

        if stub_path is not None:
            with open('player_dict.pkl', 'wb') as f:
                pickle.dump(player_dict_list, f)

        return player_dict_list

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            #continua se for nulo
            if box.id is None:
                continue
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]

            if object_cls_name == 'person':
                player_dict[track_id] = result
                
        return player_dict
            

    def draw_boxes(self, video_frames, player_detection):
        output_video_frames = []

        for frame, player_dict in zip(video_frames, player_detection):
            #Draw bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f'Player ID: {track_id}', (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames
