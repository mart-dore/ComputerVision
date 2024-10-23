# imports
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width
import cv2
import numpy as np
import pandas as pd

class Tracker:
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill() # if first frame is missing position

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections


    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks    

        # 1. detection
        detections = self.detect_frames(frames)

        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GK to normal player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            # 2. Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            # loop through detection with tracks for players and referees
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {"bbox":bbox}
            
            # loop through detection without tracks for the ball
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bbox":bbox} # hardcode to one because only one ball in field              

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3]) # bottow y because circle is drawn on the floor
        x_center, _ = get_center_of_bbox(bbox) # only need x of center
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45, # draw ellipse from angle 45 to 245 and not full circle
            endAngle=245,
            color=color, # ! note that color is in BGR format instead of RGB
            thickness=2,
            lineType=cv2.LINE_4
        )
        # coordinates of the rectangle below players
        rectangle_width = 40
        rectangle_height = 20

        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15 
        y2_rect = (y2 + rectangle_height // 2) + 15 

        if track_id is not None:
            # draw rectangle
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color=color,
                thickness=cv2.FILLED
            )
            # coordonates of text inside of rectangle
            x1_text = x_center - 12
            if track_id > 99:
                x1_text -= 10
            
            # draw text
            cv2.putText(
                frame,
                text=f'{track_id}',
                org=(int(x1_text), int(y1_rect+15)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0,0,0),
                thickness=2                 
            )


        return frame


    def draw_triangle(self, frame, bbox, color, track_id):
        x1, y1, x2, y2 = bbox # top y because triangle is draw on top of the ball
        x_center, y_center = get_center_of_bbox(bbox) 
        width = 7
        triangle_height = 12

        p1 = (x_center, int(y1))
        p2 = (int(x_center - width), int(y1 - triangle_height))
        p3 = (int(x_center + width), int(y1 - triangle_height))

        triangle_points = np.array([p1, p2, p3])

        cv2.fillPoly(
            frame,
            pts=[triangle_points],
            color=color, # ! note that color is in BGR format instead of RGB
        )
        return frame


    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # draw transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = .5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        # get number of time each team has the ball
        team1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        team1 = team1_num_frames / (team1_num_frames + team2_num_frames)
        team2 = team2_num_frames / (team2_num_frames + team1_num_frames)

        cv2.putText(frame, f'Team 1 Possession : {team1*100:.2f}%', (1400, 900),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f'Team 2 Possession : {team2*100:.2f}%', (1400, 950),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame
    
    
    def draw_annotations(self, frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(frames):
            # frame = frame.copy()

            # extract dict from objects at good frame_number
            players_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referees_dict = tracks['referees'][frame_num]

            # draw players
            for track_id, player in players_dict.items():
                color = player.get('team_color', (0,0,255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

                if player.get('has_ball', False) :
                    frame = self.draw_triangle(frame, player['bbox'], (0,0,255), track_id)

            # draw referees
            for _, referee in referees_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255))

            # draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 255, 0), track_id)

            # draw team possession
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
