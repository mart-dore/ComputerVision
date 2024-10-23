from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import cv2
import numpy as np

def main():
    print("starting main...")

    # 1 .Read video :
    frames = read_video('C:/Users/Martin/Documents/COURS/Python/MachineLearning/IA/YOLO/input_videos/08fd33_4.mp4')

    # 2. Initialize tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # # save image of players
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = frames[0]

    #     # crop bbox from frame
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     # save cropped image
    #     cv2.imwrite(f'output_video/cropped_img.jpg', cropped_image)
    #     break

    # 3. Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # 3. Assign Players Team 
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0],
                                    tracks['players'][0])
    

    for frame_number, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_number],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_number][player_id]['team'] = team
            tracks['players'][frame_number][player_id]['team_color'] = team_assigner.team_colors[team]

    # 4. Assign Ball to players and calculate ball control for each team
    team_ball_control = []
    player_assigner = PlayerBallAssigner()
    for frame_number, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_number][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_number][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_number][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1]) # if no one has the ball, possession goes to precedent player
    team_ball_control = np.array(team_ball_control)


    # 4. Draw output
    # Draw object tracks
    output_video_frames = tracker.draw_annotations(frames, tracks, team_ball_control)


    # Save video
    save_video(output_video_frames, 'output_video/output_video_1.avi')

    print("... done")



if __name__ == "__main__":
    main()
