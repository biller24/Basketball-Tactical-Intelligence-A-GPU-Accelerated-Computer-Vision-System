import cv2

class TacticalViewDrawer:
    def __init__(self, team_1_color=[255, 245, 238], team_2_color=[128, 0, 0]):
        self.start_x = 20
        self.start_y = 40
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    def draw(self,
             video_frames,
             court_image_path,
             width,
             height,
             tactical_court_keypoints
             ):
        court_image = cv2.imread(court_image_path)
        court_image = cv2.resize(court_image, (width, height))

        output_video_frames = []
        for frame_idx, frame in enumerate(video_frames):
            frame = frame.copy()

            y1 = self.start_y
            y2 = self.start_y + height
            x1 = self.start_x
            x2 = self.start_x + width

            alpha = 0.6  # Transparency factor
            overlay = frame[y1:y2, x1:x2].copy()
            cv2.addWeighted(court_image, alpha, overlay, 1 - alpha, 0, frame[y1:y2, x1:x2])

            # Draw court keypoints
            for keypoint_index, keypoint in enumerate(tactical_court_keypoints):
                x, y = keypoint
                x += self.start_x
                y += self.start_y
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(frame, str(keypoint_index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            output_video_frames.append(frame)

        return output_video_frames