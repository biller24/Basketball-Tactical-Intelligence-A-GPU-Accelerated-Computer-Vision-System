from ultralytics import YOLO
import supervision as sv
import sys
sys.path.append('../')
from utils import read_stub, save_stub


class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 12 # safe number for 1050 Ti(4GB)
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            detections_batch = self.model.predict(batch, conf=0.5, device=0)
            detections += detections_batch
        return detections

    def get_object_tracks(self,frames,read_from_stub= False, stub_path=None):
        tracks = read_stub(read_from_stub,stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)
        tracks=[]

        for frame_num,detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks.append({})

            for bbox, cls_id, track_id in zip(detection_with_tracks.xyxy,
                                              detection_with_tracks.class_id,
                                              detection_with_tracks.tracker_id):

                # Check if the detection is a 'Player'
                if cls_id == cls_names_inv["Player"]:
                    tracks[frame_num][track_id] = {"box": bbox.tolist()}
        save_stub(stub_path,tracks)
        return tracks

