from ultralytics import YOLO

model = YOLO("./runs/detect/basketball_v2_yolo11/weights/best.pt")
results = model.predict("input_videos/video_1.mp4", device = 0, save=True)
print(results)
print("zzzzzzzzzz")
for box in results[0].boxes:
    print(box)
