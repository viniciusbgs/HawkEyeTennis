from ultralytics import YOLO

model = YOLO('models/yolov8x')

result = model.track('input_videos/image.png', save=True)

