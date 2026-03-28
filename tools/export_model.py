from ultralytics import YOLO

model = YOLO('yolo26n.pt')

model.export(format='engine', device='0', half=True, imgsz=320, verbose=True)   