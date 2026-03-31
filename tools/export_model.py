from ultralytics import YOLO

model = YOLO('yolo26n.pt') # swap this with s/m model as needed before export

model.export(format='engine', device='0', half=True, imgsz=320, verbose=True)   