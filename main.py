from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.yaml")

    model.train(data="config.yaml", epochs=100)