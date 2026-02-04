from ultralytics import YOLO

def train_model():
    # 1. Load the model. 
    model = YOLO('yolov8n-cls.pt')  

    # 2. Train the model
    results = model.train(
        data='../data.yaml',
        epochs=100, 
        imgsz=320,                     
        batch=32,
        project='../trained_models',                         
        name='results',      
        device=0           
    )

    # 3. Validate the model
    metrics = model.val()

if __name__ == '__main__':
    train_model()