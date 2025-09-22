
from ultralytics import YOLO
import cv2
from pathlib import Path

def predict_image(model_path, image_path, conf=0.5):
    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf)
    result = results[0]
    
    img = result.plot()  # Draw boxes
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert for Streamlit
    
    labels = []
    for box in result.boxes:
        cls_id = int(box.cls)
        x_center, y_center, width, height = box.xywhn[0].tolist()
        labels.append({
            "class_id": cls_id,
            "bbox": [x_center, y_center, width, height]
        })
    
    return img, labels
