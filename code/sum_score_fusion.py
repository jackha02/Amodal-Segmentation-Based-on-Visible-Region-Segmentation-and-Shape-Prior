import os
import torch
from ultralytics import YOLO
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torchvision.models as models

# We assume that the YOLOv8 model is already trained for classification tasks
# Training images are saved in two separate folders
# One for single-view images and another for multi-view images

class SingleViewYOLO:
    """
    Single-view YOLOv8 classification
    """
    def __init__(self, model_path, device = 'cuda'):
        # Load the trained single-view YOLOv8-cls model
        self.model = YOLO(model_path)
        self.device = device if torch.cuda.is_available() else 'cpu'

    def predict(self, image_path):
        """
        Predicts the class of a single image
        :param image_path: file path to the single-view images
        Returns a saved csv file containing the inlet ID, predicted class name, and confidence score
        """
        results_list = []
        for image in image_path:
            if not image.endswith(".jpg"):
                continue
            results = self.model(image, device=self.device, verbose=False)
            if results[0].probs is None:
                raise ValueError("Model provided is not a classification model.")
            probs = results[0].probs.data
            best_class_idx = torch.argmax(probs).item()
            predicted_class_name = self.model.names[best_class_idx]
            confidence_score = probs[best_class_idx].item()
            inlet_id = os.path.splitext(os.path.basename(image))[0]
            results_list.append((inlet_id, predicted_class_name, confidence_score))
        df = pd.DataFrame(results_list, columns=["Inlet ID", "Predicted Class", "Confidence Score"])
        df.to_csv("single_view_predictions.csv", index=False)
        best_class_idx = torch.argmax(probs).item()
        predicted_class_name = self.model.names[best_class_idx]
        confidence_score = probs[best_class_idx].item()
        return predicted_class_name, confidence_score

class MultiViewYOLO:
    """
    Implements Multi-view sum score fusion for YOLOv8 as described in Seeland & Mader (2021)
    """
    def __init__(self, model_path: str, device: str = 'cuda'):
        # Load the trained single-view YOLOv8-cls model
        self.model = YOLO(model_path)
        self.device = device if torch.cuda.is_available() else 'cpu'

    def bounding_box_coordinates(self, test_data, results):
        """
        Extracts bounding box coordinates from YOLOv8 detection results
        :param test_data: file path to test images
        :param results: file path to save results
        :return: List of bounding box coordinates in the format (inlet ID, view type, x_min, y_min, x_max, y_max)
        """
        bbox_coords = []
        for img_path, results in zip(test_data, results):
            view_type = self.get_view_type_from_filename(img_path)  # Implement this function based on naming convention
            for box in results.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                bbox_coords.append((inlet_id, view_type, x_min, y_min, x_max, y_max))
        return bbox_coords
     
    # A new function: From a given set of inlets, we need edge coordinates of the prediction bounding boxes
        # The coordinates are saved as a csv file in the results folder
        # The format of the csv should be: inlet ID, view type (0, 1, or 2), x_min, y_min, x_max, y_max
        # Understand that view 0 is approaching, view 1 is central, and view 2 is departing

    # A new function: we need a function that takes in a set of multiview images for a single object
        # and it performs sum-score fusion to output the final predicted class and average confidence score
        # it should also output the individual view predictions and confidence scores
        # this should be saved in a separate csv file in the results folder 
        # The format of the csv should be: inlet ID, final class prediction, average confidence score, view 0 prediction, view 0 confidence, view 1 prediction, view 1 confidence, view 2 prediction, view 2 confidence

    def predict_sum_fusion(self, view_paths):
        """
        Performs sum-score fusion on a collection of views for a single object
        :param view_paths: List of file paths to images (views) of the same object
        Returns a tuple containing the average confidence score
        """
        results = self.model(view_paths, device=self.device, verbose=False)
        view_scores = []
        for res in results:
            if res.probs is None:
                raise ValueError("Model provided is not a classification model.")
            view_scores.append(res.probs.data)
        stacked_scores = torch.stack(view_scores)
        summed_scores = torch.sum(stacked_scores, dim=0)
        best_class_idx = torch.argmax(summed_scores).item()
        predicted_class_name = self.model.names[best_class_idx]
        avg_confidence = summed_scores[best_class_idx].item() / len(view_paths)
        return predicted_class_name, avg_confidence

if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), "runs/detect/results/weights/best.onnx")
    classifier = MultiViewYOLO(model_path=model_path)
    # In here, predict_sum_fusion should be an iterative process to process multiple views for each object
    # Therefore, we need to loop through each object's views and call predict_sum_fusion
    # The ground truth data must be uploaded from a different folder path, which includes the actual class labels for each object
    # Ultimately, the outcome of this main loop is to generate two CSV files with the fused predictions for each object
    # The CSV file should contain the inlet ID, final class prediction, average confidence score, and individual view predictions with their confidence scores
    # We also need to compare this to a ground truth 
    object_views = [
        "./data/car_1_front.jpg",
        "./data/car_1_side.jpg", 
        "./data/car_1_rear.jpg"
    ]
    try:
        cls_name, conf = classifier.predict_sum_fusion(object_views)
        print(f"Fused Prediction: {cls_name}")
        print(f"Average Confidence: {conf:.4f}")
        
    except Exception as e:
        print(f"Inference failed: {e}")