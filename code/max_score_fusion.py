import numpy as np
import os
from ultralytics import YOLO
from typing import List, Tuple, Dict
import pandas as pd

class MultiViewInletClassifier:
    """
    A multi-view classification architecture utilizing a YOLO model.
    This framework evaluates three distinct views of a single inlet and uses max-score fusion to determine the final clogging condition.
    """
    def __init__(self, model_weights_path: str):
        """
        Initializes the classifier with the pre-trained YOLO model.
        :param model_weights_path (str): Path to the trained YOLO weights (e.g., 'best.pt').
        """
        # Load the trained YOLO classification model
        self.model = YOLO(model_weights_path)
        
        # Assuming YOLO class mapping: 0 -> 'clean', 1 -> 'clogged'
        # Update these based on your specific model's class indices
        self.class_names = {0: 'clean', 1: 'clogged'}

    def _predict_single_view(self, image_path: str) -> np.ndarray:
        """
        Runs inference on a single image view and retrieves the softmax probabilities.
        :param image_path (str): File path to the inlet image.  
        Returns np.ndarray a 1D (1 x 2)array of confidence scores for each class.
        """
        # Run inference (imgsz can be adjusted based on your training parameters)
        results = self.model.predict(source=image_path, verbose=False)
        
        # Extract the classification probabilities (softmax scores)
        # results[0].probs.data contains the tensor of class confidences
        probs = results[0].probs.data.cpu().numpy()
        return probs

    def max_score_fusion(self, view_scores: List[np.ndarray]) -> Tuple[str, float, int]:
        """
        Applies max-score fusion across the classification scores of all views.
        Selects the prediction from the view that yields the absolute highest 
        confidence score for any class.
        :param view_scores: A list of probability arrays from each view.   
        Returns the final predicted class label, the confidence score, and the index of the view that provided this result.
        """
        # Convert list of arrays into a 2D matrix (rows = views, cols = classes)
        # Shape will be (3, 2) for 3 views and 2 classes
        score_matrix = np.vstack(view_scores)
        
        # Find the overall maximum score in the entire matrix
        max_score = np.max(score_matrix)
        
        # Find the indices of that maximum score 
        # View_idx is the view that gave the highest confidence
        # Class_idx is the predicted class (0 or 1)
        view_idx, class_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        predicted_label = self.class_names[class_idx]
        return predicted_label, max_score, view_idx

    def classify_inlet(self, view_paths: List[str]) -> Dict:
        """
        Orchestrates the multi-view classification for a single inlet.
        :param view_paths (List[str]): A list containing exactly three image paths.       
        Returns a summary dictionary containing the final prediction, confidence, and individual view scores.
        """
        if len(view_paths) != 3:
            raise ValueError("Exactly three views must be provided for the multi-view architecture.")

        # Gather probability distributions for all three views
        all_view_scores = []
        for path in view_paths:
            scores = self._predict_single_view(path)
            all_view_scores.append(scores)

        # Apply Max-Score Fusion
        final_label, best_confidence, selected_view = self.max_score_fusion(all_view_scores)

        # Construct the results summary
        results_summary = {
            "final_prediction": final_label,
            "fusion_confidence": best_confidence,
            "selected_view_index": selected_view,
            "raw_scores": [score.tolist() for score in all_view_scores]
        }

        return results_summary

if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best.pt')
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')

    # Initialize the architecture with your trained weights
    classifier = MultiViewInletClassifier(model_weights_path=model_path)
    
    # Initialize a dataframe to summarize the results
    results = pd.DataFrame()

    # Iterate through the folders that contain multiple sets of three views for a specific inlet observation
    for folder_name in os.listdir(data_path):
        inlet_views = []
        for view_set in folder_name:
            view = os.path.join(folder_name, view_set)
            inlet_views.append(view)
        
        # Perform classification
        prediction_results = classifier.classify_inlet(inlet_views)

        # Save the results summary as a dataframe
        new_row = {'inlet_id': folder_name, 'final_prediction': prediction_results['final_prediction'],
                        'fusion_confidence': prediction_results['fusion_confidence'], 'selected_view_index': prediction_results['selected_view_index'],
                        'raw_scores': prediction_results['raw_scores']}
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        