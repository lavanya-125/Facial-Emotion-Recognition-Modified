import cv2
import numpy as np
from typing import Dict, Tuple

class EmotionExplainer:
    """
    Provides explainability for emotion predictions by highlighting
    important facial regions and providing reasoning
    """
    
    def __init__(self):
        # Define facial regions and their importance for each emotion
        self.region_importance = {
            'angry': {'eyebrows': 0.4, 'eyes': 0.3, 'mouth': 0.3},
            'disgust': {'nose': 0.3, 'mouth': 0.5, 'eyes': 0.2},
            'fear': {'eyes': 0.5, 'eyebrows': 0.3, 'mouth': 0.2},
            'happy': {'mouth': 0.5, 'eyes': 0.4, 'cheeks': 0.1},
            'sad': {'eyebrows': 0.4, 'eyes': 0.3, 'mouth': 0.3},
            'surprise': {'eyes': 0.4, 'eyebrows': 0.3, 'mouth': 0.3},
            'neutral': {'eyes': 0.3, 'mouth': 0.3, 'face': 0.4}
        }
        
        # Facial region coordinates (relative to face bounding box)
        self.regions = {
            'eyebrows': (0.2, 0.15, 0.8, 0.3),  # (x_start, y_start, x_end, y_end)
            'eyes': (0.15, 0.25, 0.85, 0.45),
            'nose': (0.35, 0.4, 0.65, 0.6),
            'mouth': (0.3, 0.65, 0.7, 0.85),
            'cheeks': (0.1, 0.45, 0.9, 0.7),
            'face': (0.0, 0.0, 1.0, 1.0)
        }
    
    def generate_explanation(self, emotion: str, confidence: float, 
                            all_probs: np.ndarray, emotion_labels: list) -> str:
        """
        Generate text explanation for the prediction
        
        Args:
            emotion: Predicted emotion
            confidence: Confidence score
            all_probs: All emotion probabilities
            emotion_labels: List of emotion labels
            
        Returns:
            str: Human-readable explanation
        """
        explanation = f"Detected: {emotion.upper()} ({confidence*100:.1f}% confident)\n\n"
        
        # Get key facial features for this emotion
        if emotion in self.region_importance:
            regions = self.region_importance[emotion]
            sorted_regions = sorted(regions.items(), key=lambda x: x[1], reverse=True)
            
            explanation += "Key indicators:\n"
            for region, importance in sorted_regions[:3]:
                explanation += f"  • {region.title()}: {importance*100:.0f}% importance\n"
        
        # Show alternative emotions if confidence is low
        if confidence < 0.6:
            sorted_probs = sorted(zip(emotion_labels, all_probs), 
                                key=lambda x: x[1], reverse=True)
            explanation += f"\nAlternatives:\n"
            for alt_emotion, alt_prob in sorted_probs[1:3]:
                if alt_prob > 0.1:
                    explanation += f"  • {alt_emotion}: {alt_prob*100:.1f}%\n"
        
        return explanation
    
    def create_heatmap(self, face_roi: np.ndarray, emotion: str) -> np.ndarray:
        """
        Create attention heatmap showing important facial regions
        
        Args:
            face_roi: Face region of interest
            emotion: Detected emotion
            
        Returns:
            np.ndarray: Heatmap overlay
        """
        h, w = face_roi.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        if emotion not in self.region_importance:
            return heatmap
        
        # Create heatmap based on region importance
        for region, importance in self.region_importance[emotion].items():
            if region in self.regions:
                x1, y1, x2, y2 = self.regions[region]
                x1, x2 = int(x1 * w), int(x2 * w)
                y1, y2 = int(y1 * h), int(y2 * h)
                
                # Add Gaussian-like distribution
                region_heatmap = np.zeros((h, w), dtype=np.float32)
                region_heatmap[y1:y2, x1:x2] = importance
                
                # Smooth the region
                region_heatmap = cv2.GaussianBlur(region_heatmap, (0, 0), w * 0.1)
                heatmap += region_heatmap
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def apply_heatmap_overlay(self, frame: np.ndarray, face_roi: np.ndarray, 
                             emotion: str, x: int, y: int, w: int, h: int,
                             alpha: float = 0.4) -> np.ndarray:
        """
        Apply heatmap overlay on the original frame
        
        Args:
            frame: Original frame
            face_roi: Face region
            emotion: Detected emotion
            x, y, w, h: Face bounding box coordinates
            alpha: Overlay transparency
            
        Returns:
            np.ndarray: Frame with heatmap overlay
        """
        # Generate heatmap
        heatmap = self.create_heatmap(face_roi, emotion)
        
        # Convert heatmap to color (red = high importance)
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Resize heatmap to match face ROI
        heatmap_colored = cv2.resize(heatmap_colored, (w, h))
        
        # Blend with original frame
        face_region = frame[y:y+h, x:x+w]
        blended = cv2.addWeighted(face_region, 1-alpha, heatmap_colored, alpha, 0)
        
        # Place back on frame
        result_frame = frame.copy()
        result_frame[y:y+h, x:x+w] = blended
        
        return result_frame
    
    def draw_region_boxes(self, frame: np.ndarray, emotion: str, 
                         x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Draw boxes around important facial regions
        
        Args:
            frame: Original frame
            emotion: Detected emotion
            x, y, w, h: Face bounding box coordinates
            
        Returns:
            np.ndarray: Frame with region boxes
        """
        if emotion not in self.region_importance:
            return frame
        
        result = frame.copy()
        
        # Draw boxes for top 2 important regions
        sorted_regions = sorted(
            self.region_importance[emotion].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:2]
        
        colors = [(0, 255, 0), (0, 255, 255)]  # Green and yellow
        
        for i, (region, importance) in enumerate(sorted_regions):
            if region in self.regions:
                rx1, ry1, rx2, ry2 = self.regions[region]
                
                # Convert to absolute coordinates
                abs_x1 = x + int(rx1 * w)
                abs_y1 = y + int(ry1 * h)
                abs_x2 = x + int(rx2 * w)
                abs_y2 = y + int(ry2 * h)
                
                # Draw rectangle
                cv2.rectangle(result, (abs_x1, abs_y1), (abs_x2, abs_y2), 
                            colors[i], 2)
                
                # Add label
                label = f"{region}: {importance*100:.0f}%"
                cv2.putText(result, label, (abs_x1, abs_y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)
        
        return result
    
    def create_explanation_panel(self, width: int, height: int, 
                                explanation_text: str, 
                                emotion_probs: np.ndarray,
                                emotion_labels: list) -> np.ndarray:
        """
        Create a side panel with detailed explanation
        
        Args:
            width: Panel width
            height: Panel height
            explanation_text: Text explanation
            emotion_probs: Probability distribution
            emotion_labels: List of emotion labels
            
        Returns:
            np.ndarray: Explanation panel image
        """
        panel = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # Add title
        cv2.putText(panel, "Explainability", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add explanation text
        y_offset = 60
        for line in explanation_text.split('\n'):
            if line.strip():
                cv2.putText(panel, line[:50], (10, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                y_offset += 20
        
        # Add probability bar chart
        if emotion_probs is not None and len(emotion_probs) > 0:
            bar_y_start = height - 200
            cv2.putText(panel, "Emotion Distribution:", (10, bar_y_start),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            bar_height = 15
            max_bar_width = width - 100
            
            for i, (label, prob) in enumerate(zip(emotion_labels, emotion_probs)):
                y_pos = bar_y_start + 30 + i * 25
                bar_width = int(prob * max_bar_width)
                
                # Draw bar
                color = (0, 255, 0) if prob == max(emotion_probs) else (200, 200, 200)
                cv2.rectangle(panel, (10, y_pos), (10 + bar_width, y_pos + bar_height),
                            color, -1)
                
                # Draw label and percentage
                cv2.putText(panel, f"{label}: {prob*100:.1f}%", 
                          (10 + max_bar_width + 10, y_pos + 12),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return panel