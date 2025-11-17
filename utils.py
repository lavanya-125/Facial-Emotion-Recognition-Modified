import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import json

class EmotionVisualizer:
    """Handles visualization of emotion trends and statistics"""
    
    def __init__(self, history_size=300):
        self.history_size = history_size
        self.emotion_timeline = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        
    def add_emotion(self, emotion, timestamp=None):
        """Add emotion to timeline"""
        self.emotion_timeline.append(emotion)
        if timestamp is None:
            timestamp = datetime.now()
        self.timestamps.append(timestamp)
    
    def create_timeline_graph(self, width=400, height=150):
        """Create a mini timeline graph showing emotion history"""
        graph = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        if len(self.emotion_timeline) < 2:
            return graph
        
        # Color map for emotions
        emotion_colors = {
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'angry': (0, 0, 255),
            'surprise': (255, 255, 0),
            'fear': (128, 0, 128),
            'disgust': (0, 128, 128),
            'neutral': (128, 128, 128)
        }
        
        # Draw timeline
        timeline = list(self.emotion_timeline)
        num_points = len(timeline)
        
        for i in range(num_points):
            x = int((i / num_points) * width)
            emotion = timeline[i]
            color = emotion_colors.get(emotion, (128, 128, 128))
            
            cv2.circle(graph, (x, height // 2), 3, color, -1)
            
            if i > 0:
                x_prev = int(((i-1) / num_points) * width)
                cv2.line(graph, (x_prev, height // 2), (x, height // 2), color, 2)
        
        # Add title
        cv2.putText(graph, "Emotion Timeline", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return graph
    
    def get_emotion_statistics(self):
        """Calculate emotion statistics"""
        if not self.emotion_timeline:
            return {}
        
        from collections import Counter
        emotion_counts = Counter(self.emotion_timeline)
        total = len(self.emotion_timeline)
        
        stats = {}
        for emotion, count in emotion_counts.items():
            stats[emotion] = {
                'count': count,
                'percentage': (count / total) * 100
            }
        
        return stats


class PerformanceMonitor:
    """Monitor FPS and processing performance"""
    
    def __init__(self):
        self.frame_times = deque(maxlen=30)
        self.last_time = None
    
    def update(self):
        """Update FPS calculation"""
        import time
        current_time = time.time()
        
        if self.last_time is not None:
            frame_time = current_time - self.last_time
            self.frame_times.append(frame_time)
        
        self.last_time = current_time
    
    def get_fps(self):
        """Get current FPS"""
        if not self.frame_times:
            return 0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    
    def draw_fps(self, frame, position=(10, 30)):
        """Draw FPS on frame"""
        fps = self.get_fps()
        text = f"FPS: {fps:.1f}"
        
        cv2.putText(frame, text, position,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame


class EmotionLogger:
    """Log emotion data to file"""
    
    def __init__(self, filename="emotion_log.json"):
        self.filename = filename
        self.log_data = []
    
    def log_emotion(self, emotion, confidence, timestamp=None, metadata=None):
        """Log emotion with timestamp"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        entry = {
            'timestamp': timestamp,
            'emotion': emotion,
            'confidence': float(confidence),
            'metadata': metadata or {}
        }
        
        self.log_data.append(entry)
    
    def save_log(self):
        """Save log to file"""
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.log_data, f, indent=2)
            print(f"Log saved to {self.filename}")
        except Exception as e:
            print(f"Error saving log: {e}")
    
    def load_log(self):
        """Load log from file"""
        try:
            with open(self.filename, 'r') as f:
                self.log_data = json.load(f)
            print(f"Log loaded from {self.filename}")
        except FileNotFoundError:
            print(f"No existing log file found")
        except Exception as e:
            print(f"Error loading log: {e}")


def draw_enhanced_ui(frame, emotion, confidence, fps, emotion_stats=None):
    """Draw enhanced UI with modern styling"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay for header
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Main emotion display
    cv2.putText(frame, f"{emotion.upper()}", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
    # Confidence bar
    bar_width = 200
    bar_height = 20
    bar_x, bar_y = width - bar_width - 20, 30
    
    # Background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                 (50, 50, 50), -1)
    
    # Confidence fill
    fill_width = int(confidence * bar_width)
    color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 165, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                 color, -1)
    
    # Confidence text
    cv2.putText(frame, f"{confidence*100:.1f}%", (bar_x + bar_width + 10, bar_y + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # FPS display
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, height - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Stats display (if available)
    if emotion_stats:
        y_offset = height - 100
        cv2.putText(frame, "Session Stats:", (width - 200, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        for emotion_name, stats in sorted(emotion_stats.items(), 
                                         key=lambda x: x[1]['percentage'], 
                                         reverse=True)[:3]:
            text = f"{emotion_name}: {stats['percentage']:.1f}%"
            cv2.putText(frame, text, (width - 200, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15
    
    return frame