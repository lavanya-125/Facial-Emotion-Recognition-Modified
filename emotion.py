import cv2
import numpy as np
from deepface import DeepFace

# Import our modules
from temporal_model import TemporalEmotionModel
from explainer import EmotionExplainer
from utils import EmotionVisualizer, PerformanceMonitor, EmotionLogger

# Configuration
SHOW_HEATMAP = True
SHOW_REGION_BOXES = False
SHOW_EXPLANATION_PANEL = True
SKIP_FRAMES = 2
CONFIDENCE_THRESHOLD = 0.35  # Only show predictions above 35% confidence

# Initialize components
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
temporal_model = TemporalEmotionModel(sequence_length=30)
explainer = EmotionExplainer()
visualizer = EmotionVisualizer()
performance_monitor = PerformanceMonitor()
logger = EmotionLogger()

# Start video capture
cap = cv2.VideoCapture(0)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
panel_width = 400

frame_counter = 0
last_emotion = None
last_confidence = 0
last_probs = None

print("=" * 70)
print("IMPROVED TEMPORAL + EXPLAINABLE EMOTION DETECTION")
print("=" * 70)
print("\nControls:")
print("  'q' - Quit")
print("  'h' - Toggle heatmap")
print("  'b' - Toggle region boxes")
print("  'p' - Toggle explanation panel")
print("  'r' - Reset history")
print("  's' - Save log")
print("\nStarting...\n")

def draw_improved_ui(frame, emotion, confidence, fps):
    """Draw clean, readable UI"""
    height, width = frame.shape[:2]
    
    # Top bar with semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 100), (40, 40, 40), -1)
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    
    # Emotion label - large and clear
    cv2.putText(frame, f"EMOTION: {emotion.upper()}", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # Confidence with color coding
    conf_color = (0, 255, 0) if confidence > 0.7 else (0, 200, 255) if confidence > 0.5 else (0, 100, 255)
    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)
    
    # Confidence bar
    bar_x, bar_y = width - 230, 30
    bar_width, bar_height = 200, 30
    
    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                 (60, 60, 60), -1)
    
    # Filled portion
    fill_width = int(confidence * bar_width)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                 conf_color, -1)
    
    # Confidence percentage on bar
    cv2.putText(frame, f"{confidence*100:.0f}%", (bar_x + bar_width + 10, bar_y + 22),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, height - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def create_explanation_panel(width, height, emotion, confidence, all_probs, emotion_labels):
    """Create clean explanation panel"""
    panel = np.ones((height, width, 3), dtype=np.uint8) * 250
    
    y_offset = 30
    
    # Title
    cv2.rectangle(panel, (0, 0), (width, 60), (80, 80, 80), -1)
    cv2.putText(panel, "EMOTION ANALYSIS", (15, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    y_offset = 90
    
    # Main prediction
    cv2.putText(panel, "Detected Emotion:", (15, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_offset += 35
    
    cv2.putText(panel, emotion.upper(), (15, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 100, 255), 2)
    y_offset += 40
    
    cv2.putText(panel, f"{confidence*100:.1f}% confident", (15, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    y_offset += 40
    
    # Human-readable explanation
    cv2.line(panel, (15, y_offset), (width - 15, y_offset), (200, 200, 200), 2)
    y_offset += 30
    
    cv2.putText(panel, "Why This Emotion?", (15, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_offset += 25
    
    # Generate human-readable explanation
    explanation_text = explainer.generate_explanation(emotion, confidence, all_probs, emotion_labels)
    
    # Parse and display the explanation text
    for line in explanation_text.split('\n'):
        if line.strip():
            # Limit line length
            if len(line) > 45:
                line = line[:42] + "..."
            
            cv2.putText(panel, line, (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
            y_offset += 20
    
    y_offset += 15
    
    # Key indicators section
    cv2.line(panel, (15, y_offset), (width - 15, y_offset), (200, 200, 200), 2)
    y_offset += 30
    
    cv2.putText(panel, "Facial Region Importance:", (15, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    y_offset += 30
    
    # Get region importance for this emotion
    region_importance = explainer.region_importance.get(emotion, {})
    sorted_regions = sorted(region_importance.items(), key=lambda x: x[1], reverse=True)
    
    for i, (region, importance) in enumerate(sorted_regions[:3]):
        # Region name
        cv2.putText(panel, f"{region.title()}:", (25, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Importance bar
        bar_x = 150
        bar_width = 180
        bar_height = 15
        
        cv2.rectangle(panel, (bar_x, y_offset - 12), (bar_x + bar_width, y_offset + 3),
                     (220, 220, 220), -1)
        
        fill_width = int(importance * bar_width)
        color = (0, 200, 100) if i == 0 else (100, 150, 200) if i == 1 else (150, 150, 150)
        cv2.rectangle(panel, (bar_x, y_offset - 12), (bar_x + fill_width, y_offset + 3),
                     color, -1)
        
        cv2.putText(panel, f"{importance*100:.0f}%", (bar_x + bar_width + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        y_offset += 30
    
    y_offset += 20
    
    # Emotion distribution
    cv2.line(panel, (15, y_offset), (width - 15, y_offset), (200, 200, 200), 2)
    y_offset += 30
    
    cv2.putText(panel, "All Emotion Probabilities:", (15, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    y_offset += 25
    
    if all_probs is not None:
        # Normalize probabilities to ensure they sum to 100%
        prob_sum = np.sum(all_probs)
        if prob_sum > 0:
            normalized_probs = all_probs / prob_sum
        else:
            normalized_probs = all_probs
        
        for i, (label, prob) in enumerate(zip(emotion_labels, normalized_probs)):
            # Emotion name
            color = (0, 100, 255) if label == emotion else (100, 100, 100)
            cv2.putText(panel, f"{label}:", (25, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Percentage (now correctly showing 0-100%)
            cv2.putText(panel, f"{prob*100:.1f}%", (110, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Mini bar
            bar_x = 180
            bar_width = 160
            bar_height = 10
            
            cv2.rectangle(panel, (bar_x, y_offset - 8), (bar_x + bar_width, y_offset + 2),
                         (230, 230, 230), -1)
            
            fill = int(prob * bar_width)
            cv2.rectangle(panel, (bar_x, y_offset - 8), (bar_x + fill, y_offset + 2),
                         color, -1)
            
            y_offset += 22
    
    # Timeline at bottom
    y_offset = height - 110
    cv2.line(panel, (15, y_offset), (width - 15, y_offset), (200, 200, 200), 2)
    y_offset += 25
    
    cv2.putText(panel, "Emotion Timeline:", (15, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Draw mini timeline
    timeline_img = visualizer.create_timeline_graph(width - 30, 60)
    panel[y_offset + 10:y_offset + 70, 15:width - 15] = timeline_img
    
    return panel

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        performance_monitor.update()
        frame_counter += 1
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Process emotion detection
        if frame_counter % SKIP_FRAMES == 0 and len(faces) > 0:
            for (x, y, w, h) in faces:
                try:
                    face_roi = frame[y:y + h, x:x + w]
                    
                    result = DeepFace.analyze(face_roi, actions=['emotion'], 
                                            enforce_detection=False, silent=True)
                    
                    emotion_probs = result[0]['emotion']
                    temporal_model.update_history(emotion_probs)
                    
                    emotion, confidence, weighted_probs = temporal_model.predict_temporal_emotion()
                    
                    if emotion and confidence > CONFIDENCE_THRESHOLD:
                        last_emotion = emotion
                        last_confidence = confidence
                        last_probs = weighted_probs
                        
                        visualizer.add_emotion(emotion)
                        logger.log_emotion(emotion, confidence)
                        
                        if SHOW_HEATMAP:
                            frame = explainer.apply_heatmap_overlay(
                                frame, face_roi, emotion, x, y, w, h, alpha=0.3
                            )
                        
                        if SHOW_REGION_BOXES:
                            frame = explainer.draw_region_boxes(
                                frame, emotion, x, y, w, h
                            )
                    
                except Exception as e:
                    print(f"Analysis error: {e}")
                    continue
        
        # Draw face boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw UI
        if last_emotion:
            frame = draw_improved_ui(frame, last_emotion, last_confidence, 
                                    performance_monitor.get_fps())
        
        # Create and attach explanation panel
        if SHOW_EXPLANATION_PANEL and last_emotion:
            explanation_panel = create_explanation_panel(
                panel_width, frame_height, last_emotion, last_confidence,
                last_probs, temporal_model.emotion_labels
            )
            combined_frame = np.hstack([frame, explanation_panel])
        else:
            combined_frame = frame
        
        cv2.imshow('Temporal + Explainable Emotion Detection', combined_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            SHOW_HEATMAP = not SHOW_HEATMAP
            print(f"Heatmap: {'ON' if SHOW_HEATMAP else 'OFF'}")
        elif key == ord('b'):
            SHOW_REGION_BOXES = not SHOW_REGION_BOXES
            print(f"Region boxes: {'ON' if SHOW_REGION_BOXES else 'OFF'}")
        elif key == ord('p'):
            SHOW_EXPLANATION_PANEL = not SHOW_EXPLANATION_PANEL
            print(f"Panel: {'ON' if SHOW_EXPLANATION_PANEL else 'OFF'}")
        elif key == ord('r'):
            temporal_model.reset()
            visualizer = EmotionVisualizer()
            print("Reset!")
        elif key == ord('s'):
            logger.save_log()

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    logger.save_log()
    
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    
    stats = visualizer.get_emotion_statistics()
    if stats:
        print("\nEmotion Distribution:")
        for emotion, data in sorted(stats.items(), key=lambda x: x[1]['percentage'], reverse=True):
            bar = "█" * int(data['percentage'] / 2)
            print(f"  {emotion:10s}: {bar} {data['percentage']:5.1f}%")
    
    print(f"\nAverage FPS: {performance_monitor.get_fps():.1f}")
    print("=" * 70)
    
    cap.release()
    cv2.destroyAllWindows()