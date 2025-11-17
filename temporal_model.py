import numpy as np
from collections import deque

class TemporalEmotionModel:
    """
    Temporal model that considers emotion history for more stable 
    and context-aware predictions using weighted averaging
    """
    
    def __init__(self, sequence_length=30, num_emotions=7):
        """
        Args:
            sequence_length: Number of past frames to consider (30 frames ≈ 1 second at 30fps)
            num_emotions: Number of emotion classes
        """
        self.sequence_length = sequence_length
        self.num_emotions = num_emotions
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Store emotion probabilities over time
        self.emotion_history = deque(maxlen=sequence_length)
        
    def update_history(self, emotion_probs):
        """
        Add new emotion probabilities to history
        
        Args:
            emotion_probs: Dictionary of emotion probabilities from DeepFace (already in percentage form)
        """
        # Convert emotion dict to probability vector (normalize from percentage to 0-1)
        prob_vector = np.array([
            emotion_probs.get(emotion, 0) / 100.0 for emotion in self.emotion_labels
        ])
        
        self.emotion_history.append(prob_vector)
    
    def predict_temporal_emotion(self):
        """
        Predict emotion considering temporal context using exponential weighted average
        Recent frames have more weight than older frames
        
        Returns:
            tuple: (emotion_label, confidence, all_probabilities)
        """
        # If not enough history, return most recent
        if len(self.emotion_history) < 5:
            if len(self.emotion_history) == 0:
                return None, 0, None
            recent_probs = self.emotion_history[-1]
            emotion_idx = np.argmax(recent_probs)
            return self.emotion_labels[emotion_idx], recent_probs[emotion_idx], recent_probs
        
        # Use exponential weighted average with recent frames having more weight
        weights = np.exp(np.linspace(-2, 0, len(self.emotion_history)))
        weights = weights / weights.sum()
        
        weighted_probs = np.zeros(self.num_emotions)
        for i, probs in enumerate(self.emotion_history):
            weighted_probs += probs * weights[i]
        
        # Get prediction
        emotion_idx = np.argmax(weighted_probs)
        confidence = weighted_probs[emotion_idx]
        
        return self.emotion_labels[emotion_idx], confidence, weighted_probs
    
    def get_emotion_trend(self):
        """
        Analyze emotion trends over the sequence
        
        Returns:
            dict: Emotion trends and transitions
        """
        if len(self.emotion_history) < 2:
            return None
        
        # Get dominant emotions over time
        emotions_over_time = []
        for probs in self.emotion_history:
            emotion_idx = np.argmax(probs)
            emotions_over_time.append(self.emotion_labels[emotion_idx])
        
        # Detect transitions
        transitions = []
        for i in range(1, len(emotions_over_time)):
            if emotions_over_time[i] != emotions_over_time[i-1]:
                transitions.append({
                    'from': emotions_over_time[i-1],
                    'to': emotions_over_time[i],
                    'frame': i
                })
        
        return {
            'current_sequence': emotions_over_time,
            'transitions': transitions,
            'stability': 1 - (len(transitions) / len(emotions_over_time)) if len(emotions_over_time) > 0 else 0
        }
    
    def reset(self):
        """Reset emotion history"""
        self.emotion_history.clear()