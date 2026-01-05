import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pickle
import os

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"Warning: MediaPipe not properly installed or incompatible version: {e}")
    print("Please install with: pip3 install mediapipe")
    MEDIAPIPE_AVAILABLE = False
    mp_hands = None
    mp_drawing = None

class HandGestureModel:
    def __init__(self):
        self.model = None
        self.gesture_names = ['thumbs_up', 'peace', 'fist', 'open_palm', 'pointing']
        
    def create_model(self, input_shape=(21, 3)):
        """Create a neural network for gesture classification"""
        model = models.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(len(self.gesture_names), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def collect_training_data(self, gesture_id, num_samples=100):
        """Collect hand landmark data for training"""
        if not MEDIAPIPE_AVAILABLE:
            print("Error: MediaPipe is not available")
            return np.array([]), np.array([])
        
        cap = cv2.VideoCapture(0)
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        samples = []
        count = 0
        
        print(f"\nCollecting data for gesture: {self.gesture_names[gesture_id]}")
        print(f"Show your gesture to the camera. Collecting {num_samples} samples...")
        print("Press 'q' to stop early")
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmarks
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
                    
                    samples.append(landmarks)
                    count += 1
                    
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
            
            # Display progress
            cv2.putText(frame, f"Gesture: {self.gesture_names[gesture_id]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {count}/{num_samples}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Collect Training Data', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        
        return np.array(samples), np.full(len(samples), gesture_id)
    
    def train_model(self, X, y, epochs=50):
        """Train the model on collected data"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("\nTraining model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest accuracy: {test_acc * 100:.2f}%")
        
        return history
    
    def save_model(self, filepath='hand_gesture_model.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='hand_gesture_model.h5'):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict_gesture(self, landmarks):
        """Predict gesture from hand landmarks"""
        landmarks_array = np.array(landmarks).reshape(1, 21, 3)
        prediction = self.model.predict(landmarks_array, verbose=0)
        gesture_id = np.argmax(prediction)
        confidence = np.max(prediction)
        return gesture_id, confidence
    
    def run_live_detection(self):
        """Run real-time gesture detection"""
        if self.model is None:
            print("Error: Model not loaded. Train or load a model first.")
            return
        
        if not MEDIAPIPE_AVAILABLE:
            print("Error: MediaPipe is not available")
            return
        
        cap = cv2.VideoCapture(0)
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        print("\nStarting live detection...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmarks
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
                    
                    # Predict gesture
                    gesture_id, confidence = self.predict_gesture(landmarks)
                    gesture_name = self.gesture_names[gesture_id]
                    
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Display prediction
                    cv2.putText(frame, f"Gesture: {gesture_name}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Hand Gesture Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


def main():
    gesture_model = HandGestureModel()
    
    print("Hand Gesture Recognition System")
    print("================================")
    print("\nOptions:")
    print("1. Collect data and train new model")
    print("2. Load existing model and test")
    
    choice = input("\nEnter your choice (1 or 2): ")
    
    if choice == '1':
        # Create model
        gesture_model.create_model()
        
        # Collect data for each gesture
        all_X = []
        all_y = []
        
        samples_per_gesture = int(input("\nHow many samples per gesture? (recommended: 100-200): "))
        
        for i, gesture_name in enumerate(gesture_model.gesture_names):
            input(f"\nPress Enter to start collecting data for '{gesture_name}'...")
            X, y = gesture_model.collect_training_data(i, num_samples=samples_per_gesture)
            all_X.append(X)
            all_y.append(y)
        
        # Combine all data
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        
        print(f"\nTotal samples collected: {len(X)}")
        
        # Train model
        gesture_model.train_model(X, y, epochs=50)
        
        # Save model
        gesture_model.save_model()
        
        # Test live
        test_live = input("\nTest the model live? (y/n): ")
        if test_live.lower() == 'y':
            gesture_model.run_live_detection()
    
    elif choice == '2':
        # Load existing model
        if os.path.exists('hand_gesture_model.h5'):
            gesture_model.load_model()
            gesture_model.run_live_detection()
        else:
            print("No saved model found. Please train a model first (option 1).")
    
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
