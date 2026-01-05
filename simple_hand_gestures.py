import cv2
import mediapipe as mp
import numpy as np
import time
import urllib.request
from pathlib import Path

# Paths and constants
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "gesture_recognizer.task"


class GestureDrawer:
    def __init__(self):
        self._download_model_if_needed()

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionRunningMode.VIDEO,
        )
        self.recognizer = GestureRecognizer.create_from_options(options)

        self.canvas = None  # Drawing layer
        self.last_points = {}  # Per-hand last index fingertip pixel coords
        self.last_gestures = {}  # Per-hand last printed gesture
        self.palette_radius = 40
        self.palette_colors = [
            (0, 0, 255),      # Red
            (0, 165, 255),    # Orange
            (0, 255, 255),    # Yellow
            (0, 255, 0),      # Green
            (255, 255, 0),    # Cyan
            (255, 0, 0),      # Blue
            (255, 0, 255),    # Magenta
        ]
        self.current_color = (0, 255, 255)  # default yellow (BGR)

    def _download_model_if_needed(self):
        if MODEL_PATH.exists():
            return
        print("Downloading gesture model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")

    def _palette_points(self, width):
        # Evenly span the top of the frame with large dots
        count = len(self.palette_colors)
        spacing = width // (count + 1)
        y = 50
        dots = []
        for i, color in enumerate(self.palette_colors):
            x = (i + 1) * spacing
            dots.append({"pos": (x, y), "color": color})
        return dots

    def _check_palette(self, palette, px, py):
        r2 = self.palette_radius ** 2
        for dot in palette:
            dx = px - dot["pos"][0]
            dy = py - dot["pos"][1]
            if dx * dx + dy * dy <= r2:
                return dot["color"]
        return None

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("\nHand Gesture Detection (MediaPipe Task)")
        print("Pointing draws; open palm clears; press 'q' to quit.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            if self.canvas is None or self.canvas.shape != frame.shape:
                self.canvas = np.zeros_like(frame)
                self.last_points.clear()

            # Draw palette spanning the width
            palette = self._palette_points(w)
            for dot in palette:
                cv2.circle(frame, dot["pos"], self.palette_radius, dot["color"], -1)
                cv2.circle(frame, dot["pos"], self.palette_radius, (255, 255, 255), 2)

            # Convert to MediaPipe Image (RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)

            if result and result.gestures:
                for i, classification_list in enumerate(result.gestures):
                    if not classification_list:
                        continue
                    top_class = classification_list[0]
                    gesture_name = top_class.category_name
                    score = top_class.score

                    handedness = result.handedness[i][0].category_name if result.handedness else "Unknown"
                    landmarks = result.hand_landmarks[i]

                    # Gesture change logging
                    current = f"{gesture_name} ({handedness})"
                    if self.last_gestures.get(handedness) != current:
                        print(f"Detected: {current} score={score:.2f}")
                        self.last_gestures[handedness] = current

                    # Map index fingertip to pixels
                    index_tip = landmarks[8]
                    px = int(index_tip.x * w)
                    py = int(index_tip.y * h)

                    # Drawing when pointing
                    if gesture_name == "Pointing_Up":
                        selected = self._check_palette(palette, px, py)
                        if selected is not None:
                            self.current_color = selected
                            self.last_points[handedness] = None
                        else:
                            prev = self.last_points.get(handedness)
                            if prev:
                                cv2.line(self.canvas, prev, (px, py), self.current_color, 4)
                            else:
                                cv2.circle(self.canvas, (px, py), 4, self.current_color, -1)
                            self.last_points[handedness] = (px, py)
                    else:
                        self.last_points.pop(handedness, None)

                    # Clear canvas on open palm
                    if gesture_name == "Open_Palm":
                        self.canvas[:] = 0
                        self.last_points.clear()
                        print("Canvas cleared")

                    # Overlay text near wrist
                    wrist = landmarks[0]
                    tx = int(wrist.x * w) - 50
                    ty = int(wrist.y * h) - 30
                    cv2.putText(
                        frame,
                        f"{handedness}: {gesture_name}",
                        (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
            else:
                cv2.putText(
                    frame,
                    "No hands detected",
                    (w // 2 - 120, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # Combine frame with drawing canvas
            display_frame = cv2.addWeighted(frame, 0.8, self.canvas, 1.0, 0)
            cv2.imshow("Hand Gesture Recognition", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\nGesture detection stopped.")


def main():
    app = GestureDrawer()
    app.run()


if __name__ == "__main__":
    main()
