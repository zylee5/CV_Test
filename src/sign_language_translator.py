import cv2
import numpy as np
import torch
import constants
import streamlit as st
import joblib
from streamlit_webrtc import VideoProcessorBase
from my_model import MyModel
from feature_extractor import FeatureExtractor
from utils import draw_landmarks
from mock_model import MockModel

USE_MOCK_MODEL = False


# Run once when app starts
@st.cache_resource(show_spinner=False)
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = None

    if USE_MOCK_MODEL:
        model = MockModel()
        model.to(device)
        model.eval()
    else:
        try:
            model = MyModel()
            state_dict = torch.load(constants.WEIGHT_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            scaler = joblib.load(constants.SCALER_PATH)

            print("Model and scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading resources: {e}")
            model = None
            scaler = None

    return model, scaler, device


cached_model, cached_scaler, cached_device = load_resources()


class SignLanguageTranslator(VideoProcessorBase):
    def __init__(self):
        self.device = cached_device
        self.model = cached_model
        self.scaler = cached_scaler

        self.extractor = FeatureExtractor()
        self.frame_sequence = []
        self.predictions = []  # Predicted class indices
        self.sentence = []  # Predicted glosses for display
        self.missing_frames = 0

    def __del__(self):
        # Clean up MediaPipe
        if hasattr(self, "extractor"):
            self.extractor.close()

    # Process and return each frame
    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")

        # --- Extract keypoints ---
        landmarks_results, keypoints = self.extractor.process_frame(img_bgr)

        # --- Visualize landmarks ---
        img_bgr.flags.writeable = True
        draw_landmarks(img_bgr, landmarks_results)

        hands_strictly_present = landmarks_results.left_hand_landmarks or landmarks_results.right_hand_landmarks

        if hands_strictly_present:
            self.missing_frames = 0
        else:
            self.missing_frames += 1

        # This keeps the model running even if hands drop for a split second
        hands_active = hands_strictly_present or (self.missing_frames < 5 and len(self.frame_sequence) > 0)

        # Yellow if in grace period, green if strictly present
        if hands_strictly_present:
            status_color = (0, 255, 0)  # Green
            status_text = "Hands: YES"
        elif hands_active:
            status_color = (0, 255, 255)  # Yellow
            status_text = "Hands: LOST (Holding...)"
        else:
            status_color = (0, 0, 255)  # Red
            status_text = "Hands: NO"

        cv2.putText(img_bgr, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        if hands_active:
            # If hands are strictly present, use new keypoints
            # If in grace period (hands lost but < 5 frames), reuse last keypoints
            if not hands_strictly_present and len(self.frame_sequence) > 0:
                keypoints = self.frame_sequence[-1]

            self.frame_sequence.append(keypoints)
            self.frame_sequence = self.frame_sequence[-constants.FRAME_SEQUENCE_LENGTH:]

            if len(self.frame_sequence) == constants.FRAME_SEQUENCE_LENGTH:
                if self.model and self.scaler:
                    sequence_array = np.array(self.frame_sequence)
                    sequence_scaled = self.scaler.transform(sequence_array)
                    input_tensor = torch.tensor(sequence_scaled).float().unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        logits = self.model(input_tensor)

                    probs = torch.softmax(logits, dim=1)

                    # --- Top 5 predictions ---
                    top5_probs, top5_indices = torch.topk(probs, 5)
                    top5_probs = top5_probs.cpu().numpy()[0]
                    top5_indices = top5_indices.cpu().numpy()[0]

                    cv2.putText(img_bgr, "Top 5 Predictions:", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    for i in range(len(top5_indices)):
                        gloss_idx = top5_indices[i]
                        prob_score = top5_probs[i]
                        try:
                            gloss_name = list(constants.TRAINED_GLOSSES)[gloss_idx]
                        except IndexError:
                            gloss_name = "Unknown"

                        text = f"{i + 1}. {gloss_name}: {prob_score * 100:.1f}%"
                        color = (0, 255, 0) if i == 0 else (0, 165, 255)
                        cv2.putText(img_bgr, text, (10, 110 + (i * 25)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    predicted_prob_tensor, predicted_class_tensor = torch.max(probs, dim=1)
                    predicted_prob = predicted_prob_tensor.item()
                    predicted_class_idx = predicted_class_tensor.item()

                    self.predictions.append(predicted_class_idx)
                    if len(self.predictions) > 15:
                        self.predictions = self.predictions[-15:]

                    last_n_predictions = self.predictions[-5:]
                    if len(last_n_predictions) == 5:
                        unique_classes, counts = np.unique(last_n_predictions, return_counts=True)
                        best_class_idx = unique_classes[np.argmax(counts)]
                        count = counts[np.argmax(counts)]

                        if count >= 4 and predicted_prob > constants.PREDICTION_THRESHOLD:
                            confirmed_gloss = list(constants.TRAINED_GLOSSES)[best_class_idx]

                            if len(self.sentence) > 0:
                                if confirmed_gloss != self.sentence[-1]:
                                    self.sentence.append(confirmed_gloss)
                            else:
                                self.sentence.append(confirmed_gloss)

        else:
            # Only reset if we have truly lost hands for > 5 frames
            if len(self.frame_sequence) > 0:
                self.frame_sequence = []
                self.predictions = []

        if len(self.sentence) > 5:
            self.sentence = self.sentence[-5:]

        height, width, _ = img_bgr.shape

        text_to_show = " ".join(self.sentence)
        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = max(0.5, width / 1000.0)
        thickness = max(1, int(font_scale * 2))

        (text_w, text_h), _ = cv2.getTextSize(text_to_show, font, font_scale, thickness)

        text_x = int((width - text_w) / 2)

        bar_height = int(text_h * 2.5)
        cv2.rectangle(img_bgr, (0, height - bar_height), (width, height), (0, 0, 0), -1)

        text_y = int(height - (bar_height / 2) + (text_h / 2))

        cv2.putText(img_bgr, text_to_show, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return frame.from_ndarray(img_bgr, format="bgr24")