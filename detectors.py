import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.hands import Hands

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection


def hand_detect(im: np.ndarray) -> np.ndarray:
    with Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        im_color = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        results = hands.process(im_color)

        # Draw the hand annotations on the image.
        im_color = cv2.cvtColor(im_color, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    im_color,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
            return im_color
        return im


def face_detect(im: np.ndarray) -> np.ndarray:
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1) as face_detection:
        results = face_detection.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if results.detections:
            annotated_image = im.copy()
            for detection in results.detections:
                mp_drawing.draw_detection(annotated_image, detection)
            return annotated_image
        return im


def face_mesh(im: np.ndarray) -> np.ndarray:
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            annotated_image = im.copy()
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )
            return annotated_image
        return im
