import datetime

import numpy as np
import pyvirtualcam
import cv2

from detectors import hand_detect, face_mesh, face_detect

VIDEO_HEIGHT = 480
VIDEO_WIDTH = 640

MODES = {49: "face_detect", 50: "face_mesh", 51: "hands"}


def add_text(img: np.ndarray, mode: str, proc_time: int) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.8
    font_color = (0, 0, 255)
    font_thickness = 2
    overlay = img.copy()
    output = img.copy()
    cv2.rectangle(overlay, (0, 0), (350, 80), (255, 255, 255), -1)
    overlay = cv2.putText(overlay, f"Mode: {mode}", (5, 20), font, font_size, font_color, font_thickness)
    overlay = cv2.putText(
        overlay,
        f"Processing time: {proc_time} ms",
        (5, 50),
        font,
        font_size,
        font_color,
        font_thickness,
    )
    cv2.addWeighted(overlay, 0.8, output, 0.2, 0, output)
    return output


with pyvirtualcam.Camera(width=VIDEO_WIDTH, height=VIDEO_HEIGHT, fps=60, device="/dev/video2") as cam:
    current_mode = MODES.get(48)
    print(f"Using virtual camera: {cam.device}")
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    prev_frame_time = datetime.datetime.now()
    new_frame_time = datetime.datetime.now()

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue
        image.flags.writeable = False
        new_frame_time = datetime.datetime.now()
        fps = new_frame_time - prev_frame_time
        prev_frame_time = new_frame_time
        image = add_text(image, mode=current_mode, proc_time=int(fps.microseconds / 1000))
        if current_mode == "face_detect":
            image = face_detect(image)
        if current_mode == "hands":
            image = hand_detect(image)
        elif current_mode == "face_mesh":
            image = face_mesh(image)
        cam_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cam_image = cv2.flip(cam_image, 1)
        cv2.imshow("Cam", image)
        cam.send(cam_image)
        cam.sleep_until_next_frame()

        key = cv2.waitKey(5) & 0xFF
        if key != 255:
            current_mode = MODES.get(key, "empty")

    cap.release()
