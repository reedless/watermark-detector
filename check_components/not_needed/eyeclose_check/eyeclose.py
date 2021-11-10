import cv2
import numpy as np
import math
from ..common.utils.CheckStatus import CheckStatus

from ..gaze_check.dlib_utils.calibration import Calibration
from ..gaze_check.dlib_utils.eye import Eye
from ..gaze_check.dlib_utils.pupil import Pupil

def check_eyeclose(cfg, landmarks):
    """
    Check if eyes are closed by using landmarks to calculate the blinking ratio.
    """
    status = CheckStatus.STATUS_UNKNOWN.value
    remarks = "No such landmarks."
    blinking_ratio = 0
    if cfg["landmark_det"]["selector"] == "hrnet":
        if cfg["landmark_det"]["hrnet"]["landmark"] == "WFLW":
            # Check if right eye is blinking
            top = [landmarks[62][0], landmarks[62][1]]
            bottom = [landmarks[66][0], landmarks[66][1]]
            left = [landmarks[60][0], landmarks[60][1]]
            right = [landmarks[64][0], landmarks[64][1]]

            eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
            eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

            right_blinking_ratio = eye_height/eye_width

            #Check if left eye is blinking
            top = [landmarks[70][0], landmarks[70][1]]
            bottom = [landmarks[74][0], landmarks[74][1]]
            left = [landmarks[68][0], landmarks[68][1]]
            right = [landmarks[72][0], landmarks[72][1]]

            eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
            eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

            left_blinking_ratio = eye_height/eye_width
            blinking_ratio = (left_blinking_ratio + right_blinking_ratio) / 2
    
    elif cfg["landmark_det"]["selector"] == "dlib":
        if cfg["landmark_det"]["selector"] == "multipie":
            # Instantiate eye objects
            left_eye = Eye(image, landmarks, 0, Calibration())
            right_eye = Eye(image, landmarks, 1, Calibration())
    
            blinking_ratio = (left_eye.blinking + right_eye.blinking) / 2

    if blinking_ratio < cfg["gaze_check"]["blinking_thresh"]:
        status = CheckStatus.STATUS_FAIL.value
        remarks = "Eyes are closed."
    else: 
        status = CheckStatus.STATUS_PASS.value
        remarks = "Eyes are open."

    return {"status": status, "remarks": remarks}