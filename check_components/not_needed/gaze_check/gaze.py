import cv2
import numpy as np
import math
from ..common.utils.CheckStatus import CheckStatus

from .dlib_utils.calibration import Calibration
from .dlib_utils.eye import Eye
from .dlib_utils.pupil import Pupil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_gaze(cfg, image, landmarks, face_parsing_res):
    status = CheckStatus.STATUS_UNKNOWN.value
    remarks = "No such landmarks."
    # Check for face symmetry using face segmentation results
    if not check_face_segmentation(cfg, face_parsing_res):
        status = CheckStatus.STATUS_ERROR.value
        remarks = "Face is obstructed/tilted or eyes not clearly visible."
        return {"status": status, "remarks": remarks}
    
    if cfg["landmark_det"]["selector"] == "hrnet":
        if cfg["landmark_det"]["hrnet"]["landmark"] == "WFLW": 
            #Check if right eye is blinking
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
            if blinking_ratio < cfg["gaze_check"]["blinking_thresh"]:
                status = CheckStatus.STATUS_FAIL.value
                remarks = "Eyes are closed."
            else:
                #Calculate Horizontal Ratio
                left_eye_center = int((landmarks[60][0] + landmarks[64][0]) /2 ) - landmarks[60][0]
                left_eye_pupil = landmarks[96][0] - landmarks[60][0]
                lefthratio = left_eye_pupil / (left_eye_center * 2)

                right_eye_center = int((landmarks[68][0] + landmarks[72][0]) /2 ) - landmarks[68][0]
                right_eye_pupil = landmarks[97][0] - landmarks[68][0]
                righthratio = right_eye_pupil / (right_eye_center * 2)

                final = (lefthratio + righthratio) / 2
                if cfg["gaze_check"]["left_thresh"] < final < cfg["gaze_check"]["right_thresh"]:
                    status = CheckStatus.STATUS_PASS.value
                    remarks = "Eyes looking center."
                else:
                    status = CheckStatus.STATUS_FAIL.value
                    remarks = "Eyes not looking center."

    elif cfg["landmark_det"]["selector"] == "dlib":
        if cfg["landmark_det"]["dlib"]["landmark"] == "multipie":
            # Instantiate eye objects
            left_eye = Eye(image, landmarks, 0, Calibration())
            right_eye = Eye(image, landmarks, 1, Calibration())
            
            # Check if eyes are blinking
            blinking_ratio = (left_eye.blinking + right_eye.blinking) / 2
            if blinking_ratio < cfg["gaze_check"]["blinking_thresh"]:
                status = CheckStatus.STATUS_FAIL.value
                remarks = "Eyes are closed."
            else:
                # Get coordinates of both left and right pupils for plotting
                # left_pupil_coords = [left_eye.origin[0] + left_eye.pupil.x, left_eye.origin[1] + left_eye.pupil.y]
                # right_pupil_coords = [right_eye.origin[0] + right_eye.pupil.x, right_eye.origin[1] + right_eye.pupil.y]

                # Display position of pupils
                # cv2.imshow("pupils", annotated_frame(image,left_pupil_coords,right_pupil_coords))
                # cv2.waitKey(0)

                # Calculate horizontal ratio
                left_pupil_ratio = left_eye.pupil.x / (left_eye.center[0] * 2 - 4.83)
                right_pupil_ratio = right_eye.pupil.x / (right_eye.center[0] * 2 - 4.83)
                
                final = (left_pupil_ratio + right_pupil_ratio) / 2
                if cfg["gaze_check"]["left_thresh"] < final < cfg["gaze_check"]["right_thresh"]:
                    status = CheckStatus.STATUS_PASS.value
                    remarks = "Eyes looking center."
                else:
                    status = CheckStatus.STATUS_FAIL.value
                    remarks = "Eyes not looking center."

    return {"status": status, "remarks": remarks}

# If user wants to see annotated image of dlib pupils
def annotated_frame(image, p1, p2):
        """Returns the main frame with pupils highlighted.
        Takes in the image to be drawn, and coordinates of pupils p1 and p2
        """
        frame = image.copy()

        color = (0, 255, 255)
        x_left, y_left = p1[0], p1[1]
        x_right, y_right = p2[0], p2[1]
        cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color, 2)
        cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color, 2)
        cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color, 2)
        cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color, 2)
            
        return frame


def check_face_segmentation(cfg, face_parsing_res):
    """
    Takes in config file and face parsing results and returns a boolean. 
    If face is not obstructed, a symmetrical check will be conducted using the bounding boxes of the eyes
    to determine if person is tilted. 
    Obstruction here is defined as the eye having a certain amount of pixels compared against a threshold. 
    """
    Leye_mask = face_parsing_res['res_label_np'] == 4
    Reye_mask = face_parsing_res['res_label_np'] == 5

    Leye_Npixels = np.count_nonzero(Leye_mask)
    Reye_Npixels = np.count_nonzero(Reye_mask)

    pixel_counts = [Leye_Npixels, Reye_Npixels]
    part_mask = [Leye_mask, Reye_mask]
    part_presence = [x > cfg["face_parsing"]["part_thres"] for x in pixel_counts]
    # If both eyes are present, proceed to get bounding boxes
    if all(part_presence):
        midpoints = []
        for part in part_mask:
            if part.size > 0 and part.any():
                contours, hierarchy = cv2.findContours((part * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(contours) != 0:
                    c = max(contours, key=cv2.contourArea)
                    x,y,w,h = cv2.boundingRect(c)
                    # cv2.rectangle(face_parsing_res["res_img_cv"], (x, y), (x + w, y + h), (0, 255, 255), 5)
                    y1 = y
                    y2 = y+h
                    x1 = x
                    x2 = x+w

                    est_mask = part[y1:y2, x1:x2] 
                    midx, midy = (y1 + y2) / 2, (x1 + x2) / 2
                    midpoints.append([midy, midx])       # Storing in (x,y) relative to user and not image
                    coverage = np.count_nonzero(part) / est_mask.size
                else:
                    logging.error("Only 1 contour found.")
                    return False
            else:
                logging.error("Mask size = 0 or no part masks found.")
                return False
        # Symmetry check 
        Leye_midpt, Reye_midpt = midpoints[0], midpoints[1]
        dx = Leye_midpt[0] - Reye_midpt[0]
        dy = Leye_midpt[1] - Reye_midpt[1]
        angle = np.degrees(np.arctan(dy/dx))
        if abs(angle) <= cfg["face_parsing"]["angle_sym"]:  # +- angle_sym
            return True
        else:
            logging.error("Angle between eyes exceeds threshold. Face titled or obstructed. ")
            return False
    else:
        logging.error("At least 1 eye is not detected.")
        return False
