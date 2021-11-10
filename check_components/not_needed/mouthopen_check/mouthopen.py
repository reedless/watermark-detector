import cv2
import numpy as np
import math
from ..common.utils.CheckStatus import CheckStatus
from ..common.facelandmark.utils_landmarks import show_landmarks
from ..common.facelandmark.LandmarkMapping import *

def check_mouthopen(cfg, img, landmarks):
    """
    Checks if mouth is open by getting the distance between the inner lips(mouth height) and 
    the size of the smallest lip(lip height) 
    """
    status = CheckStatus.STATUS_UNKNOWN.value
    remarks = "No such landmarks."
    # For visualisation only
    lmks = get_lip_landmarks(cfg, landmarks)   
    mouthimg = drawLipsPolyline(img, lmks)
    # cv2.imshow("mouth", mouthimg)
    # cv2.waitKey(0)
    mouth_height, top_lip_height, bottom_lip_height = 0, 0, 0
    if cfg["landmark_det"]["selector"] == "hrnet":
        if cfg["landmark_det"]["hrnet"]["landmark"] == "WFLW":
            dist = 0
            for i in [78, 79, 80]:                        
                # 78, 79, 80 and 89, 90, 91 refer to the 3 points in the center of the top lip
                dist += math.sqrt((landmarks[i][0] - landmarks[i+11][0])**2 + 
                                  (landmarks[i][1] - landmarks[i+11][1])**2)
            top_lip_height = dist / 3
            dist = 0
            for i in [84, 85, 86]:
                # 84, 85, 86 and 93, 94, 95 refer to the 3 points in the center of the bottom lip
                dist += math.sqrt((landmarks[i][0] - landmarks[i+9][0])**2 + 
                                  (landmarks[i][1] - landmarks[i+9][1])**2)
            bottom_lip_height = dist / 3

            dist = 0
            j = 3
            for i in [89, 90, 91]:
                # 89, 90, 91 and 95, 94 ,93 refer to the 3 points in the center of the inner lip of top and bottom lips
                dist += math.sqrt((landmarks[i][0] - landmarks[i+j*2][0])**2 +
                                    (landmarks[i][1] - landmarks[i+j*2][1])**2)
                j -= 1

            mouth_height = dist / 3

        else:    
            remarks = f'{cfg["landmark_det"]["selector"]["landmark"]} not supported.'
            status = CheckStatus.STATUS_ERROR.value

    elif cfg["landmark_det"]["selector"] == "dlib":
        if cfg["landmark_det"]["dlib"]["landmark"] == "multipie":
            dist = 0
            for i in [50, 51, 52]:
                dist += math.sqrt((landmarks[i][0] - landmarks[i+11][0])**2 +
                                  (landmarks[i][1] - landmarks[i+11][1])**2)
            top_lip_height = dist / 3
            dist = 0
            for i in [56, 57, 58]:
                dist += math.sqrt((landmarks[i][0] - landmarks[i+9][0])**2 + 
                                  (landmarks[i][1] - landmarks[i+9][1])**2)
            bottom_lip_height = dist /3
            dist = 0
            j = 3
            for i in [61, 62, 63]:
                dist += math.sqrt((landmarks[i][0] - landmarks[i+j*2][0])**2 + 
                                   (landmarks[i][1] - landmarks[i+j*2][1])**2)
                j -= 1
            mouth_height = dist / 3


        else:
            remarks = f'{cfg["landmark_det"]["selector"]["landmark"]} not supported.'
            status = CheckStatus.STATUS_ERROR.value
    else:
        remarks = f'{cfg["landmark_det"]["selector"]} not supported. '
        status = CheckStatus.STATUS_ERROR.value

    # If mouth height is less than min lip height * ratio, check pass
    # First check if lip heights were updated. Mouth height can be 0.
    if top_lip_height != 0 or bottom_lip_height != 0:
        if mouth_height < cfg["mouth_open_check"]["ratio"] * min(top_lip_height, bottom_lip_height):
            remarks = "Mouth is closed."
            status = CheckStatus.STATUS_PASS.value
        else:
            remarks = "Mouth is open."
            status = CheckStatus.STATUS_FAIL.value

    return mouthimg, {"status": status, "remarks": remarks}

def get_lip_landmarks(cfg, landmarks):
    """
    Returns the specific landmarks corresponding to the lips. Index varies across different landmark types.
    This function is more for development and visualisation purposes. 
    """
    if cfg["landmark_det"]["selector"] == "hrnet":
        if cfg["landmark_det"]["hrnet"]["landmark"] == "WFLW":
            top_lip = [i for i in range(76,82+1)] + [i for i in range(88,92+1)]
            bottom_lip = [i for i in range(82,87+1)] + [76, 88, 92, 93, 94, 95] 
            return landmarks[np.array(top_lip + bottom_lip)]

    elif cfg["landmark_det"]["selector"] == "dlib":
        if cfg["landmark_det"]["dlib"]["landmark"] == "multipie":
            top_lip = [i for i in range(48,54+1)] + [i for i in range(60,64+1)]
            bottom_lip = [i for i in range(54,59+1)] + [48, 60, 64, 65, 66, 67]
            return landmarks[np.array((top_lip + bottom_lip))]
        else:
            remarks = " landmark not supported."
            status = CheckStatus.STATUS_ERROR.value
        
    else:
        remarks = "model not supported."
        status = CheckStatus.STATUS_ERROR.value


def drawLipsPolyline(image, lip_landmarks):
    """
    Fills the area corresponding to the lips. For visualisation purposes only.
    2nd argument for fillPoly must be an int32 list. 
    """
    overlay = image.copy()
    cv2.fillPoly(overlay, np.int32([lip_landmarks]), (0, 255, 0))
    cv2.fillPoly(overlay, np.int32([lip_landmarks]), (0, 255, 0))
    alpha = 0.4
    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image_new
