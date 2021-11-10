from .landmark_interface import LandmarkDetector
from .utils_inference import get_lmks_by_img, get_model_by_name, get_preds, decode_preds, crop
from .utils_landmarks import show_landmarks, get_all_landmarks_from_net, get_five_landmarks_from_net, \
    get_lip_landmarks_from_net, alignment_orig, show_img_landmarks, set_circles_on_img
from .LandmarkMapping import wflw, aflw, multipie
import numpy as np
import torch
import cv2
import sys
import logging
import matplotlib.pyplot as plt
import dlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FaceLandmark(LandmarkDetector):
    def __init__(self, cfg):
        self._cfg = cfg
        self._device = "cpu"
        self._selected_model = ""
        self.landmarks = []
        try:
            self._load_model()
        except Exception as e:
            logger.error(e)
            logger.error("Unable to load face landmark model")
            sys.exit()

    def _load_model(self):
        """
        Load face landmark models
        """
        self._selected_model = self._cfg["landmark_det"]["selector"]
        has_cuda = torch.cuda.is_available()  # check if GPU exists

        if has_cuda:
            self._device = "cuda" if self._cfg["landmark_det"][self._selected_model]["use_gpu"] else "cpu"

        if self._cfg["setting"]["force_cpu"]:
            self._device = "cpu"
            
        if self._selected_model == "hrnet":
            self._detector = get_model_by_name(self._cfg["landmark_det"]["hrnet"]["landmark"], device=self._device)

        elif self._selected_model == "dlib":
            model_path = self._cfg["landmark_det"]["dlib"]["model_path"]
            self._detector = dlib.shape_predictor(model_path)

    def return_coords(self, image, selected_points):
        """
        Returns a numpy array of all feature coordinates of face, and numpy array of selected feature coordinates. 
        """
        if self._selected_model == "hrnet":
            self.landmarks = get_lmks_by_img(self._detector, image, dev=self._device)

        elif self._selected_model == "dlib":
            gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            frame = dlib.rectangle(0, 0, image.shape[1], image.shape[0])
            self.landmarks = self._detector(gray_frame, frame)  # save the object for easier display
            self.landmarks = np.array([[self.landmarks.part(i).x, self.landmarks.part(i).y] for i in range(68)])

        specific_landmarks = np.array([])
        if selected_points:  # if there are specific points selected
            specific_landmarks = self.landmarks[np.array(selected_points)]

        return self.landmarks, specific_landmarks

    def feature_add(self, landmark_type, points):  # Points of different landmark types added here
        """
        Add the indices of feature points wanted and returns a list
        """
        feature_list = []
        if self._selected_model == "hrnet":
            if landmark_type == 'WFLW':
                for point in points:
                    if point == "BOTTOM_FACE_CONTOUR":
                        feature_list += wflw.BOTTOM_FACE_CONTOUR.value
                    elif point == "LEFT_EYEBROW":
                        feature_list += wflw.LEFT_EYEBROW.value
                    elif point == "RIGHT_EYEBROW":
                        feature_list += wflw.RIGHT_EYEBROW.value
                    elif point == "NOSE":
                        feature_list += wflw.NOSE.value
                    elif point == "LEFT_EYE":
                        feature_list += wflw.LEFT_EYE.value
                    elif point == "RIGHT_EYE":
                        feature_list += wflw.RIGHT_EYE.value
                    elif point == "OUTER_LIP":
                        feature_list += wflw.OUTER_LIP.value
                    elif point == "INNER_LIP":
                        feature_list += wflw.INNER_LIP.value
                    elif point == "LEFT_PUPIL":
                        feature_list += wflw.LEFT_PUPIL.value
                    elif point == "RIGHT_PUPIL":
                        feature_list += wflw.RIGHT_PUPIL.value
            elif landmark_type == 'AFLW':
                for point in points:
                    if point == "BOTTOM_FACE_CONTOUR":
                        feature_list += aflw.BOTTOM_FACE_CONTOUR.value
                    elif point == "LEFT_EYEBROW":
                        feature_list += aflw.LEFT_EYEBROW.value
                    elif point == "RIGHT_EYEBROW":
                        feature_list += aflw.RIGHT_EYEBROW.value
                    elif point == "NOSE":
                        feature_list += aflw.NOSE.value
                    elif point == "LEFT_EYE":
                        feature_list += aflw.LEFT_EYE.value
                    elif point == "RIGHT_EYE":
                        feature_list += aflw.RIGHT_EYE.value
                    elif point == "OUTER_LIP":
                        feature_list += aflw.OUTER_LIP.value
                    elif point == "INNER_LIP":
                        feature_list += aflw.INNER_LIP.value
                    elif point == "LEFT_PUPIL":
                        feature_list += aflw.LEFT_PUPIL.value
                    elif point == "RIGHT_PUPIL":
                        feature_list += aflw.RIGHT_PUPIL.value

        elif self._selected_model == "dlib":
            if landmark_type == "multipie":
                for point in points:
                    if point == "BOTTOM_FACE_CONTOUR":
                        feature_list += multipie.BOTTOM_FACE_CONTOUR.value
                    elif point == "LEFT_EYEBROW":
                        feature_list += multipie.LEFT_EYEBROW.value
                    elif point == "RIGHT_EYEBROW":
                        feature_list += multipie.RIGHT_EYEBROW.value
                    elif point == "NOSE":
                        feature_list += multipie.NOSE.value
                    elif point == "LEFT_EYE":
                        feature_list += multipie.LEFT_EYE.value
                    elif point == "RIGHT_EYE":
                        feature_list += multipie.RIGHT_EYE.value
                    elif point == "OUTER_LIP":
                        feature_list += multipie.OUTER_LIP.value
                    elif point == "INNER_LIP":
                        feature_list += multipie.INNER_LIP.value
                    elif point == "LEFT_PUPIL":
                        feature_list += multipie.LEFT_PUPIL.value
                    elif point == "RIGHT_PUPIL":
                        feature_list += multipie.RIGHT_PUPIL.value
        return feature_list

    def get_points(self):
        """
        Returns a list of points for all landmarks, and another list of points for the type of landmarks chosen
        """
        if self._selected_model == "hrnet":
            points = self._cfg["landmark_det"]["hrnet"]["features"]["points"]
            custom = self._cfg["landmark_det"]["hrnet"]["features"]["custom"]
            if self._cfg["landmark_det"]["hrnet"]["landmark"] == "WFLW":
                all_points = [i for i in range(98)]
                if not points and not custom:  # If both are None, return all landmarks
                    return all_points, all_points
                elif points and custom:
                    features = self.feature_add("WFLW", points)
                    return all_points, list(set(features + custom))  # Use set to remove duplicate indices
                elif points is not None:
                    return all_points, self.feature_add("WFLW", points)
                else:
                    return all_points, custom

            if self._cfg["landmark_det"]["hrnet"]["landmark"] == "AFLW":
                all_points = [i for i in range(21)]
                if not points and not custom:
                    return all_points, all_points
                elif points and custom:
                    features = self.feature_add("AFLW", points)
                    return all_points, list(set(features + custom))
                elif points is not None:
                    return all_points, self.feature_add("AFLW", points)
                else:
                    return all_points, custom

        if self._selected_model == "dlib":
            all_points = [i for i in range(68)]
            points = self._cfg["landmark_det"]["dlib"]["features"]["points"]
            custom = self._cfg["landmark_det"]["dlib"]["features"]["custom"]
            if not points and not custom:
                return all_points, all_points
            elif points and custom:
                features = self.feature_add("multipie", points)
                return all_points, list(set(features + custom))
            elif points is not None:
                return all_points, self.feature_add("multipie", points)
            else:
                return all_points, custom

    def show_landmarks(self, image, landmarks, display=0):
        """
        Displays the annotated image if display = 1, and returns an image
        """
        img = set_circles_on_img(image, landmarks, is_copy=True)
        if display == 1:
            plt.imshow(img)
            plt.show()
        return img
