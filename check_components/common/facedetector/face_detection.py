import cv2
import torch
import imutils
import numpy as np
import logging
import sys
from .LightFaceDetector.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from .LightFaceDetector.vision.ssd.config.fd_config import define_img_size
from .obj_det_interface import ObjectDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FaceDetection(ObjectDetector):
    def __init__(self, cfg):
        self._cfg = cfg
        self._device = "cpu"
        self._selected_model = ""
        try:
            self._load_model()
        except Exception as e:
            logger.error(e)
            logger.error("Unable to load face detection model!")
            sys.exit()

    def _load_model(self):
        """
         Load face detector models.
        """
        self._selected_model = self._cfg["face_det"]["selector"]
        has_cuda = torch.cuda.is_available()  # check if GPU exists
        if has_cuda:
            self._device = "cuda" if self._cfg["face_det"][self._selected_model]["use_gpu"] else "cpu"

        if self._cfg["setting"]["force_cpu"]:
            self._device = "cpu"

        if self._selected_model == "lfd":
            define_img_size(self._cfg["face_det"]["lfd"]["input_size"])

            lfd_class_names = [name.strip() for name in open(self._cfg["face_det"]["lfd"]["label_path"]).readlines()]
            lfd_net = create_Mb_Tiny_RFB_fd(len(lfd_class_names), is_test=True, device=self._device)
            self._detector = create_Mb_Tiny_RFB_fd_predictor(lfd_net, candidate_size= \
                self._cfg["face_det"]["lfd"]["candidate_size"], device=self._device)
            lfd_net.load(self._cfg["face_det"]["lfd"]["weight_path"])

        elif self._selected_model == "opencv_dnn":
            self._detector = cv2.dnn.readNetFromCaffe(self._cfg["face_det"]["opencv_dnn"]["model_file_path"],
                                                      self._cfg["face_det"]["opencv_dnn"]["weight_path"])
            if self._device == "cuda":
                self._detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self._detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def single_detect(self, cv_image):
        """
         Get the bounding box of faces in image.
        """
        faces = []

        if self._selected_model == "lfd":
            ratio = cv_image.shape[0] / 500.0
            orig = cv_image.copy()
            image = imutils.resize(cv_image, height=500)
            im_h = image.shape[0]
            im_w = image.shape[1]

            if self._device == "gpu":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            boxes, labels, probs = self._detector.predict(image, self._cfg["face_det"]["lfd"]["candidate_size"] / 2,
                                                          self._cfg["face_det"]["lfd"]["threshold"])
            for i in range(len(boxes)):
                box = boxes[i, :]
                boxint = box.numpy().astype(int)
                boxint[boxint < 0] = 0  # avoid negative values
                x1, y1, x2, y2 = boxint[0], boxint[1], boxint[2], boxint[3]
                faces.append([x1/im_w, y1/im_h, x2/im_w, y2/im_h, probs[i]])

        elif self._selected_model == "opencv_dnn":
            rows, cols, _ = cv_image.shape
            confidences = []

            self._detector.setInput(
                cv2.dnn.blobFromImage(
                    cv_image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False
                )
            )
            detections = self._detector.forward()

            for result in detections[0, 0, :, :]:
                confidence = result[2]
                if confidence > self._cfg["face_det"]["opencv_dnn"]["threshold"]:
                    x_left_bottom = result[3]
                    y_left_bottom = result[4]
                    x_right_top = result[5]
                    y_right_top = result[6]
                    faces.append([x_right_top, y_right_top, x_left_bottom, y_left_bottom, confidence])
        return faces

    def plot_boxes(self, cv_image, boxes):
        res_img = cv_image.copy()  # Make a copy so won't overwrite input image
        im_h = res_img.shape[0]
        im_w = res_img.shape[1]
        for box in boxes:
            cv2.rectangle(res_img, (int(box[0] * im_w), int(box[1] * im_h)), (int(box[2] * im_w), int(box[3] * im_h)),
                          (255, 0, 0), 1)

        return res_img

    def get_cropped_face(self, cv_image, res):
        """
        Takes in an image and the results from _detect method.
        Assume there is only one face bounding box
        Checks if face is cut off, or if it is too near the border of the image
        Returns the coordinates of the bounding box.
        """
        im_h = cv_image.shape[0]
        im_w = cv_image.shape[1]

        face_validity = True

        x1, y1, x2, y2 = int(res[0][0]*im_w), int(res[0][1]*im_h), int(res[0][2]*im_w), int(res[0][3]*im_h)
        cropped_image = cv_image[y1+1:y2, x1+1:x2]        # crop to remove bounding box
        crop_w = self._cfg["face_det"]["cropped_face_width"]
        crop_h = self._cfg["face_det"]["cropped_face_height"]
        cropped_image = cv2.resize(cropped_image, (crop_w, crop_h))

        # cv2.imshow("temp",cv_image)
        # cv2.waitKey()

        # Aspect Ratio - Check if bounding box ratio is too thin or wide. Could indicate cropped pictures. Optimal ratio to be determined
        bb_width = x2 - (x1+1)
        bb_height = y2 - (y1+1)
        bb_AR = bb_width/bb_height 
        if bb_AR < self._cfg["face_det"]["min_bbox_ratio"] or bb_AR > self._cfg["face_det"]["max_bbox_ratio"]: # 0.6 and 1.2 are rough estimates
            logger.error("Face partially obstructed")
            face_validity = False

        # Bounding box and border contact 
        if x1 == 0 or x2 == (im_h-1) or y1 == 0 or y2 == (im_w - 1):
            logger.error("Face is too close to the edge")
            face_validity = False
        
        # Ratio of cropped area to image area
        cropped_area = (x2 - x1) * (y2 - y1)
        img_area = im_h * im_w
        ratio = cropped_area / img_area
        return face_validity, cropped_image, ratio

    # Helper function to get the probability of the face detected
    def get_proba(self, face_res):
        return face_res[0][4]


    def detect(self, cv_image, orientation):
        # Function to try and catch wrongly oriented images, based on the proability output of face detector. Do note that
        # some images in wrong orientation can produce better probabilities.
        # Use .numpy() to get more accurate probabilities

        # Keep a cached copy of the results for the original image.
        original_res = self.single_detect(cv_image)
        # Optimal_image and optimal_res are the pointers that will change when a better result is found.
        optimal_image, rotated_image = cv_image, cv_image
        optimal_res = original_res
        # proba is to store the best probability result. 
        proba, p1, p2, p3 = 0, 0, 0, 0
        # First try to catch upside down images
        try:
            p1 = self.get_proba(optimal_res)
            if p1 > proba:
                proba = p1
        except:
            pass

        # If image is in landscape, turn 90 degrees and try detecting face
        if orientation == False:
            rotated_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
            res = self.single_detect(rotated_image)
            try:
                p2 = self.get_proba(res)
                if p2 > proba:
                    optimal_image = rotated_image
                    proba = p2
                    optimal_res = res
            except:
                pass

        # Rotate image by 180 degrees and try detecting face again
        rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_180)
        res = self.single_detect(rotated_image)
        try:
            p3 = self.get_proba(res)
            if p3 > proba:
                optimal_image = rotated_image
                proba = p3
                optimal_res = res
        except:
            pass
        if (optimal_image.shape != cv_image.shape) or (np.any(cv2.subtract(optimal_image,cv_image))):
            logging.error("Input image may be in wrong orientation.")

        # Use original orientation if both original and rotated orientation give high confidence scores. This does not catch
        # the case if original image is wrongly oriented, yet gives a higher probability score. Logic can be better improved.
        if (p1 > 0.99) and (p3 > 0.99):
            return original_res, cv_image
        return optimal_res, optimal_image

