import warnings
import logging
import cv2
import yaml
import base64
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from check_components.common.utils.CheckStatus import CheckStatus
from check_components.common.utils.CheckConfig import CheckConfig
from check_components.common.facedetector.face_detection import FaceDetection
from check_components.common.facesegment.foreground_segment import ForegroundSegmentation
from check_components.common.facesegment.face_parsing import FaceParsing
from check_components.common.facesegment.specular_highlight_segment import SpecularHighlightSegmentation
from check_components.filetype_check.filetype import check_file_type
from check_components.background_check.background import check_background
from check_components.watermark_check.watermark import check_watermark

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def image_resize_fixed_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize image while maintaining aspect ratio
    Either width or height must be specified
    """

    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # target width is specified
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


class Check(object):
    def __init__(self, cfg_filepath):
        self._cfg = None
        self.input_im = None
        self.im_h, self.im_w, self.im_c = 0, 0, 0

        # DL/ML Modules
        self._face_detector = None
        self._face_landmark_detector = None
        self._face_foreground_segmentor = None
        self._face_parsing_segmentor = None
        self._face_highlight_segmentor = None
        self._shoulder_alignment_classifier = None
        self._pixelation_classifier = None

        # Results from common components
        self.orientation = True
        self.face_boxes_res = None
        self.face_abs_width_height = list()
        self.cropped_ratio = None
        self.face_location_validity = True
        self.cropped_face = None
        self.face_landmarks_res = None
        self.face_landmarks_points = None
        self.face_fgbg_res = None
        self.face_parsing_res = dict()
        self.face_highlight_res = None
        self.shoulder_segment_res = None

        _INVALID_REMARKS = "Unable to perform check."
        _UNAVAIL_REMARKS = "Check not available yet"
        # Results from all checks
        self.check_results = {
            'face_presence_check': {
                "status": CheckStatus.STATUS_UNKNOWN,  # Face presence check
                "remarks": _INVALID_REMARKS},
            'file_type_check': {
                "status": CheckStatus.STATUS_UNKNOWN,
                "remarks": _INVALID_REMARKS},
            'background_check': {
                "status": CheckStatus.STATUS_UNKNOWN,
                "remarks": _INVALID_REMARKS},
            'watermark_check': {
                "status": CheckStatus.STATUS_UNKNOWN,
                "remarks": _INVALID_REMARKS},
        }

        # TODO handle case when error reading config file
        with open(cfg_filepath, "r") as ymlfile:
            self._cfg = yaml.safe_load(ymlfile)

            # TODO handle case when config inputs are wrong
            CheckConfig(self._cfg)

            logger.info("Loading common modules...")
            self._face_detector = FaceDetection(self._cfg)
            # self._face_landmark_detector = FaceLandmark(self._cfg)
            self._face_foreground_segmentor = ForegroundSegmentation(self._cfg)
            self._face_parsing_segmentor = FaceParsing(self._cfg)
            self._face_highlight_segmentor = SpecularHighlightSegmentation(self._cfg)
            logger.info("Common modules successfully loaded.")

            logger.info("Loading check specific models")
            # self._shoulder_alignment_classifier = ShoulderAlignmentClassifier(self._cfg)
            # self._pixelation_classifier = PixelationClassifier(self._cfg)
            logger.info("Check specific models successfully loaded")

    def _process_common(self):
        """
         Preprocess image with common components
        """

        if self._cfg["setting"]["display_image"]:
            cv2.imshow("Input Image", self.input_im)
            # cv2.waitKey(0)

        # Run face detection
        logging.info("[Common Modules] Face detection")
        # self.face_boxes_res = self._face_detector.detect(self.input_im)
        self.face_boxes_res, self.input_im = self._face_detector.detect(self.input_im, self.orientation)

        det_face_im = self._face_detector.plot_boxes(self.input_im, self.face_boxes_res)

        # Make sure there is only single face detected, otherwise skip all checks
        # if len(self.face_boxes_res) == 1:
        if True:
            remark_msg = "Single face is detected."

            self.face_location_validity, self.cropped_face, self.cropped_ratio = self._face_detector.get_cropped_face(
                self.input_im,
                self.face_boxes_res)

            im_h = self.input_im.shape[0]
            im_w = self.input_im.shape[1]
            bb = self.face_boxes_res[0]

            x1, y1, x2, y2 = int(bb[0] * im_w), int(bb[1] * im_h), int(bb[2] * im_w), int(bb[3] * im_h)

            # Detect face only occupies small region of image
            face_w = (x2-x1)
            face_h = (y2-y1)
            if (face_w * face_h)/(im_w * im_h) < 0.1:
                margin = int((face_w + face_h) * 0.25)  # margin to crop is average of face width and height
                y1_new = max(0, y1 + 1 - margin)
                y2_new = min(y2 + margin, im_h-1)
                x1_new = max(0, x1 + 1 - margin)
                x2_new = min(x2 + margin, im_w-1)
                cropped_face_margin = self.input_im[y1_new:y2_new, x1_new:x2_new]

                cropped_face_margin = image_resize_fixed_ratio(cropped_face_margin,
                                                               width=self._cfg["image_size_check"]["im_width"])

                if self._cfg["setting"]["display_image"]:
                    cv2.imshow("Cropped_face_margin", cropped_face_margin)
                    # cv2.waitKey(0)

                # Recompute bounding box
                self.input_im = cropped_face_margin
                self.face_boxes_res, self.input_im = self._face_detector.detect(self.input_im, True)

        # else:
        #     if len(self.face_boxes_res) == 0:
        #         remark_msg = "No face detected."
        #     elif len(self.face_boxes_res) > 1:
        #         remark_msg = "More than 1 face detected."

        #     self.check_results['face_presence_check']['status'] = CheckStatus.STATUS_FAIL.value
        #     self.check_results['face_presence_check']['remarks'] = remark_msg
        #     logger.error(remark_msg)
        #     return -1

        # Make sure face is not too small or too big
        if not (self._cfg["face_det"]["min_face_ratio"] < self.cropped_ratio
                < self._cfg["face_det"]["max_face_ratio"]):
            remark_msg = remark_msg + " Warning: face size may be too large or small w.r.t. whole image."
            logger.warning(remark_msg)

        self.check_results['face_presence_check']['status'] = CheckStatus.STATUS_PASS.value
        self.check_results['face_presence_check']['remarks'] = remark_msg

        # Get the face bounding box
        self.im_h, self.im_w, self.im_c = self.input_im.shape
        box = self.face_boxes_res[0][:-1]
        bb = [int(box[0] * self.im_w), int(box[1] * self.im_h), int(box[2] * self.im_w), int(box[3] * self.im_h)]
        self.face_abs_width_height = [bb[2] - bb[0], bb[3] - bb[1]]

        # Run foreground/background segmentation
        logging.info("[Common Modules] Face foreground segmentation")
        self.face_fgbg_res = self._face_foreground_segmentor.segment(self.input_im)

        if self._cfg["setting"]["display_image"]:
            cv2.imshow("[Common] Face Foreground", self.face_fgbg_res)
            # cv2.waitKey(0)

        # Run face specular highlight detection
        logging.info("[Common Modules] Face specular highlight segmentation")
        self.face_highlight_res = self._face_highlight_segmentor.segment(self.input_im)

        if self._cfg["setting"]["display_image"]:
            cv2.imshow("[Common] Specular Highlight", self.face_highlight_res["res_img_cv"])
            # cv2.waitKey(0)

    def _check_file_type(self, base64_img_str: str):
        """
         Perform compliance check 1: file type , only png, jpg, and jpeg are allowed
         If file type is correct and image can be read, return cv_image as OpenCV image, otherwise as None
        """
        cv_image, filetype_result = check_file_type(base64_img_str)

        if cv_image is not None:
            self.input_im = cv_image  # Set input image

        return filetype_result

    def _check_background(self):
        """
        Perform compliance check x: background check
        """
        return check_background(self._cfg, self.input_im, self.face_fgbg_res)

    def _check_watermark(self):
        """
        Perform compliance check x: watermark check
        """

        status_remarks, processed_img =  check_watermark(self._cfg, 
                                                  self.input_im, 
                                                  self.face_fgbg_res, 
                                                  self.check_results['background_check'], 
                                                  self.face_highlight_res)

        if self._cfg["setting"]["display_image"]:
                cv2.imshow("Watermark Image", processed_img)

        return status_remarks, processed_img


    def process(self, payload: str):
        """
         Perform common modules processing and subsequent individual checks

         Attributes:
            file: fastAPI File class object that has properties 'filename' and 'content_type'
            contents: file contents in bytes
        """

        if payload is None:
            # raise HTTPException(status_code=422, detail="Request payload is empty")
            raise ValueError('Payload is empty')

        logging.info("Performing file type check ...")
        self.check_results['file_type_check'] = self._check_file_type(payload)

        # Ensure image is readable before other checks
        if self.check_results['file_type_check']['status'] == CheckStatus.STATUS_PASS.value:
            # logging.info("Performing image size check ...")
            # self.check_results['image_size_check'], self.orientation = self._check_image_size()  # Check image size

            # Perform common modules processing
            self._process_common()

            # Proceed other checks only when there is only one face detected
            if self.check_results['face_presence_check']['status'] == CheckStatus.STATUS_PASS.value:

                logging.info("Performing background check ...")
                self.check_results['background_check'] = self._check_background()

                logging.info("Performing watermark check ...")
                self.check_results['watermark_check'], processed_img = self._check_watermark()

        else:
            logger.error("File type error or image not readable.")

        # post_processed_result = self._post_process()

        if self._cfg["setting"]["display_image"]:
            cv2.waitKey(0)
            # cv2.destroyAllWindows()

        return self.check_results, self.input_im, processed_img, self.face_fgbg_res, self.face_highlight_res["res_img_cv"]

if __name__ == '__main__':
    checkMain = Check('app_config.yml')
    for i in tqdm(os.listdir('dataset/benchmarkv2')):
        if i[-15:] == 'Zone.Identifier':
            continue
    # i = 'clean_Img_00196.jpg'

        with open(f'dataset/benchmarkv2/{i}', "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())

        check_results, input_img, processed_img, face_fgbg_res, face_highlight_res = checkMain.process(encoded_string)

        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        ax[0][0].imshow(processed_img)

        ax[0][1].imshow(input_img[:, :, ::-1])

        ax[1][0].imshow(face_fgbg_res)
        ax[1][1].imshow(face_highlight_res)

        if not os.path.exists('./output/benchmarkv2'):
            os.mkdir('./output/benchmarkv2')
        os.makedirs('./output/benchmarkv2/', exist_ok=True)
        plt.savefig(f'./output/benchmarkv2/{i}')
        plt.close()

        for key in check_results.keys():
            print(key, check_results[key])