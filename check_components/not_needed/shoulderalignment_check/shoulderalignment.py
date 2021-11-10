import numpy as np
import cv2
import yaml
import sys
import pickle
import logging
#from sklearn.ensemble import RandomForestClassifier
from src.check_components.common.facesegment.shoulder_segment import ShoulderSegmentation
from ..common.utils.CheckStatus import CheckStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def preprocess(seg_image):
    """
    Takes in a segmented image and returns a vector consisting of the number of nonzero pixels columnwise, then rowwise.  
    """
    seg_image = cv2.resize(seg_image, (512, 512)) 
    ret, seg_image = cv2.threshold(seg_image, 127, 255, cv2.THRESH_BINARY)
    norm_img = cv2.normalize(seg_image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    fixed_img = cv2.resize(norm_img, (400, 120))
    processed_img = fixed_img[:, :, 0]
    horizontal = np.count_nonzero(processed_img == 1, axis=0)
    vertical = np.count_nonzero(processed_img == 1, axis=1)
    # diff is used for padding of the vectors so that they are all of the same length
    diff = len(horizontal) - len(vertical)
    if diff > 0:
        vertical = np.concatenate((np.zeros(diff), vertical))
    elif diff < 0:
        vertical = vertical[abs(diff):]

    finalvec = np.concatenate((horizontal, vertical), axis=None)
    finalvec = np.add.reduceat(finalvec, np.arange(0, len(finalvec), 24))
    return finalvec


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


def get_slope(p):
    eps = 1e-5  # avoid divide by zero
    slope = (p[3] - p[1]) / (p[2] - p[0] + eps)
    return slope


def shoulder_area(p, im_height):
    b1 = max(1, ((im_height-1) - p[1]))
    b2 = max(0, ((im_height-1) - p[3]))
    h = p[2] - p[0]
    area = 0.5*h*(b1+b2)

    return area


class ShoulderAlignmentClassifier(object):
    def __init__(self, cfg):
        self._cfg = cfg
        self._classifier = None
        self._shoulder_segmentor = None
        self._load_model()

    def _load_model(self):
        shoulder_model = self._cfg["shoulder_alignment_check"]["selector"]
        model_path = self._cfg['shoulder_alignment_check'][shoulder_model]['model_path']
        if shoulder_model == "shoulder_seg":  # Only load shoulder segmentation model if option is selected
            try:
                self._shoulder_segmentor = ShoulderSegmentation(self._cfg)
            except Exception as e:
                logger.error(e)
                logger.error("Unable to load shoulder segmentation model!".format(model_path))
                sys.exit()
        else:
            try:
                self._classifier = pickle.load(open(model_path, 'rb'))
            except Exception as e:
                logger.error(e)
                logger.error("Unable to load shoulder alignment classifier model from {}!".format(model_path))
                sys.exit()

    def check_shoulderalignment(self, im_cv, seg_image):
        """
        Checks shoulder alignment using method 1) shoulder segmentation 2) silhouette
        """
        status = CheckStatus.STATUS_UNKNOWN.value
        remarks = "Unable to perform check."
        height, width, channel = im_cv.shape
        im_debug = im_cv.copy()

        if self._shoulder_segmentor is not None:
            shoulder_segment_res = self._shoulder_segmentor.segment(im_cv)
            shoulder_bb = shoulder_segment_res["res_shoulder_boxes"]
            boxes, scores, classes = shoulder_bb  # Unpack

            best_left_box = None
            best_right_box = None
            best_left_score = 0
            best_right_score = 0
            left_area = 0
            right_area = 0

            # Check how many left/right shoulder detected
            left_count, right_count = classes.count(0), classes.count(1)

            for row, box in enumerate(boxes):
                if classes[row] == 0:  # if left shoulder:
                    if best_left_score < scores[row]:
                        best_left_score = scores[row]
                        best_left_box = box

                if classes[row] == 1:  # if right shoulder:
                    if best_right_score < scores[row]:
                        best_right_score = scores[row]
                        best_right_box = box

            # Prioritise aligned left and right pair
            if best_left_box is not None and best_right_box is not None:
                if iou_batch([best_left_box], [best_right_box]) < 0.1:  # if poor alignment detected
                    if left_count == 1 and right_count > 1:
                        # Find the best aligned right shoulder bounding box
                        best_iou_score = 0
                        for row, box in enumerate(boxes):
                            if classes[row] == 1:
                                # Align right box to left box vertically
                                shift_x = abs(best_left_box[0] - box[0])
                                shifted_right_box = [box[0] - shift_x, box[1],
                                                     box[2] - shift_x, box[3]]

                                iou_score = iou_batch([best_left_box], [shifted_right_box])
                                if iou_score > best_iou_score:
                                    best_iou_score = iou_score
                                    best_right_box = box
                    elif left_count > 1 and right_count == 1:
                        # Find the best aligned left shoulder bounding box
                        best_iou_score = 0
                        for row, box in enumerate(boxes):
                            if classes[row] == 0:
                                # Align left box to left box vertically
                                shift_x = abs(box[0] - best_right_box[0])

                                # Always shift box from right to left
                                shifted_right_box = [best_right_box[0] - shift_x, best_right_box[1],
                                                     best_right_box[2] - shift_x, best_right_box[3]]

                                iou_score = iou_batch([shifted_right_box], [box])
                                if iou_score > best_iou_score:
                                    best_iou_score = iou_score
                                    best_left_box = box
                    elif left_count > 1 and right_count > 1:
                        # TODO: Find the best aligned pair below nose/mouth
                        pass
            else:  # if at least one side of shoulder not detected
                status = CheckStatus.STATUS_FAIL.value
                remarks = "Shoulder is not clearly visible."
                return {"status": status, "remarks": remarks}

            if best_left_box is not None:
                box = best_left_box
                cv2.rectangle(im_debug, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                bl_tr = [box[0], box[3], box[2], box[1]]

                cv2.line(im_debug, (bl_tr[0], bl_tr[1]), (bl_tr[2], bl_tr[3]), (255, 128, 0), 1)
                left_area = shoulder_area([bl_tr[0], bl_tr[1], bl_tr[2], bl_tr[3]], height)
                cv2.putText(im_debug, f'{left_area:.2f}', (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1, 255)

                slope_l = get_slope(bl_tr)

            if best_right_box is not None:
                box = best_right_box
                cv2.rectangle(im_debug, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                tl_br = box
                cv2.line(im_debug, (tl_br[0], tl_br[1]), (tl_br[2], tl_br[3]), (0, 128, 255), 1)
                right_area = shoulder_area([tl_br[0], tl_br[1], tl_br[2], tl_br[3]], height)
                cv2.putText(im_debug, f'{right_area:.2f}', (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1, 255)
                slope_r = get_slope(tl_br)

            if left_area != 0 and right_area != 0:
                area_ratio = min(left_area, right_area) / max(left_area, right_area)
                cv2.putText(im_debug, f'Ratio: {area_ratio:.2f}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, 255)

                # Compute IOU of left and right shoulder
                # Align right box to left box vertically
                shift_x = best_right_box[0] - best_left_box[0]
                shifted_right_box = [best_right_box[0] - shift_x, best_right_box[1],
                                     best_right_box[2] - shift_x, best_right_box[3]]

                iou_score = iou_batch([best_left_box], [shifted_right_box])

                cv2.putText(im_debug, f'IOU: {iou_score[0][0]:.2f}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, 255)
                cv2.putText(im_debug, f'Slope_l: {abs(slope_l):.2f}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, 255)
                cv2.putText(im_debug, f'Slope_r: {abs(slope_r):.2f}', (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, 255)

                #slope_diff = abs(abs(slope_l) - abs(slope_r))

                # Compute left and right box top y coord:
                delta_y_ratio = abs(bl_tr[3] - tl_br[1]) / height

                mid_x = int(width / 2)
                mid_y = int(height / 2)

                # TODO: Impose minimum height of shoulder
                if area_ratio > self._cfg['shoulder_alignment_check']["shoulder_seg"]['area_ratio_thres'] \
                        and iou_score > self._cfg['shoulder_alignment_check']["shoulder_seg"]['iou_thres']\
                        and delta_y_ratio < self._cfg['shoulder_alignment_check']["shoulder_seg"]['delta_y_ratio_thres']:
                    cv2.putText(im_cv, 'Pass', (mid_x, mid_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                    status = CheckStatus.STATUS_PASS.value
                    remarks = "Shoulder is aligned."
                else:
                    cv2.putText(im_cv, 'Fail', (mid_x, mid_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                    status = CheckStatus.STATUS_FAIL.value
                    remarks = "Shoulder is not aligned or not clearly visible."

        else:
            sample = preprocess(seg_image).reshape(1, -1)
            ypred = self._classifier.predict(sample)

            if ypred[0] == 1:
                status = CheckStatus.STATUS_PASS.value
                remarks = "Shoulder is aligned."
            else:
                status = CheckStatus.STATUS_FAIL.value
                remarks = "Shoulder is not aligned."

        return {"status": status, "remarks": remarks}
