import numpy as np
import cv2
import logging
from ..common.utils.CheckStatus import CheckStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_hair_touch_eyebrow_pixel_touch_count(segmask):
    left_eye_region = 4
    right_eye_region = 5
    left_eyebrow_region = 6  # number of region whose neighbors we want
    right_eyebrow_region = 7

    y = segmask == left_eye_region  # convert to Boolean
    y = np.logical_or(y, segmask == right_eye_region)
    y = np.logical_or(y, segmask == left_eyebrow_region)
    y = np.logical_or(y, segmask == right_eyebrow_region)

    rolled = np.roll(y, 1, axis=0)  # shift down
    rolled[0, :] = False
    z = np.logical_or(y, rolled)

    rolled = np.roll(y, -1, axis=0)  # shift up
    rolled[-1, :] = False
    z = np.logical_or(z, rolled)

    rolled = np.roll(y, 1, axis=1)  # shift right
    rolled[:, 0] = False
    z = np.logical_or(z, rolled)

    rolled = np.roll(y, -1, axis=1)  # shift left
    rolled[:, -1] = False
    z = np.logical_or(z, rolled)

    intersect_region = np.extract(z, segmask)
    n_pixels_hair_touch = np.count_nonzero(intersect_region == 13)

    return n_pixels_hair_touch


def estimate_eyebrow_blocked_coverage(segmask, img, face_bb, part_presence_flags, pixel_counts, is_wear_glasses):
    missing_side_eye_mask = np.array([])
    eyebrow_mask = np.array([])
    # print(PartPresence)
    if not part_presence_flags[2]:  # if left eyebrow is missing
        eyebrow_mask = segmask == 7  # use right eyebrow / left eye as location estimator
        if part_presence_flags[0]:  # if left eye presence
            missing_side_eye_mask = segmask == 4
    elif not part_presence_flags[3]:
        eyebrow_mask = segmask == 6  # use left eyebrow / right eye as location estimator
        if part_presence_flags[1]:  # if right eye presence
            missing_side_eye_mask = segmask == 5
    else:  # both eyebrows detected, select the longest one as ref
        if pixel_counts[2] < pixel_counts[3]:  # if right eyebrow thicker than left eyebrow
            eyebrow_mask = segmask == 7
            if part_presence_flags[0]:  # if left eye presence
                missing_side_eye_mask = segmask == 4
        else:
            eyebrow_mask = segmask == 6  # use left eyebrow / right eye as location estimator
            if part_presence_flags[1]:  # if right eye presence
                missing_side_eye_mask = segmask == 5

    if eyebrow_mask.size > 0 and eyebrow_mask.any():

        # assuming there could be multiple blob of right eyebrow pixels due to noisy segmentation
        contours, hierarchy = cv2.findContours((eyebrow_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            # find the biggest area of the contour
            c = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(c)
            EB_y1 = y
            EB_y2 = y + h
            EB_x1 = x
            EB_x2 = x + w
            eb_W = EB_x2 - EB_x1

        if missing_side_eye_mask.size > 0 and missing_side_eye_mask.any():
            contours, hierarchy = cv2.findContours((missing_side_eye_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_NONE)
            if len(contours) != 0:
                # find the biggest area of the contour
                c = max(contours, key=cv2.contourArea)

                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                E_y1 = y
                E_y2 = y + h
                E_x1 = x
                E_x2 = x + w
                e_W = E_x2 - E_x1
        else:
            if is_wear_glasses:  # glasses possibly causing eyes not detected
                return 0.0
            else:
                return 1.0

        if missing_side_eye_mask.any():  # estimate from missing side eyes
            Cx = (E_x1 + E_x2) * 0.5  # find the midpoint x of missing side eye
            x1 = int(max(0, Cx - eb_W * 0.5))
            x2 = int(min(Cx + eb_W * 0.5, img.shape[1] - 1))
        else:  # Can only estimate from visible eyebrow
            if part_presence_flags[2] == False:  # if left eyebrow missing
                xgap = EB_x1 - face_bb[0]
                x2 = face_bb[2] - xgap
                x1 = x2 - eb_W
            else:
                xgap = face_bb[2] - EB_x2
                x1 = face_bb[0] + xgap
                x2 = x1 + eb_W

        # Assuming eye and eyebrows are leveled
        y1 = EB_y1
        y2 = EB_y2

        estimated_eyebrow_mask = segmask[y1:y2, x1:x2]
        coverage = np.count_nonzero(estimated_eyebrow_mask == 13) / estimated_eyebrow_mask.size
        logging.info("Coverage:{}".format(coverage))

        return coverage
    else:
        return 1.0  # return if no eyebrow presence at all


def get_eye_eyebrows_pixel_count(segmask):
    # Remember the left right is based on the person in the picture, not from viewer perspective!!!
    left_eye_n_pixels = np.count_nonzero(segmask == 4)
    right_eye_n_pixels = np.count_nonzero(segmask == 5)
    left_eyebrow_n_pixels = np.count_nonzero(segmask == 6)
    right_eyebrow_n_pixels = np.count_nonzero(segmask == 7)

    return [left_eye_n_pixels, right_eye_n_pixels, left_eyebrow_n_pixels, right_eyebrow_n_pixels]


def check_glasses_presence(segmask, faceBB, thres=0.1):
    faceArea = (faceBB[2] - faceBB[0]) * (faceBB[3] - faceBB[1])
    pixel_thres = thres * faceArea
    Glasses_Npixels = np.count_nonzero(segmask == 3)
    logging.info("Glass_Npixels: {}, threshold: {}".format(Glasses_Npixels, pixel_thres))
    if Glasses_Npixels > pixel_thres:
        return True
    else:
        return False


def forehead_hair_covered(segmask, bb, thres=0.65):
    forehead_height = int(0.25 * (bb[3] - bb[1]))
    offset_y = int(0.05 * (bb[3] - bb[1]))
    offset_x = int(0.05 * (bb[2] - bb[0]))
    forehead_region = segmask[bb[1] + offset_y:bb[1] + forehead_height, bb[0] + offset_x:bb[2] - offset_x]
    hair_npixels = np.count_nonzero(forehead_region == 13)

    coverage = hair_npixels / (forehead_region.shape[0] * forehead_region.shape[1])
    logging.info("Forehead hair coverage: {}".format(coverage))
    if coverage > thres:
        return True
    else:
        return False


def check_hair_cover_eye(cfg, cv_image, face_boxes_res, face_parsing_res):
    """
    Check if eye or eyebrow is covered by hair
    """
    # Read in threshold parameters from config file
    HAIR_TOUCH_THRES = cfg["hair_cover_eye_check"]["hairtouch_thres"]
    HAIR_COVER_THRES = cfg["hair_cover_eye_check"]["haircover_thres"]
    GLASSES_THRES = cfg["hair_cover_eye_check"]["glasses_thres"]

    im_h = face_parsing_res["res_img_cv"].shape[0]
    im_w = face_parsing_res["res_img_cv"].shape[1]
    img_cv = cv2.resize(cv_image, (im_w, im_h))  # resize to 512x512

    # Get the face bounding box (common module must only pass in single face)
    box = face_boxes_res[0][:-1]
    bb = [int(box[0] * im_w), int(box[1] * im_h), int(box[2] * im_w), int(box[3] * im_h)]

    parts_label = face_parsing_res["res_label_np"]
    # Merge hat (class 14) into hair class (Class 13) to solve hijab issue
    parts_label = np.where(parts_label == 14, 13, parts_label)

    if len(face_boxes_res) != 1 or parts_label.max() > 0:  # if face parsing contains foreground parts
        is_hair_cover_eye = forehead_hair_covered(parts_label, bb, 0.65)

        pixel_counts = get_eye_eyebrows_pixel_count(parts_label)
        part_presence = [x > cfg["hair_cover_eye_check"]["part_thres"] for x in pixel_counts]
        n_pixels_hair_touch = check_hair_touch_eyebrow_pixel_touch_count(parts_label)
        logging.info("Hair touch pixel: {}".format(n_pixels_hair_touch))

        # check whether wear glasses first because eye may not be detected when where glasses
        is_glass_present = check_glasses_presence(parts_label, bb, GLASSES_THRES)

        if part_presence[2] and part_presence[3]:  # if both eyebrows present
            # check if both eyebrows pixel counts similar
            diff_brow = abs(pixel_counts[2] - pixel_counts[3]) / min(pixel_counts[2], pixel_counts[3])
            if diff_brow > 0.2:  # if more than 60% difference in pixel counts, suspect eyebrows is blocked
                if n_pixels_hair_touch > HAIR_TOUCH_THRES:
                    result = -1  # eyebrow/eyes blocked by hair
                else:
                    eyebrow_coverage = estimate_eyebrow_blocked_coverage(parts_label, img_cv, bb,
                                                                         part_presence, pixel_counts, is_glass_present)
                    if eyebrow_coverage > HAIR_COVER_THRES:
                        result = -2
                    else:
                        result = 0  # just fine, some people may be with imbalanced eyebrow
            else:  # if no significance difference between both eye brow pixel counts
                if n_pixels_hair_touch > HAIR_TOUCH_THRES:
                    result = -1  # eyebrow/eyes blocked by hair
                else:
                    result = 0
        elif part_presence[2] is False and part_presence[3] is False:  # both eyebrow not detected
            logging.info("Both eyebrows not detected")
            if is_hair_cover_eye:
                result = -1
            else:
                result = 0  # just fine, some people maybe without eyebrow

        else:  # if either one eyebrow not detected
            if n_pixels_hair_touch > HAIR_TOUCH_THRES:
                result = -1  # eyebrow/eyes blocked by hair
            else:
                eyebrow_coverage = estimate_eyebrow_blocked_coverage(parts_label, img_cv, bb,
                                                                     part_presence, pixel_counts, is_glass_present)
                if eyebrow_coverage > HAIR_COVER_THRES:
                    result = -2
                else:
                    result = 0  # just fine, some people may be with imbalanced eyebrow
    else:
        result = -4  # Not single face or no foreground object

    if result == -1 or result == -2:
        status = CheckStatus.STATUS_FAIL.value
        remarks = "Eye or eyebrow blocked by hair."
    elif result == -4:
        status = CheckStatus.STATUS_ERROR.value
        remarks = "Not single face or face parsing parts empty."
    else:
        status = CheckStatus.STATUS_PASS.value
        remarks = "Eye or eyebrow not covered by hair."

    return {"status": status, "remarks": remarks}
