import numpy as np
import cv2
import logging
import math
from ..common.utils.CheckStatus import CheckStatus


def image_brightness(img, mask):
    b_mean, g_mean, r_mean, _ = cv2.mean(img, mask)
    return math.sqrt(0.241 * (r_mean ** 2) + 0.691 * (g_mean ** 2) + 0.068 * (b_mean ** 2))


def uneven_brightness(img, mask):
    # Uneven lighting check
    # Divide image into four quadrants
    hh = int(img.shape[0]*0.5)
    hw = int(img.shape[1]*0.5)

    quadrant_brightness = []

    for i in range(4):
        if i == 0:
            im = img[:hh, :hw]
            submask = mask[:hh, :hw]
        elif i == 1:
            im = img[:hh, hw+1:]
            submask = mask[:hh, hw+1:]
        elif i == 2:
            im = img[hh+1:, :hw]
            submask = mask[hh+1:, :hw]
        elif i == 3:
            im = img[hh+1:, hw+1:]
            submask = mask[hh+1:, hw+1:]

        quadrant_brightness.append(image_brightness(im, submask))

    return np.std(quadrant_brightness)


def check_exposure(cfg, cv_image, face_boxes_res, face_parsing_res):
    selected_method = cfg["exposure_check"]["selector"]

    UNEVEN_THRES = cfg["exposure_check"]["uneven_thres"]
    UNDER_EXPOSURE_THRES = cfg["exposure_check"][selected_method]["under_exp_thres"]
    OVER_EXPOSURE_THRES = cfg["exposure_check"][selected_method]["over_exp_thres"]

    if len(face_boxes_res) == 1:
        status = CheckStatus.STATUS_PASS.value
        remarks = "Normal exposure."

        im_h = cv_image.shape[0]
        im_w = cv_image.shape[1]

        # Get the face bounding box (common module must only pass in single face)
        box = face_boxes_res[0][:-1]
        bb = [int(box[0] * im_w), int(box[1] * im_h), int(box[2] * im_w), int(box[3] * im_h)]

        x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]
        face_crop = cv_image[y1:y2, x1:x2]
        #cv2.imwrite("face_crop.jpg", face_crop)

        # Get skin area
        skin_region = 1
        nose_region = 2
        parts_label = face_parsing_res["res_label_np"]
        # Merge nose (class 2) into skin class (Class 1)
        parts_label = np.where(parts_label == nose_region, skin_region, parts_label)
        skin_mask_np = parts_label == skin_region  # convert to Boolean
        skin_mask_cv = cv2.resize(skin_mask_np.astype(np.uint8) * 255, (im_w, im_h))  # resize mask to ori img size
        skin_mask_crop = skin_mask_cv[y1:y2, x1:x2]
        crop_norm_size = 128
        face_crop_norm = cv2.resize(face_crop, (crop_norm_size, crop_norm_size))
        skin_mask_cv = cv2.resize(skin_mask_crop, (crop_norm_size, crop_norm_size))
        _, skin_mask_cv_thres = cv2.threshold(skin_mask_cv, 127, 255, cv2.THRESH_BINARY)

        fail_exposure = False
        # cv2.imshow("face_crop_norm", face_crop_norm)
        # cv2.imshow("mask", skin_mask_cv_thres)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if selected_method == "hist":
            # Histogram method

            image_hsv = cv2.cvtColor(face_crop_norm, cv2.COLOR_BGR2HSV)
            hist_v256 = cv2.calcHist([image_hsv], [2], skin_mask_cv_thres, [256], [0, 256])

            #hist_v256 = cv2.calcHist([face_crop_norm], [1], skin_mask_cv_thres, [256], [0, 256])  # use blue channel

            hist_v256 = hist_v256.flatten()
            pixel_range = [0, 50, 100, 150, 235, 255]
            zones = {"black": 0, "shadow": 1, "midtone": 2, "highlight": 3, "white": 4}
            total_pixel = crop_norm_size * crop_norm_size
            sum_of_zone_pixels = []
            for idx in range(len(pixel_range) - 1):
                sum_of_zone_pixels.append(hist_v256[pixel_range[idx]:pixel_range[idx + 1]].sum())

            dominant_zone_idx = np.argmax(sum_of_zone_pixels)
            shadow_ratio = sum(sum_of_zone_pixels[0:2]) / total_pixel

            if dominant_zone_idx == zones["white"]:
                if sum_of_zone_pixels[4]/sum(sum_of_zone_pixels[:3]) > OVER_EXPOSURE_THRES:
                    print("Sum of zone pixel = {}".format(sum_of_zone_pixels))
                    status = CheckStatus.STATUS_FAIL.value
                    remarks = "Overexposure (too bright)."
                    fail_exposure = True
            elif shadow_ratio > UNDER_EXPOSURE_THRES:
                status = CheckStatus.STATUS_FAIL.value
                remarks = "Underexposure (too dark)."
                fail_exposure = True

        else:  # Default use pixel average
            # Exposure check
            brightness = image_brightness(face_crop_norm, skin_mask_cv_thres)
            print("Brightness: {}".format(brightness))

            if brightness < UNDER_EXPOSURE_THRES:
                status = CheckStatus.STATUS_FAIL.value
                remarks = "Underexposure (too dark)."
                fail_exposure = True
            elif brightness > OVER_EXPOSURE_THRES:
                status = CheckStatus.STATUS_FAIL.value
                remarks = "Overexposure (too bright)."
                fail_exposure = True

        # Compute uneven lighting check
        uneven_score = uneven_brightness(face_crop_norm, skin_mask_cv_thres)
        print("Uneven Lighting Score = {}".format(uneven_score))

        if uneven_score > UNEVEN_THRES:
            status = CheckStatus.STATUS_FAIL.value
            if fail_exposure:
                # Append error msg
                remarks = remarks + " Uneven lighting."
            else:
                remarks = "Uneven lighting."

    else:
        status = CheckStatus.STATUS_ERROR.value
        remarks = "Not a single face."

    return {"status": status, "remarks": remarks}
