import numpy as np
import cv2
import logging
from ..common.utils.CheckStatus import CheckStatus


def check_specular_highlight(cfg, cv_image, face_wh, specular_seg_res, face_parsing_res):
    status = CheckStatus.STATUS_PASS.value
    remarks = "No specular highlight on skin."

    height, width, channel = cv_image.shape
    face_abs_area = face_wh[0] * face_wh[1]
    skin_region = 1
    nose_region = 2
    parts_label = face_parsing_res["res_label_np"]
    # Merge nose (class 2) into skin class (Class 1)
    parts_label = np.where(parts_label == nose_region, skin_region, parts_label)
    skin_mask_np = parts_label == skin_region  # convert to Boolean
    skin_mask_cv = cv2.resize(skin_mask_np.astype(np.uint8) * 255, (width, height))

    ts = cfg["specular_highlight_check"]["ts"]
    tv = cfg["specular_highlight_check"]["tv"]
    gray_thres = cfg["specular_highlight_check"]["mask_gray_thres"] * 255

    combined_mask = cv2.bitwise_and(skin_mask_cv, specular_seg_res)

    # find all connected components (white blobs in image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time
    # we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = cfg["specular_highlight_check"]["blob_min_ratio"] * float(face_abs_area)
    print("Min blob size threshold = {}".format(min_size))

    filtered_combined_mask = np.zeros(output.shape)
    # for every component in the image, keep it only if it's above min_size
    for i in range(0, nb_components):
        #print(sizes[i])
        if sizes[i] >= min_size:
            filtered_combined_mask[output == i + 1] = 255

    # find the contour of highlight region
    contours, hierarchy = cv2.findContours(filtered_combined_mask.astype(np.uint8), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    disp_im = cv_image.copy()
    cv2.drawContours(disp_im, contours, -1, (255, 0, 0), 2)
    #print("Number of blobs = {}".format(len(contours)))
    #cv2.imshow("Final Highlight Regions", disp_im)

    if len(contours) > 0:
        status = CheckStatus.STATUS_FAIL.value
        remarks = "{} regions of specular highlight detected.".format(len(contours))

    return disp_im, {"status": status, "remarks": remarks}
