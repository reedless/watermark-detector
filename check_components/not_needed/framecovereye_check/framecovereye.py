import numpy as np
import cv2
import logging
from ..common.utils.CheckStatus import CheckStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_eyes_pixel_count(segmask):
    left_eye_pixels = np.count_nonzero(segmask == 4)
    right_eye_pixels = np.count_nonzero(segmask == 5)
    return [left_eye_pixels, right_eye_pixels]


def check_frame_touch_eyes_region_pixels(cv_image, segmask, dilate_kernel_size=5, dilate_iter=3, has_glasses=False):
    im_h = cv_image.shape[0]
    im_w = cv_image.shape[1]

    y1 = segmask
    z1 = np.where(y1 == 3, 255, y1)
    if has_glasses:
        z1 = np.where(z1 == 4, 255, z1)
        z1 = np.where(z1 == 5, 255, z1)
    glasses_pixels = np.repeat(z1[:, :, np.newaxis], 3, axis=2)

    y1 = segmask
    z1 = np.where(y1 != 255, 255, y1)
    background_pixels = np.repeat(z1[:, :, np.newaxis], 3, axis=2)
    glasses_img = cv2.bitwise_xor(np.uint8(glasses_pixels), np.uint8(background_pixels))

    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    v = np.median(glasses_img)
    sigma = 0.33
    # ---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(glasses_img, lower, upper)
    glasses_dilate_edge = cv2.dilate(edges, kernel, iterations=dilate_iter)
    glasses_dilate_edge = np.repeat(glasses_dilate_edge[:, :, np.newaxis], 3, axis=2)
    glasses_dilate_edge_img = np.uint8(glasses_dilate_edge)
    glasses_outline_edge = glasses_dilate_edge_img

    y1 = segmask
    z1 = np.where(y1 == 4, 255, y1)
    z1 = np.where(z1 == 5, 255, z1)
    eye_pixels = np.repeat(z1[:, :, np.newaxis], 3, axis=2)
    eye_img = np.uint8(eye_pixels)

    v = np.median(glasses_img)
    sigma = 0.33
    # ---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(eye_img, lower, upper)
    eye_dilated_edge = cv2.dilate(edges, kernel, iterations=dilate_iter)
    eye_dilated_edge = np.repeat(eye_dilated_edge[:, :, np.newaxis], 3, axis=2)
    eye_dilated_edge_img = np.uint8(eye_dilated_edge)
    eye_outline_edge = eye_dilated_edge_img

    intersect_img = cv2.bitwise_and(glasses_dilate_edge_img, eye_dilated_edge_img)
    intersect_n_pixels = np.count_nonzero(intersect_img == 255)
    logging.info("Intersect Pixels: {}".format(intersect_n_pixels))

    processed_im = get_processed_image(cv_image, glasses_outline_edge, eye_outline_edge)

    return intersect_n_pixels, processed_im


def check_glasses_presence(segmask, face_bb, thres=0.1):
    face_area = (face_bb[2] - face_bb[0]) * (face_bb[3] - face_bb[1])
    pixel_thres = thres * face_area
    glasses_n_pixels = np.count_nonzero(segmask == 3)
    logging.info("Glass_Npixels: {}, threshold: {}".format(glasses_n_pixels, pixel_thres))
    if glasses_n_pixels > pixel_thres:
        has_glasses = True
    else:
        has_glasses = False
    return has_glasses


def get_processed_image(cv_image, glasses_outline_edge, eye_outline_edge):
    mask_h = glasses_outline_edge.shape[0]
    mask_w = glasses_outline_edge.shape[1]

    im_h = cv_image.shape[0]
    im_w = cv_image.shape[1]

    norm_cv_image = cv2.resize(cv_image, (mask_w, mask_h))  # Resize image to mask dimension
    spectacle_region_image = glasses_outline_edge

    (B, G, R) = cv2.split(spectacle_region_image)
    B = np.where(B == 255, 0, B)
    G = np.where(G == 255, 0, G)
    R = np.where(R == 255, 255, R)
    spectacle_region_image = cv2.merge([B, G, R])

    dst = cv2.addWeighted(spectacle_region_image, 0.5, norm_cv_image, 1, 0)

    eye_region_image = eye_outline_edge

    (B, G, R) = cv2.split(eye_region_image)
    B = np.where(B == 255, 255, B)
    G = np.where(G == 255, 0, G)
    R = np.where(R == 255, 0, R)
    eye_region_image = cv2.merge([B, G, R])

    dst = cv2.addWeighted(eye_region_image, 0.5, dst, 1, 0)
    dst = cv2.resize(dst, (im_w, im_h))  # Resize back to original image size

    return dst


def check_frame_cover_eye(cfg, cv_image, face_boxes_res, face_parsing_res):

    # Read in threshold parameters from config file
    GLASSES_THRES = cfg["frame_cover_eye_check"]["glasses_thres"]
    DILATION_SIZE = cfg["frame_cover_eye_check"]["dilation_kernel_size"]
    DILATION_ITER = cfg["frame_cover_eye_check"]["dilation_iter"]

    status = CheckStatus.STATUS_PASS.value
    remarks = "Frame is not covering eyes."

    im_h = face_parsing_res["res_img_cv"].shape[0]
    im_w = face_parsing_res["res_img_cv"].shape[1]
    img_cv = cv2.resize(cv_image, (im_w, im_h))  # resize to 512x512

    # Get the face bounding box (common module must only pass in single face)
    box = face_boxes_res[0][:-1]
    bb = [int(box[0] * im_w), int(box[1] * im_h), int(box[2] * im_w), int(box[3] * im_h)]
    parts_label = face_parsing_res["res_label_np"]

    if parts_label.max() > 0:
        has_glasses = check_glasses_presence(parts_label, bb, GLASSES_THRES)

        n_pixels_frame_touch,  processed_im = check_frame_touch_eyes_region_pixels(cv_image, parts_label, DILATION_SIZE,
                                                                                   DILATION_ITER, has_glasses)
        logging.info("Frame touch eyes pixel: {}".format(n_pixels_frame_touch))

        if has_glasses:
            if n_pixels_frame_touch > 0:
                status = CheckStatus.STATUS_FAIL.value
                remarks = "Frame is covering eyes."
        else:
            status = CheckStatus.STATUS_PASS.value
            remarks = "Not wearing glasses."
    else:
        status = CheckStatus.STATUS_ERROR.value
        remarks = "Face parsing result is empty."

    return processed_im, {"status": status, "remarks": remarks}
