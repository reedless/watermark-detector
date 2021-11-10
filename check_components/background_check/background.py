import cv2
import numpy as np
from ..common.utils.CheckStatus import CheckStatus
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_background(cfg, cv_image, fg_image):
    """
    fg_image: background is black color and foreground (person) is white color
    """
    # Read in threshold parameters from config file
    BACKGROUND_THRES = cfg["background_check"]["background_thres"]

    status = CheckStatus.STATUS_PASS.value
    remarks = "Background is white."

    dim_im = cv_image.shape
    dim_mask = fg_image.shape

    if not all(x == y for x, y in zip(dim_im, dim_mask)):  # if image and mask dimension is different
        status = CheckStatus.STATUS_FAIL.value
        remarks = "Error: input image and foreground mask dimension different."
        logging.error("[BackgroundCheck] input image and foreground mask dimension different.")
    else:
        mask_gray = cv2.cvtColor(fg_image, cv2.COLOR_BGR2GRAY)
        locs = np.where(mask_gray < 20)  # get background pixels (only confident one)
        pixels = cv_image[locs]
        mean_background_pixel_val = np.mean(pixels)/255

        logging.info("Mean background pixel: {}".format(mean_background_pixel_val))

        if mean_background_pixel_val < BACKGROUND_THRES:
            status = CheckStatus.STATUS_FAIL.value
            remarks = "Background is not white."

    return {"status": status, "remarks": remarks}
