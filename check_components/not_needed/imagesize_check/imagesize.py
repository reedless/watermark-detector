import logging
from ..common.utils.CheckStatus import CheckStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_image_size(cfg, cv_image):
    _IM_HEIGHT = cfg["image_size_check"]["im_height"]
    _IM_WIDTH = cfg["image_size_check"]["im_width"]

    im_sz_str = "W={}px, H={}px".format(_IM_WIDTH, _IM_HEIGHT)
    status = CheckStatus.STATUS_PASS.value
    remarks = "Image size is {}.".format(im_sz_str)

    height, width, channel = cv_image.shape
    # TODO: to handle situation where image orientation is embedded in metadata
    # Check if image is in portrait or landscape orientation
    orientation = True
    if height < width:
        orientation = False
        logging.error("Image is in landscape not portrait.")

    if width != _IM_WIDTH or height != _IM_HEIGHT:
        status = CheckStatus.STATUS_FAIL.value
        remarks = "Image size is not {}.".format(im_sz_str)
        if width * height < _IM_WIDTH * _IM_HEIGHT * 0.7:
            remarks += " Warning: image size is too small, all check results may be inaccurate."

    return {"status": status, "remarks": remarks}, orientation



