import cv2
import base64
import imghdr
import numpy as np

from ..common.utils.CheckStatus import CheckStatus


# def cv_img_from_base64(base64_str):
#
#     try:
#         buff = base64.b64decode(base64_str)
#         # Find the extension
#         file_ext = imghdr.what(None, h=buff)
#         #print("File extension: {}".format(file_ext))
#
#         im_np = np.frombuffer(buff, dtype=np.uint8)
#         im_cv = cv2.imdecode(im_np, flags=1)
#     except Exception as e:
#
#     return file_ext, im_cv


def check_file_type(base64_img: str):
    cv_image = None
    status = CheckStatus.STATUS_PASS.value
    remarks = "Valid image file type."

    try:
        buff = base64.b64decode(base64_img)
        # Find the extension
        file_ext = imghdr.what(None, h=buff)
        #print("File extension: {}".format(file_ext))

        extension = file_ext in ("jpg", "jpeg", "png")

        if not extension:
            status = CheckStatus.STATUS_FAIL.value
            remarks = "Invalid image file type, only JPG, JPEG, and PNG files are allowed."
        else:
            im_np = np.frombuffer(buff, dtype=np.uint8)
            cv_image = cv2.imdecode(im_np, flags=1)

    except Exception as e:
        status = CheckStatus.STATUS_FAIL.value
        remarks = "Error reading image file."

    return cv_image, {"status": status, "remarks": remarks}

    # #img_ext, im_cv = cv_img_from_base64(base64_img)
    # #extension = img_ext in ("jpg", "jpeg", "png")
    #
    # if not extension:
    #     status = CheckStatus.STATUS_FAIL.value
    #     remarks = "Invalid image file type, only JPG, JPEG, and PNG files are allowed."
    # else:
    #     # decode only if file format is correct
    #     try:
    #         np_arr = np.fromstring(contents, np.uint8)
    #         cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    #     except Exception as e:
    #         status = CheckStatus.STATUS_FAIL.value
    #         remarks = "Error reading image file."
    #
    # return cv_image, {"status": status, "remarks": remarks}



