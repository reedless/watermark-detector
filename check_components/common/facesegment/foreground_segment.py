import cv2
import torch
from torchvision import transforms
from PIL import Image
from .U2netForegroundSeg.u2net import U2NET, U2NETP
from .U2netForegroundSeg.data_utils import RescaleT, ToTensorLab
from .obj_seg_interface import ObjectSegmentation
import numpy as np
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


class ForegroundSegmentation(ObjectSegmentation):
    def __init__(self, cfg):
        self._cfg = cfg
        self._device = "gpu"
        self._selected_model = ""
        try:
            self._load_model()
        except Exception as e:
            logger.error(e)
            logger.error("Unable to load foreground segmentation model!")
            sys.exit()

        self._transform = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])

    def _load_model(self):
        self._selected_model = self._cfg["foreground_seg"]["selector"]
        has_cuda = torch.cuda.is_available()  # check if GPU exists
        if has_cuda:
            self._device = "cuda" if self._cfg["foreground_seg"][self._selected_model]["use_gpu"] else "cpu"

        if self._cfg["setting"]["force_cpu"]:
            self._device = "cpu"

        if self._selected_model == "u2net":
            self._segmentor = U2NET(3, 1)
        elif self._selected_model == "u2netp":
            self._segmentor = U2NETP(3, 1)

        self._segmentor.load_state_dict(torch.load(self._cfg["foreground_seg"][self._selected_model]["weight_path"],
                                                   map_location="cpu"))  # load into cpu by default
        if self._device == "cuda":
            self._segmentor.cuda()

        self._segmentor.eval()

    def segment(self, cv_image):
        """
         Segment people (foreground) from background
         Input: cv_image: OpenCV RGB image
         Output: cv_result_img: OpenCV Grayscale (3-channel) Mask
        """

        rgb_im = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img = self._transform(rgb_im).type(torch.FloatTensor)
        img = torch.stack([img])

        if self._device == "cuda":
            img = img.cuda()

        d1, d2, d3, d4, d5, d6, d7 = self._segmentor(img)
        # Normalization
        pred = d1[:, 0, :, :]
        pred = norm_pred(pred)

        predict = pred.squeeze()
        predict_np = predict.cpu().data.numpy()

        pil_im = Image.fromarray(predict_np * 255).convert("RGB")
        cv_result_img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        cv_result_img = cv2.resize(cv_result_img, (cv_image.shape[1], cv_image.shape[0]))
        # cv2.imshow("ForegroundMask", cv_result_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return cv_result_img

