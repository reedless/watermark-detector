import cv2
import base64
import torch
import torch.nn as nn
import torchvision.utils
from torchvision import transforms
from torchvision.utils import save_image
import PIL
from PIL import Image
from .obj_seg_interface import ObjectSegmentation
from .UnetFaceParsing import unet
from .UnetFaceParsing.utils import *
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def transformer(resize, totensor, normalize, centercrop, imsize):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(160))
    if resize:
        options.append(
            transforms.Resize((imsize, imsize), interpolation=PIL.Image.NEAREST)
        )
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(options)

    return transform


class FaceParsing(ObjectSegmentation):
    """
    Accurate pixel segmentation of facial parts (such as eyes, nose, mouth, etc.)
    """

    def __init__(self, cfg):
        self._cfg = cfg
        self._device = "cpu"
        self._selected_model = ""
        self._imsize = 0  # Different face parsing model may require different input image size

        try:
            self._load_model()
        except Exception as e:
            logger.error(e)
            logger.error("Unable to load face parsing model!")
            sys.exit()

        self.labels_predict_plain = None
        self.labels_predict_color = None

    def _load_model(self):
        self._selected_model = self._cfg["face_parsing"]["selector"]
        has_cuda = torch.cuda.is_available()  # check if GPU exists
        if has_cuda:
            self._device = "cuda" if self._cfg["face_parsing"][self._selected_model]["use_gpu"] else "cpu"

        if self._cfg["setting"]["force_cpu"]:
            self._device = "cpu"

        if self._selected_model == "parsenet":
            self._segmentor = unet()
            self._imsize = 512

        self._segmentor.load_state_dict(torch.load(self._cfg["face_parsing"][self._selected_model]["weight_path"]
                                                   , map_location="cpu"))  # load into cpu by default
        if self._device == "cuda":
            self._segmentor.cuda()

        self._segmentor.eval()

    def segment(self, cv_image):
        transform = transformer(True, True, True, False, self._imsize)
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._imsize, self._imsize))  # resize to 512x512
        img = transform(Image.fromarray(img))
        img = torch.stack([img])

        if self._device == "cuda":
            img = img.cuda()

        labels_predict = self._segmentor(img)

        self.labels_predict_plain = generate_label_plain(labels_predict, self._imsize)
        self.labels_predict_color = generate_label(labels_predict, self._imsize)

        # Convert tensor result to OpenCV image
        np_img = self.labels_predict_color.squeeze().numpy().transpose(1, 2, 0)
        img_n = cv2.normalize(src=np_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv_labels_predict_color = cv2.cvtColor(img_n, cv2.COLOR_RGB2BGR)

        return {"res_img_cv": cv_labels_predict_color, "res_label_np": self.labels_predict_plain[0]}
