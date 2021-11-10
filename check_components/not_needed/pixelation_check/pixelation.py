import numpy as np
import cv2
import logging
import sys
from ..common.utils.CheckStatus import CheckStatus

import PIL
from torchvision import models, transforms
import torch.nn as nn
import torch
from torch.hub import load_state_dict_from_url

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class PixelationClassifier(object):
    def __init__(self, cfg):
        self._cfg = cfg
        self._classifier = None
        self._device = "cpu"
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            self._device = "cuda" if self._cfg["pixelation_check"]["use_gpu"] else "cpu"

        if self._cfg["setting"]["force_cpu"]:
            self._device = "cpu"

        self._load_model()

    def _load_model(self):
        model_path = self._cfg["pixelation_check"]["model_path"]

        try:
            if self._device != "cuda":
                device = torch.device("cpu")
                model = torch.load(model_path, map_location=device)
            else:
                model = torch.load(model_path)
                model.to(self._device)

            model.eval()
            self._classifier = model

        except Exception as e:
            logger.error(e)
            logger.error("Unable to load pixelation classifier model from {}!".format(model_path))
            sys.exit()

    def _preprocess_input(self, ipt_img):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, 
                                                             std=std)])
        return transform(ipt_img)

    def check_pixelation(self, img):
        """
        Checks if there are pixelations in image
        """
        img = cv2.resize(img, (400, 514))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(img)

        ipt_img = self._preprocess_input(ipt_img=img)
        ipt_img = ipt_img.unsqueeze(0)

        if self._device == "cuda":
            ipt_img = ipt_img.to(self._device)

        with torch.no_grad():
            pred = self._classifier(ipt_img)

        pred = int(torch.max(pred.data, 1)[1].cpu().detach().numpy())

        if pred == 1:
            status = CheckStatus.STATUS_PASS.value
            remarks = "Image is not pixelated."
        else:
            status = CheckStatus.STATUS_FAIL.value
            remarks = "Image is pixelated."

        return {"status": status, "remarks": remarks}
