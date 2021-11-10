from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, Metadata

import torch
import cv2
import numpy as np
from .obj_seg_interface import ObjectSegmentation
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SpecularHighlightSegmentation(ObjectSegmentation):
    def __init__(self, cfg):
        self._cfg = cfg
        self._device = "cpu"

        try:
            self._load_model()
        except Exception as e:
            logger.error(e)
            logger.error("Unable to load specular highlight segmentation model!")
            sys.exit()

    def _load_model(self):
        seg_cfg = get_cfg()
        seg_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        has_cuda = torch.cuda.is_available()  # check if GPU exists
        if has_cuda:
            self._device = "cuda" if self._cfg["specular_seg"]["use_gpu"] else "cpu"
        else:
            self._device = "cpu"

        if self._cfg["setting"]["force_cpu"]:
            self._device = "cpu"

        if self._device == "cpu":
            seg_cfg.MODEL.DEVICE = "cpu"

        seg_cfg.MODEL.WEIGHTS = self._cfg["specular_seg"]["weight_path"]
        seg_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self._cfg["specular_seg"]["threshold"]
        seg_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        self._segmentor = DefaultPredictor(seg_cfg)

    def segment(self, cv_image):
        outputs = self._segmentor(cv_image)

        mask = outputs['instances'].get('pred_masks')
        mask = mask.to('cpu')
        num, h, w = mask.shape
        bin_mask = np.zeros((h, w))

        for m in mask:
            bin_mask += m.numpy()
        bin_mask = bin_mask.astype(np.uint8)*255
        #cv2.imshow("Highlight_mask", bin_mask)

        v = Visualizer(cv_image[:, :, ::-1],
                       metadata=Metadata(name='highlight_data', thing_classes=['Highlight']),
                       scale=1.0,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #cv2.imshow("Highlight", v.get_image()[:, :, ::-1])
        return {"res_img_cv": v.get_image()[:, :, ::-1], "res_highlight_mask": bin_mask}
