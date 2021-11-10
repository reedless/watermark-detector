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


class ShoulderSegmentation(ObjectSegmentation):
    def __init__(self, cfg):
        self._cfg = cfg
        self._device = "cpu"

        try:
            self._load_model()
        except Exception as e:
            logger.error(e)
            logger.error("Unable to load shoulder segmentation model!")
            sys.exit()

    def _load_model(self):
        seg_cfg = get_cfg()
        seg_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        has_cuda = torch.cuda.is_available()  # check if GPU exists

        if self._cfg["setting"]["force_cpu"]:
            self._device = "cpu"
        elif has_cuda and self._cfg["shoulder_alignment_check"]["shoulder_seg"]["use_gpu"]:
            self._device = "cuda"
        else:
            self._device = "cpu"
            
        if self._device == "cpu":
            seg_cfg.MODEL.DEVICE = "cpu"

        seg_cfg.MODEL.WEIGHTS = self._cfg["shoulder_alignment_check"]["shoulder_seg"]["model_path"]
        seg_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self._cfg["shoulder_alignment_check"]["shoulder_seg"]["seg_thres"]
        seg_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

        self._segmentor = DefaultPredictor(seg_cfg)

    def segment(self, cv_image):
        im_cv = cv_image.copy()
        outputs = self._segmentor(cv_image)
        predictions = outputs["instances"]
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        boxes = boxes.tensor.cpu().detach().numpy().astype(int)
        scores = predictions.scores if predictions.has("scores") else None
        scores = scores.cpu().detach().numpy()
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None

        shoulder_dets = [boxes, scores, classes]

        v = Visualizer(cv_image[:, :, ::-1],
                       metadata=Metadata(name='shoulder_data', thing_classes=['left', 'right']),
                       scale=1.0,
                       instance_mode=ColorMode.IMAGE_BW)

        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        return {"res_img_cv": v.get_image()[:, :, ::-1], "res_shoulder_boxes": shoulder_dets}
