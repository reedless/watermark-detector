# from PIL import Image
# import numpy as np
import cv2
# import os.path as osp
# import os
# import sys
# import datetime
import torch
# from tqdm import tqdm
# import time
# import logging
# import matplotlib.pyplot as plt
# from glob import glob
# from detectron2.utils.logger import setup_logger
# from detectron2.structures import BoxMode
# from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import MetadataCatalog
# from detectron2.engine import DefaultTrainer
# from detectron2.engine.hooks import HookBase
# from detectron2.evaluation import COCOEvaluator
# from detectron2.utils.logger import log_every_n_seconds
# from detectron2.data import DatasetMapper, build_detection_test_loader
# from detectron2.checkpoint import DetectionCheckpointer
# import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

# setup_logger()

# def get_dataset_dicts(input_image_path, watermark_mask_path, word_mask_path):

#     input_image_files = sorted(os.listdir(input_image_path))
#     watermark_mask_files = sorted(os.listdir(watermark_mask_path))
#     word_mask_files = sorted(os.listdir(word_mask_path))

#     dataset_dicts = []

#     for id in tqdm(range(len(input_image_files))):
#         if input_image_files[id][-15:] == 'Zone.Identifier':
#             continue
#         input_image_file = osp.join(input_image_path, input_image_files[id])
#         watermark_mask_file = osp.join(watermark_mask_path, watermark_mask_files[id])
#         word_mask_file = osp.join(word_mask_path, word_mask_files[id])

#         watermark_mask_img = Image.open(watermark_mask_file)
#         word_mask_img = Image.open(word_mask_file)

#         img_width, img_height = watermark_mask_img.size
#         record = {"file_name": input_image_file,
#                   "height": img_height,
#                   "width": img_width,
#                   "image_id": id,
#                   "annotations": []}

#         for idx, mask in enumerate([watermark_mask_img, word_mask_img]):
#             W = np.array(mask).astype(np.uint8)

#             img_gray = cv2.cvtColor(W, cv2.COLOR_BGR2GRAY)
#             _, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
#             kernel = np.ones((3, 3), np.uint8)
#             thresh = cv2.dilate(thresh, kernel, iterations=1)
#             contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             contours = [contour for contour in contours if contour.shape[0] > 3]

#             objs = []
#             for contour in contours:
#                 pairs = [pair[0] for pair in contour]
#                 px = [int(a[0]) for a in pairs]
#                 py = [int(a[1]) for a in pairs]
#                 poly = [int(p) for x in pairs for p in x]

#                 obj = {
#                     "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
#                     "bbox_mode": BoxMode.XYXY_ABS,
#                     "segmentation": [poly],
#                     "category_id": idx,
#                     "iscrowd": 0
#                 }
#                 objs.append(obj)
#             record['annotations'] += objs

#         dataset_dicts.append(record)

#     return dataset_dicts

def check_watermark(cfg, input_im, face_foregroun_background_res, background_check_result, face_highlight_res):
    model_cfg = get_cfg()
    model_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    model_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    model_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    model_cfg.MODEL.WEIGHTS = cfg["watermark"]["weight_path"]

    # check if GPU exists
    has_cuda = torch.cuda.is_available()  
    if cfg["setting"]["force_cpu"]:
        model_cfg.MODEL.DEVICE = "cpu"
    elif has_cuda and cfg["watermark"]["use_gpu"]:
        model_cfg.MODEL.DEVICE = "cuda"
    else:
        model_cfg.MODEL.DEVICE = "cpu"

    processed_img = 0

    return {'status': 0, 'remarks': 'Watermark detected.'}, processed_img

if __name__ == '__main__':

    watermarks_metadata = MetadataCatalog.get("watermarks_train")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.WEIGHTS = 'models/watermark/detectron2.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    im = cv2.imread('dataset/benchmarkv2/clean_Img_001301.jpg')
    output = predictor(im)

    v = Visualizer(im[:, :, ::-1],
                    metadata=watermarks_metadata,
                    scale=1,
                    instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                    )
    v = v.draw_instance_predictions(output["instances"].to("cpu"))

    cv2.imshow("Watermark Check", cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)

    # # score_benchmark
    # for i in tqdm(os.listdir('dataset/benchmarkv2')):
    #     if i[-15:] == 'Zone.Identifier':
    #         continue
    #     im = cv2.imread(os.path.join('./dataset/benchmarkv2', i))
    #     outputs = predictor(im)

    #     # Filter out results with background and specular highlight

    #     v = Visualizer(im[:, :, ::-1],
    #                     metadata=watermarks_metadata,
    #                     scale=1,
    #                     instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
    #                     )
    #     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     fig, ax = plt.subplots(1, 2, figsize=(14, 10))
    #     ax[0].imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    #     ax[1].imshow(im[:, :, ::-1])

    #     if not os.path.exists('./output/benchmarkv2'):
    #         os.mkdir('./output/benchmarkv2')
    #     os.makedirs('./output/benchmarkv2/', exist_ok=True)
    #     plt.savefig(f'./output/benchmarkv2/{i}')
    #     plt.close()

