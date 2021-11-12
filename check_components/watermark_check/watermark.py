import cv2
import torch
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import numpy as np

def overlap_percentage(watermark, mask):
    """
    watermark, mask: numpy.ndarray of type bool
    returns: [0, 1] percentage of overlap
    """
    overlap = np.logical_and(watermark, mask)
    return overlap.sum() / watermark.sum()

def check_watermark(cfg, input_im, face_foreground_background_res, background_check_result, face_highlight_res):
    status = 0
    remarks = 'Watermark detected.'

    model_cfg = get_cfg()
    model_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    model_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    model_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    model_cfg.MODEL.WEIGHTS = cfg["watermark_check"]["model_path"]

    # check if GPU exists
    has_cuda = torch.cuda.is_available()  
    if cfg["setting"]["force_cpu"]:
        model_cfg.MODEL.DEVICE = "cpu"
    elif has_cuda and cfg["watermark_check"]["use_gpu"]:
        model_cfg.MODEL.DEVICE = "cuda"
    else:
        model_cfg.MODEL.DEVICE = "cpu"

    predictor = DefaultPredictor(model_cfg)
    output = predictor(input_im)

    selected_idx = []
    for i in range(len(output['instances'].pred_masks)):
        instance = output['instances'].pred_masks[i].cpu().numpy().reshape(-1)

        # don't add instance if background is not white and watermark instance is mostly in the background
        # need to invert the face_foreground_background_res mask because the background is black
        if (background_check_result["status"] == 0 and 
            overlap_percentage(instance, ~face_foreground_background_res[:,:,0].reshape(-1)) > 0.9):
                continue

        # don't add instance if watermark is mostly a specular reflection
        if overlap_percentage(instance, face_highlight_res["res_highlight_mask"].reshape(-1) > 0.9):
            continue

        # add to selected index
        selected_idx.append(i)

    o = output['instances']
    classes = o.pred_classes[selected_idx]
    scores = o.scores[selected_idx]
    boxes = o.pred_boxes[selected_idx]
    masks = o.pred_masks[selected_idx]

    filtered_instances = Instances(image_size=(618, 516))

    filtered_instances.set('pred_classes', classes)
    filtered_instances.set('scores', scores)
    filtered_instances.set('pred_boxes', boxes)
    filtered_instances.set('pred_masks', masks)

    v = Visualizer(input_im[:, :, ::-1],
                    metadata=MetadataCatalog.get("watermarks_train"),
                    scale=1,
                    instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                    )
    v = v.draw_instance_predictions(filtered_instances.to("cpu"))

    if len(filtered_instances) == 0:
        status = 1
        remarks = 'No watermark detected.'

    return {'status': status, 'remarks': remarks}, cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)

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

