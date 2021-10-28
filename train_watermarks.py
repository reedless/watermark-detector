from PIL import Image
import numpy as np
import cv2
import os.path as osp
import os
import sys
import datetime
import torch
from tqdm import tqdm
import time
import logging
import matplotlib.pyplot as plt
from glob import glob
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context, COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer, Checkpointer
import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
import copy

setup_logger()


def get_dataset_dicts(input_image_path, watermark_mask_path, word_mask_path):

    input_image_files = sorted(os.listdir(input_image_path))
    watermark_mask_files = sorted(os.listdir(watermark_mask_path))
    word_mask_files = sorted(os.listdir(word_mask_path))

    dataset_dicts = []

    for id in tqdm(range(len(input_image_files))):
        if input_image_files[id][-15:] == 'Zone.Identifier':
            continue
        input_image_file = osp.join(input_image_path, input_image_files[id])
        watermark_mask_file = osp.join(watermark_mask_path, watermark_mask_files[id])
        word_mask_file = osp.join(word_mask_path, word_mask_files[id])

        watermark_mask_img = Image.open(watermark_mask_file)
        word_mask_img = Image.open(word_mask_file)

        img_width, img_height = watermark_mask_img.size
        record = {"file_name": input_image_file,
                  "height": img_height,
                  "width": img_width,
                  "image_id": id,
                  "annotations": []}

        for idx, mask in enumerate([watermark_mask_img, word_mask_img]):
            W = np.array(mask).astype(np.uint8)

            img_gray = cv2.cvtColor(W, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [contour for contour in contours if contour.shape[0] > 3]

            objs = []
            for contour in contours:
                pairs = [pair[0] for pair in contour]
                px = [int(a[0]) for a in pairs]
                py = [int(a[1]) for a in pairs]
                poly = [int(p) for x in pairs for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": idx,
                    "iscrowd": 0
                }
                objs.append(obj)
            record['annotations'] += objs

        dataset_dicts.append(record)

    return dataset_dicts


# def custom_mapper(dataset_dict):
#     dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#     image = utils.read_image(dataset_dict["file_name"], format="BGR")
#     transform_list = [
#         T.RandomBrightness(0.8, 1.2),
#         T.RandomContrast(0.8, 1.2),
#         T.RandomSaturation(0.8, 1.2),
#         T.RandomLighting(0.8),
#         T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
#     ]
#
#     # color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
#     # bright_v = random.uniform(color_jitter.brightness[0], color_jitter.brightness[1])
#     # contrast_v = random.uniform(color_jitter.contrast[0], color_jitter.contrast[1])
#     # sat_v = random.uniform(color_jitter.saturation[0], color_jitter.saturation[1])
#     # hue_v = random.uniform(color_jitter.hue[0], color_jitter.hue[1])
#
#     image, transforms = T.apply_transform_gens(transform_list, image)
#     dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
#
#     annos = [
#         utils.transform_instance_annotations(obj, transforms, image.shape[:2])
#         for obj in dataset_dict.pop("annotations")
#         if obj.get("iscrowd", 0) == 0
#     ]
#     instances = utils.annotations_to_instances(annos, image.shape[:2])
#     dataset_dict["instances"] = utils.filter_empty_instances(instances)
#     return dataset_dict


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self.best_validation_loss = 10000000

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            # if torch.cuda.is_available():
                # torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        if mean_loss < self.best_validation_loss:
            checkpointer = DetectionCheckpointer(self._model, save_dir=cfg.OUTPUT_DIR)
            checkpointer.save("best_model")

        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class MyTrainer(DefaultTrainer):
    # @classmethod
    # def build_train_loader(cls, cfg):
    #     return build_detection_train_loader(cfg, mapper=custom_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        print("test_dataset_name = {}".format(dataset_name))
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks


def image_files_from_folder(folder, upper=True):
    extensions = ['png', 'jpg', 'jpeg']
    vid_files = []
    for ext in extensions:
        vid_files += glob('%s/*.%s' % (folder, ext))
        if upper:
            vid_files += glob('%s/*.%s' % (folder, ext.upper()))
    return vid_files


if __name__ == '__main__':
    data_path = 'data'
    for d in ["train", "val", "test"]:
        DatasetCatalog.register("watermarks_" + d,
                                lambda d=d: get_dataset_dicts(f'{data_path}/{d}/input',
                                                              f'{data_path}/{d}/mask_watermark',
                                                              f'{data_path}/{d}/mask_word')
                                )
        MetadataCatalog.get("watermarks_" + d).set(thing_classes=['watermark', 'text'])

    watermarks_metadata = MetadataCatalog.get("watermarks_train")
    watermarks_metadata_val = MetadataCatalog.get("watermarks_val")

    # # visualise
    # dataset_dicts = DatasetCatalog.get("watermarks_val")
    # import random
    # for d in random.sample(dataset_dicts, 20):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=watermarks_metadata_val, scale=1)
    #     vis = visualizer.draw_dataset_dict(d)
    #     print(d["file_name"])
    #     print(vis.get_image()[:, :, ::-1].shape, img.shape)
    #
    #     pic = np.concatenate((vis.get_image()[:, :, ::-1], img), axis = 1)
    #     cv2.imshow(f"{d['file_name']}", pic)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # quit()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("watermarks_train",)
    cfg.DATASETS.TEST = ("watermarks_val",)
    cfg.TEST.EVAL_PERIOD = 100
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 4000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.DEVICE = "cuda"
    # cfg.MODEL.DEVICE = "cpu"

    # # visualising augmented samples
    # trainer = MyTrainer(cfg)
    # train_data_loader = trainer.build_train_loader(cfg)
    # data_iter = iter(train_data_loader)
    # batch = next(data_iter)
    # rows, cols = 3, 3
    # plt.figure(figsize=(20, 20))
    # for i in range(3):
    #     for i, per_image in enumerate(batch[:int(rows * cols)]):
    #         plt.subplot(rows, cols, i + 1)
    #
    #         # Pytorch tensor is in (C, H, W) format
    #         img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
    #         img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)
    #
    #         visualizer = Visualizer(img, metadata=watermarks_metadata, scale=0.5)
    #
    #         target_fields = per_image["instances"].get_fields()
    #         labels = None
    #         vis = visualizer.overlay_instances(
    #             labels=labels,
    #             boxes=target_fields.get("gt_boxes", None),
    #             masks=target_fields.get("gt_masks", None),
    #             keypoints=target_fields.get("gt_keypoints", None),
    #         )
    #         plt.imshow(vis.get_image())
    #     plt.show()
    #     batch = next(data_iter)

    if sys.argv[1] == "train":
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = MyTrainer(cfg)
        trainer.resume_or_load(resume=False)
        # trainer.train()

        # print metrics for test dataset
        cfg.DATASETS.TEST = ("watermarks_test",)
        evaluator = COCOEvaluator("watermarks_test", cfg, False, output_dir="./output/")
        val_loader = build_detection_test_loader(cfg, "watermarks_test")
        
        print(inference_on_dataset(trainer.model, val_loader, evaluator))

    elif sys.argv[1] == "test":
        test_folder = f'{data_path}/test/input'

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        predictor = DefaultPredictor(cfg)

        test_image_list = image_files_from_folder(test_folder)

        # i = 0
        # for d in tqdm(test_image_list):
        #     im = cv2.imread(d)
        #     outputs = predictor(im)
        #     v = Visualizer(im[:, :, ::-1],
        #                    metadata=watermarks_metadata,
        #                    scale=1.0,
        #                    instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
        #                    )
        #     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #     im = np.concatenate([v.get_image()[:, :, ::-1], im], axis=1)

        #     os.makedirs('output/test_result', exist_ok=True)
        #     cv2.imwrite(os.path.join('output/result', f"{i}.jpg"), im)
        #     i += 1

        # for i in tqdm(os.listdir('dataset/ica_rejected')):
        #     if i[-16:] == ':Zone.Identifier':
        #         continue
        #     im = cv2.imread(os.path.join('dataset/ica_rejected', i))
        #     outputs = predictor(im)
        #     v = Visualizer(im[:, :, ::-1],
        #                    metadata=watermarks_metadata,
        #                    scale=1,
        #                    instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
        #                    )
        #     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #     fig, ax = plt.subplots(1, 2, figsize=(14, 10))
        #     ax[0].imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        #     ax[1].imshow(im[:, :, ::-1])

        #     # if not os.path.exists('./output/score2'):
        #     #     os.mkdir('./output/score2')

        #     os.makedirs('output/ica_rejected/', exist_ok=True)
        #     plt.savefig(f'./output/ica_rejected/{i}')
        #     plt.close()

        # # score_passport
        # for i in tqdm(os.listdir('dataset/score_passport')):
        #     if i[-16:] == ':Zone.Identifier':
        #         continue
        #     im = cv2.imread(os.path.join('dataset/score_passport', i))
        #     outputs = predictor(im)
        #     v = Visualizer(im[:, :, ::-1],
        #                    metadata=watermarks_metadata,
        #                    scale=1,
        #                    instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
        #                    )
        #     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #     fig, ax = plt.subplots(1, 2, figsize=(14, 10))
        #     ax[0].imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        #     ax[1].imshow(im[:, :, ::-1])

        #     # if not os.path.exists('./output/score'):
        #     #     os.mkdir('./output/score')
        #     os.makedirs('output/score_passport/', exist_ok=True)
        #     plt.savefig(f'./output/score_passport/{i}')
        #     plt.close()

        # score_benchmark
        for i in tqdm(os.listdir('dataset/benchmark')):
            if i[-15:] == 'Zone.Identifier':
                continue
            im = cv2.imread(os.path.join('dataset/benchmark', i))
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=watermarks_metadata,
                           scale=1,
                           instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                           )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            fig, ax = plt.subplots(1, 2, figsize=(14, 10))
            ax[0].imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
            ax[1].imshow(im[:, :, ::-1])

            # if not os.path.exists('./output/score'):
            #     os.mkdir('./output/score')
            os.makedirs('output/benchmark/', exist_ok=True)
            plt.savefig(f'./output/benchmark/{i}')
            plt.close()

