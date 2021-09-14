# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset


class Carla(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=8,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(Carla, self).__init__(ignore_label, base_size,
                                    crop_size, downsample_rate, scale_factor, mean, std, )

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip

        self.img_list = [line.strip().split() for line in open(root + list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.label_mapping = {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
            5: 5, 6: 6, 7: 7, 8: 8
        }

        # Nothing, Bridges, Pedestrian, Road, Road line, Side walk, Traffic sign, Traffic light, Vehicles

        # self.class_weights = None

        self.class_weights = torch.FloatTensor([
            0.0373, 0.9345, 1.4843, 0.8373,
            1.2577, 0.918, 1.2489, 1.245, 1.037
        ]).cuda()

        self.color_palette = [
            (0, 0, 0),
            (150, 100, 100),
            (220, 20, 60),
            (128, 64, 128),
            (157, 234, 50),
            (244, 35, 232),
            (220, 220, 0),
            (250, 170, 30),
            (0, 0, 142)
        ]

    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, 'carla', item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label = cv2.imread(os.path.join(self.root, 'carla', item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        image, label = self.gen_sample(image, label,
                                       self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                  ori_height, ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                     new_h, new_w]).cuda()
                count = torch.zeros([1, 1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]

            preds = F.interpolate(
                preds, (ori_height, ori_width),
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
            final_pred += preds
        return final_pred

    def save_pred(self, image, preds, sv_path, name):
        h, w = preds.shape
        image_canvas = np.empty((w, h*2, 3), np.uint8)
        image_canvas[:h] = image[:, :, ::-1]
        image_canvas[h:] = self.fill_color_palette(self.color_palette, preds)
        cv2.imwrite(
            os.path.join(sv_path, name[0] + '.png'),
            image_canvas
        )

    def fill_color_palette(self, labels, segmentation):
        canvas = np.zeros((*segmentation.shape, 3), np.uint8)
        for idx, label in enumerate(labels):
            indices = np.where(segmentation == idx)
            canvas[indices[0], indices[1], :] = label
        return canvas
