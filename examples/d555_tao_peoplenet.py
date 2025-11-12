# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# See README.md for detailed information.

import argparse
import ctypes
import logging
import os

import cupy as cp
import holoscan
import numpy as np
from cuda import cuda

import hololink as hololink_module


class FormatInferenceInputOp(holoscan.core.Operator):
    """Operator to format input image for inference"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # Transpose
        tensor = cp.asarray(in_message.get("preprocessed")).get()
        # OBS: Numpy conversion and moveaxis is needed to avoid strange
        # strides issue when doing inference
        tensor = np.moveaxis(tensor, 2, 0)[None]
        tensor = cp.asarray(tensor)

        # Output tensor
        op_output.emit({"preprocessed": tensor}, "out")


class PostprocessorOp(holoscan.core.Operator):
    """Operator to post-process inference output:
    * Reparameterize bounding boxes
    * Non-max suppression
    * Make boxes compatible with Holoviz

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec):
        spec.input("in")
        spec.input("in_depth")
        spec.output("out")
        spec.param("iou_threshold", 0.15)
        spec.param("score_threshold", 0.5)
        spec.param("image_width", None)
        spec.param("image_height", None)
        spec.param("box_scale", None)
        spec.param("box_offset", None)
        spec.param("grid_height", None)
        spec.param("grid_width", None)
        spec.param("crop", False)

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")
        depth_msg = op_input.receive("in_depth")
        depth_image = cp.asarray(depth_msg.get("depth_image"))  # shape: (H, W, 3), uint8
        depth_raw = cp.asarray(depth_msg.get("depth_raw"))  # shape: (H, W), float32
        depth_scale = 0.001  # 1 mm to cm
        # depth_image = cp.flipud(depth_image)
        # Convert input to cupy array
        # ONNX models produce a singleton batch dimension that we skip.
        boxes = cp.asarray(in_message.get("boxes"))[0, ...]
        scores = cp.asarray(in_message.get("scores"))[0, ...]

        # PeopleNet has three classes:
        # 0. Person
        # 1. Bag
        # 2. Face
        # Here we only keep the Person and Face classes
        boxes = boxes[[0, 1, 2, 3, 8, 9, 10, 11], ...][None]
        scores = scores[[0, 2], ...][None]

        # Loop over label classes
        out = {"person": None, "faces": None}
        for i, label in enumerate(out):
            # Reparameterize boxes
            out[label], scores_nms = self.reparameterize_boxes(
                boxes[:, 0 + i * 4 : 4 + i * 4, ...],
                scores[:, i, ...][None],
            )

            # Non-max suppression
            out[label], _ = self.nms(out[label], scores_nms)

            # Reshape for HoloViz
            if len(out[label]) == 0:
                out[label] = np.zeros([1, 2, 2]).astype(np.float32)
            else:
                out[label][:, [0, 2]] /= self.image_width
                out[label][:, [1, 3]] /= self.image_height
                out[label] = cp.reshape(out[label][None], (1, -1, 2))
                out[label] = cp.asnumpy(out[label])

        output_data = {
            "person": out["person"],
            "faces": out["faces"],
        }
        
        # 4. Mask outside face/person boxes in depth image
        masked_image = cp.zeros_like(depth_image)
        cropped_mask = cp.zeros(depth_image.shape[:2], dtype=cp.bool_)
        depth_h, depth_w = depth_image.shape[:2]

        def apply_crop(label, boxes_tensor):
            if boxes_tensor is None or boxes_tensor.shape[1] == 0:
                return
            for i in range(0, boxes_tensor.shape[1], 2):
                x0_n, y0_n = boxes_tensor[0, i]
                x1_n, y1_n = boxes_tensor[0, i + 1]

                x0 = int(x0_n * depth_w)
                y0 = int(y0_n * depth_h)
                x1 = int(x1_n * depth_w)
                y1 = int(y1_n * depth_h)

                x0, x1 = sorted([max(0, x0), min(depth_w, x1)])
                y0, y1 = sorted([max(0, y0), min(depth_h, y1)])
                if x1 <= x0 or y1 <= y0:
                    continue

                region_mask = cp.logical_not(cropped_mask[y0:y1, x0:x1])
                region_mask = region_mask[:, :, cp.newaxis]
                masked_image[y0:y1, x0:x1, :] = cp.where(
                    region_mask,
                    depth_image[y0:y1, x0:x1, :],
                    masked_image[y0:y1, x0:x1, :]
                )
                cropped_mask[y0:y1, x0:x1] = True

        # Crop faces first, then persons
        if self.crop:
            apply_crop("faces", out["faces"])
            apply_crop("person", out["person"])
            # Add alpha channel: 0 outside crop, 255 inside
            alpha = cp.where(cropped_mask, 255, 0).astype(cp.uint8)
            alpha = alpha[:, :, cp.newaxis]  # shape (H, W, 1)

            # Convert masked_image to RGBA if it's not already
            if masked_image.shape[2] == 3:
                masked_image = cp.concatenate((masked_image, alpha), axis=2)
            else:
                masked_image[:, :, 3] = alpha[:, :, 0]

        # --- Dynamic text label positioning ---
        depth_ranges_m = [(round(i * 0.1, 1), round((i + 1) * 0.1, 1)) for i in range(50)]
        text_entries = [[0.0, 0.0, 0.0001] for _ in range(len(depth_ranges_m))]

        def place_labels(boxes_tensor):
            for i in range(0, boxes_tensor.shape[1], 2):
                p0 = boxes_tensor[0, i]
                p1 = boxes_tensor[0, i + 1]
                x0, y0 = p0
                x1, y1 = p1
                x0_px, y0_px = int(x0 * depth_w), int(y0 * depth_h)
                x1_px, y1_px = int(x1 * depth_w), int(y1 * depth_h)
               # Get center 10x10 region (clamped)
                center_x_px = (x0_px + x1_px) // 2
                center_y_px = (y0_px + y1_px) // 2
                half_size = 5

                x_start = max(0, center_x_px - half_size)
                x_end   = min(depth_w, center_x_px + half_size)
                y_start = max(0, center_y_px - half_size)
                y_end   = min(depth_h, center_y_px + half_size)

                region_raw_depth = depth_raw[y_start:y_end, x_start:x_end].astype(cp.float32)
                region_mask = region_raw_depth > 0
                if not cp.any(region_mask):
                    continue

                median_depth_m = cp.median(region_raw_depth[region_mask]).item() / 1000.0

                label_index = next(
                    (idx for idx, (low, high) in enumerate(depth_ranges_m)
                    if low <= median_depth_m < high),
                    2
                )
                center_x = (x0 + x1) / 2.0
                top_y = min(y0, y1) - 0.05
                text_entries[label_index] = [center_x, top_y, 0.04]

        place_labels(out["faces"])
        # place_labels(out["person"])

        output_data["text"] = np.ascontiguousarray(
            np.array(text_entries, dtype=np.float32).reshape(1, -1, 3)
        )
        if self.crop:
            output_data["depth_image_cropped"] = masked_image
        else:
            output_data["depth_image_cropped"] = depth_image
        op_output.emit(output_data, "out")

    def nms(self, boxes, scores):
        """Non-max suppression (NMS)

        Parameters
        ----------
        boxes : array (4, n)
        scores : array (n,)

        Returns
        ----------
        boxes : array (m, 4)
        scores : array (m,)

        """
        if len(boxes) == 0:
            return cp.asarray([]), cp.asarray([])

        # Get coordinates
        x0, y0, x1, y1 = boxes[0, :], boxes[1, :], boxes[2, :], boxes[3, :]

        # Area of bounding boxes
        area = (x1 - x0 + 1) * (y1 - y0 + 1)

        # Get indices of sorted scores
        indices = cp.argsort(scores)

        # Output boxes and scores
        boxes_out, scores_out = [], []

        # Iterate over bounding boxes
        while len(indices) > 0:
            # Get index with highest score from remaining indices
            index = indices[-1]

            # Pick bounding box with highest score
            boxes_out.append(boxes[:, index])
            scores_out.append(scores[index])

            # Get coordinates
            x00 = cp.maximum(x0[index], x0[indices[:-1]])
            x11 = cp.minimum(x1[index], x1[indices[:-1]])
            y00 = cp.maximum(y0[index], y0[indices[:-1]])
            y11 = cp.minimum(y1[index], y1[indices[:-1]])

            # Compute IOU
            width = cp.maximum(0, x11 - x00 + 1)
            height = cp.maximum(0, y11 - y00 + 1)
            overlap = width * height
            union = area[index] + area[indices[:-1]] - overlap
            iou = overlap / union

            # Threshold and prune
            left = cp.where(iou < self.iou_threshold)
            indices = indices[left]

        # To array
        boxes = cp.asarray(boxes_out)
        scores = cp.asarray(scores_out)

        return boxes, scores

    def reparameterize_boxes(self, boxes, scores):
        """Reparameterize boxes from corner+width+height to corner+corner.

        Parameters
        ----------
        boxes : array (1, 4, grid_height, grid_width)
        scores : array (1, 1, grid_height, grid_width)

        Returns
        ----------
        boxes : array (4, n)
        scores : array (n,)

        """
        cell_height = self.image_height / self.grid_height
        cell_width = self.image_width / self.grid_width

        # Generate the grid coordinates
        mx, my = cp.meshgrid(cp.arange(self.grid_width), cp.arange(self.grid_height))
        mx = mx.astype(np.float32).reshape((1, 1, self.grid_height, self.grid_width))
        my = my.astype(np.float32).reshape((1, 1, self.grid_height, self.grid_width))

        # Compute the box corners
        xmin = -(boxes[0, 0, ...] + self.box_offset) * self.box_scale + mx * cell_width
        ymin = -(boxes[0, 1, ...] + self.box_offset) * self.box_scale + my * cell_height
        xmax = (boxes[0, 2, ...] + self.box_offset) * self.box_scale + mx * cell_width
        ymax = (boxes[0, 3, ...] + self.box_offset) * self.box_scale + my * cell_height
        boxes = cp.concatenate([xmin, ymin, xmax, ymax], axis=1)

        # Select the scores that are above the threshold
        scores_mask = scores > self.score_threshold
        scores = scores[scores_mask]
        scores_mask = cp.repeat(scores_mask, 4, axis=1)
        boxes = boxes[scores_mask]

        # Reshape after masking
        n = int(boxes.size / 4)
        boxes = boxes.reshape(4, n)

        return boxes, scores