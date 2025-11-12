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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import holoscan
from cuda import cuda

import hololink as hololink_module

from d555_tao_peoplenet import FormatInferenceInputOp, PostprocessorOp


class HoloscanApplication(holoscan.core.Application):
    def __init__(
         self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel_stream_depth,
        camera_stream_depth,
        hololink_channel_stream_rgb,
        camera_stream_rgb,
        frame_limit,
        engine,
        crop=False
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel_stream_depth = hololink_channel_stream_depth
        self._camera_stream_depth = camera_stream_depth
        self._hololink_channel_stream_rgb = hololink_channel_stream_rgb
        self._camera_stream_rgb = camera_stream_rgb
        self._frame_limit = frame_limit
        self._engine = engine
        self._crop = crop
        self.is_metadata_enabled = True
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        logging.info("compose")
        if self._frame_limit:
            self._count_stream1 = holoscan.conditions.CountCondition(
                self,
                name="count_stream1",
                count=self._frame_limit,
            )
            condition_stream1 = self._count_stream1
            self._count_stream2 = holoscan.conditions.CountCondition(
                self,
                name="count_stream2",
                count=self._frame_limit,
            )
            condition_stream2 = self._count_stream2
        else:
            self._ok_stream1 = holoscan.conditions.BooleanCondition(
                self, name="ok_stream1", enable_tick=True
            )
            condition_stream1 = self._ok_stream1
            self._ok_stream2 = holoscan.conditions.BooleanCondition(
                self, name="ok_stream2", enable_tick=True
            )
            condition_stream2 = self._ok_stream2

        image_decoder_allocator_pool = holoscan.resources.UnboundedAllocator(self)
        
        image_decoder_stream1 = hololink_module.operators.ImageDecoderOp(
            self,
            name="image_decoder_stream1",
            out_tensor_name="depth_image",
            align_depth_to_rgb=True,
            allocator=image_decoder_allocator_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera_stream_depth.configure_converter(image_decoder_stream1)

        image_decoder_stream2 = hololink_module.operators.ImageDecoderOp(
            self,
            name="image_decoder_stream2",
            allocator=image_decoder_allocator_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera_stream_rgb.configure_converter(image_decoder_stream2)
        
        frame_size = image_decoder_stream1.get_csi_length()
        frame_context = self._cuda_context

        receiver_operator_stream1 = hololink_module.operators.LinuxReceiverOperator(
            self,
            condition_stream1,
            name="receiver_stream1",
            frame_size=frame_size,
            frame_context=frame_context,
            udp_port=54739 + hololink_module.sensors.d555.d555_mode.RealSense_StreamId.DEPTH.value,
            hololink_channel=self._hololink_channel_stream_depth,
            device=self._camera_stream_depth,
        )

        receiver_operator_stream2 = hololink_module.operators.LinuxReceiverOperator(
            self,
            condition_stream2,
            name="receiver_stream2",
            frame_size=frame_size,
            frame_context=frame_context,
            udp_port=54739 + hololink_module.sensors.d555.d555_mode.RealSense_StreamId.RGB.value,
            hololink_channel=self._hololink_channel_stream_rgb,
            device=self._camera_stream_rgb,
        )

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=self._fullscreen,
            headless=self._headless,
            framebuffer_srgb=False,
            width=1280,
            height=720,
            **self.kwargs("holoviz"),
        )

        image_shift = hololink_module.operators.ImageShiftToUint8Operator(
            self, name="image_shift", shift=0
        )

        pool = holoscan.resources.UnboundedAllocator(self)
        preprocessor_args = self.kwargs("preprocessor")
        preprocessor = holoscan.operators.FormatConverterOp(
            self,
            name="preprocessor",
            pool=pool,
            **preprocessor_args,
        )
        format_input = FormatInferenceInputOp(
            self,
            name="transpose",
            pool=pool,
        )
        inference = holoscan.operators.InferenceOp(
            self,
            name="inference",
            allocator=pool,
            model_path_map={
                "face_detect": self._engine,
            },
            **self.kwargs("inference"),
        )
        postprocessor_args = self.kwargs("postprocessor")
        postprocessor_args["image_width"] = preprocessor_args["resize_width"]
        postprocessor_args["image_height"] = preprocessor_args["resize_height"]
        postprocessor_args["crop"] = self._crop
        postprocessor = PostprocessorOp(
            self,
            name="postprocessor",
            allocator=pool,
            **postprocessor_args,
        )

        self.add_flow(receiver_operator_stream1, image_decoder_stream1, {("output", "input")})
        self.add_flow(receiver_operator_stream2, image_decoder_stream2, {("output", "input")})

        self.add_flow(image_decoder_stream2, image_shift, {("output", "input")})
        self.add_flow(image_shift, preprocessor, {("output", "")})
        self.add_flow(image_shift, visualizer, {("output", "receivers")})
        self.add_flow(preprocessor, format_input)
        self.add_flow(format_input, inference, {("", "receivers")})
        self.add_flow(image_decoder_stream1, postprocessor, {("output", "in_depth")})
        self.add_flow(inference, postprocessor, {("transmitter", "in")})
        self.add_flow(postprocessor, visualizer, {("out", "receivers")})

def main():
    parser = argparse.ArgumentParser()
    modes_depth = hololink_module.sensors.d555.d555_mode.RealSense_Depth_Mode
    mode_depth_choices = [mode.value for mode in modes_depth]
    mode_help = " ".join([f"{mode.value}:{mode.name}" for mode in modes_depth])
    parser.add_argument(
        "--camera-mode-depth",
        type=int,
        choices=mode_depth_choices,
        default=mode_depth_choices[4], # 1280x720 30fps depth
        help=mode_help,
    )
    modes_rgb = hololink_module.sensors.d555.d555_mode.RealSense_RGB_Mode
    mode_rgb_choices = [mode.value for mode in modes_rgb]
    mode_help = " ".join([f"{mode.value}:{mode.name}" for mode in modes_rgb])
    parser.add_argument(
        "--camera-mode-rgb",
        type=int,
        choices=mode_rgb_choices,
        default=mode_rgb_choices[6], # 1280x720 30fps rgb
        help=mode_help,
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--crop", action="store_true", help="crop depth in bounding box", default=True)
    parser.add_argument(
        "--fullscreen", action="store_true", help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    default_configuration = os.path.join(
            os.path.dirname(__file__), "d555_tao_peoplenet.yaml"
    )
    parser.add_argument(
        "--configuration", default=default_configuration, help="Configuration file"
    )
    parser.add_argument(
        "--hololink",
        default="192.168.11.55",
        help="IP address of Hololink board",
    )
    default_engine = os.path.join(
        os.path.dirname(__file__), "resnet34_peoplenet_int8.onnx"
    )
    parser.add_argument(
        "--engine",
        default=default_engine,
        help="TRT engine model",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "--expander-configuration",
        type=int,
        default=0,
        choices=(0, 1),
        help="I2C Expander configuration",
    )
    parser.add_argument(
        "--pattern",
        type=int,
        choices=range(12),
        help="Configure to display a test pattern.",
    )
    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)
    logging.getLogger().setLevel(args.log_level)
    logging.info("Initializing.")
    # Get a handle to the GPU
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Get a handle to data sources.  First, find an enumeration packet
    # from the IP address we want to use.
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    overrides = hololink_module.Metadata({
        "vsync_enable": 0,   # or 1
        "block_enable": 0,   # or 1
    })
    channel_metadata.update(overrides)
    logging.info(f"{channel_metadata=}")

    # # Get a handle to the Hololink device
    channel_metadata_stream1 = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_stream1, 0)
    channel_metadata_stream2 = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_stream2, 1)
     #
    hololink_channel_stream_depth = hololink_module.DataChannel(channel_metadata_stream1)
    hololink_channel_stream_rgb = hololink_module.DataChannel(channel_metadata_stream2)
    # Get a handle to the camera
    camera_stream_depth = hololink_module.sensors.d555.d555.RealsenseCamD555(hololink_channel_stream_depth, hololink_module.sensors.d555.d555_mode.RealSense_StreamId.DEPTH)
    camera_stream_rgb = hololink_module.sensors.d555.d555.RealsenseCamD555(hololink_channel_stream_rgb, hololink_module.sensors.d555.d555_mode.RealSense_StreamId.RGB)


    logging.info("camera mode stream Depth: %s", args.camera_mode_depth)
    logging.info("camera mode stream RGB: %s", args.camera_mode_rgb)

    camera_mode_depth = hololink_module.sensors.d555.d555_mode.RealSense_Depth_Mode(
        args.camera_mode_depth
    )

    camera_mode_rgb = hololink_module.sensors.d555.d555_mode.RealSense_RGB_Mode(
        args.camera_mode_rgb
    )

    # Set up the application
    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        hololink_channel_stream_depth,
        camera_stream_depth,
        hololink_channel_stream_rgb,
        camera_stream_rgb,
        args.frame_limit,
        args.engine,
        args.crop,
    )
    application.config(args.configuration)

    # # Run it.
    hololink = hololink_channel_stream_depth.hololink()
    assert hololink is hololink_channel_stream_rgb.hololink()
    hololink.start() 

    camera_stream_depth.configure(camera_mode_depth)
    camera_stream_rgb.configure(camera_mode_rgb)
    os.environ["GXF_MEMORY_DEBUG"] = "1"
    application.run()
    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
