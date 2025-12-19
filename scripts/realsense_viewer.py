#!/usr/bin/env python3
"""
Simple Intel RealSense depth camera viewer.
Requires pyrealsense2 (and optionally OpenCV for visualization).
"""

import argparse
import signal
import sys
import time
from typing import Optional, Tuple

import pyrealsense2 as rs

try:
    import cv2
    import numpy as np

    HAS_CV2 = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_CV2 = False
    cv2 = None  # type: ignore
    np = None  # type: ignore


class GracefulExit(Exception):
    """Raised when the user requests shutdown."""


def _install_signal_handlers():
    def _handler(signum, frame):  # noqa: ARG001 - required signature
        raise GracefulExit

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def build_pipeline(enable_color: bool) -> Tuple[rs.pipeline, rs.config]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if enable_color:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    return pipeline, config


def run_without_cv(pipeline: rs.pipeline):
    print("OpenCV 不可用，将以文本形式输出深度信息。按 Ctrl+C 退出。")
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        dist = depth_frame.get_distance(depth_frame.get_width() // 2, depth_frame.get_height() // 2)
        print(f"[{time.strftime('%H:%M:%S')}] 中心点距离: {dist:.3f} 米", end="\r", flush=True)


def run_with_cv(pipeline: rs.pipeline, show_color: bool):
    print("按 ESC 退出。")
    colorizer = rs.colorizer()
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        depth_color_frame = colorizer.colorize(depth_frame)
        depth_image = np.asanyarray(depth_color_frame.get_data())

        if show_color:
            color_frame = frames.get_color_frame()
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                combined = np.hstack((color_image, depth_image))
            else:
                combined = depth_image
        else:
            combined = depth_image

        cv2.imshow("RealSense", combined)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="打开 Intel RealSense 深度相机并显示画面。")
    parser.add_argument(
        "--color",
        action="store_true",
        help="同时显示 RGB 画面（需要摄像头支持）。",
    )
    args = parser.parse_args()

    _install_signal_handlers()

    pipeline, config = build_pipeline(enable_color=args.color)
    try:
        profile = pipeline.start(config)
        device: Optional[rs.device] = profile.get_device()
        print(f"已连接到设备: {device.get_info(rs.camera_info.name)}")
        print("开始获取帧...")

        if HAS_CV2:
            run_with_cv(pipeline, args.color)
        else:
            run_without_cv(pipeline)
    except GracefulExit:
        print("\n收到退出信号，正在停止管线...")
    finally:
        pipeline.stop()
        print("已关闭 RealSense 管线。")


if __name__ == "__main__":
    try:
        main()
    except GracefulExit:
        print("\n用户中断。")
    except rs.error as err:
        print(f"RealSense SDK 调用失败: {err}")
        sys.exit(1)

