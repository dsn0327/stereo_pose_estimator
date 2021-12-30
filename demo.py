import argparse
import os

import torch

import cv2

from src.model.arch import build_model
from src.util import cfg, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="model config file path")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image",
    )
    args = parser.parse_args()
    return args


def get_image_files(path):
    image_files = {"dir": None, "fid": [], "ext": None}
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            fid = int(os.path.splitext(filename)[0])
            ext = os.path.splitext(filename)[1]
            if image_files["dir"] is None:
                image_files["dir"] = dirpath
            if fid not in image_files["fid"]:
                image_files["fid"].append(fid)
            if image_files["ext"] is None:
                image_files["ext"] = ext
    image_files["fid"].sort()
    return image_files


def main():
    args = parse_args()

    load_config(cfg, args.config)
    model = build_model(cfg.model)

    data_path = os.path.join("./data", cfg.data.val.img_path)
    if os.path.isdir(data_path):
        image_files = get_image_files(data_path)
    else:
        RuntimeError("{} is not a valid dir path".format(data_path))

    i = 0
    N = len(image_files["fid"])
    while i in range(0, N):
        fid = image_files["fid"][i]
        filename = os.path.join(image_files["dir"], str(fid)+image_files["ext"])
        # Note: load image as 8-bit grayscale here
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
        meta = {"img": img, "roi": None}

        with torch.no_grad():
            model.inference(meta)

        cv2.imshow("imgl_detect", meta["imgl_detect"])
        cv2.imshow("imgr_detect", meta["imgr_detect"])
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
        elif ch == ord("a") or ch == ord("A"):
            i = max(0, i-1)
        elif ch == ord("d") or ch == ord("D"):
            i = min(N-1, i+1)


if __name__ == "__main__":
    main()
