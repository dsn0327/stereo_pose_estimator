import argparse
import os

import cv2
# from src.detector import Detector
# from src.util import cfg, load_config
# from util import cfg, load_config
# from util.path import mkdir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="model config file path")
    parser.add_argument("--path", default="./data", help="path to images")
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
    detector = Detector(cfg)

    if os.path.isdir(args.path):
        image_files = get_image_files(args.path)
    else:
        RuntimeError("{} is not a valid dir path".format(args.path))

    for fid in image_files["fid"]:
        filename = os.path.join(image_files["dir"], str(fid)+image_files["ext"])
        # Note: load image as 8-bit grayscale here
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)

        detector.inference(img)

        cv2.imshow("img raw", img)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


if __name__ == "__main__":
    main()
