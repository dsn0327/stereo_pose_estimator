import argparse
import os
from matplotlib import use

import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="model config file path")
    parser.add_argument("--in_path", help="path to load raw images")
    parser.add_argument("--out_path", help="path to save processed images")
    parser.add_argument(
        "--use_raw_data",
        action="store_true",
        default=False,
        help="whether to load image from raw data file ",
    )
    args = parser.parse_args()
    return args


def process_images(in_path, out_path, use_raw_data=True):
    if use_raw_data:
        print("Load image from raw data file")
    else:
        print("Load image from image file")

    for dirpath, dirnames, filenames in os.walk(in_path):
        for filename in filenames:
            ###################################
            # Tao: only for iqiyi stereo data #
            ###################################
            apath = os.path.join(dirpath, filename)

            if use_raw_data:
                # Load image from raw data
                if filename.endswith(('.txt', '.jpg')):
                    continue
                raw_data = np.fromfile(apath, dtype=np.uint8)
                img = np.reshape(raw_data, (480, 1280))
            else:
                # Load image from image data
                img = cv2.imread(apath, cv2.IMREAD_GRAYSCALE)

            # Rename image
            fid = filename.split('_')[1]
            ext = ".jpg"
            apath = os.path.join(out_path, fid+ext)
            # Transpose & Flip image
            img = cv2.flip(cv2.transpose(img), 0)
            height, _ = img.shape
            half_height = height // 2
            imgr = img[0:half_height, :]
            imgl = img[half_height:height, :]
            # Concat
            img = cv2.hconcat([imgl, imgr])
            # Write image
            img = cv2.cvtColor(
                src=img,
                code=cv2.COLOR_GRAY2RGB
            )
            cv2.imwrite(apath, img)
            ###################################


def main():
    args = parse_args()

    if not os.path.isdir(args.in_path):
        RuntimeError("{} is not a valid in_path".format(args.in_path))
    if not os.path.isdir(args.out_path):
        RuntimeError("{} is not a valid out_path".format(args.out_path))

    print("processing raw data ...")
    process_images(args.in_path, args.out_path, args.use_raw_data)
    print("done!")


if __name__ == "__main__":
    main()
