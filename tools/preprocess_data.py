import argparse
import os

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="model config file path")
    parser.add_argument("--in_path", help="path to load raw images")
    parser.add_argument("--out_path", help="path to save processed images")
    args = parser.parse_args()
    return args


def rename_images(in_path, out_path):
    for dirpath, dirnames, filenames in os.walk(in_path):
        for filename in filenames:
            apath = os.path.join(dirpath, filename)
            img = cv2.imread(apath, cv2.IMREAD_ANYDEPTH)
            fid = filename.split('_')[1]
            ext = os.path.splitext(filename)[1]
            apath = os.path.join(out_path, fid+ext)
            cv2.imwrite(apath, img)


def main():
    args = parse_args()

    if not os.path.isdir(args.in_path):
        RuntimeError("{} is not a valid in_path".format(args.in_path))
    if not os.path.isdir(args.out_path):
        RuntimeError("{} is not a valid out_path".format(args.out_path))

    print("rename images ...")
    rename_images(args.in_path, args.out_path)
    print("done!")


if __name__ == "__main__":
    main()
