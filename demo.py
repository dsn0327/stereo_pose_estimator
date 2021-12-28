import argparse
import os

import cv2
# from util import cfg, load_config
# from util.path import mkdir

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


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


def get_image_list(path):
    image_names = {"dir": None, "fid": [], "ext": None}
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            dir, file = os.path.split(apath)
            fid = int(file.split('_')[0])
            ext = os.path.splitext(file)[1]
            if image_names["dir"] is None:
                image_names["dir"] = dir
            if fid not in image_names["fid"]:
                image_names["fid"].append(fid)
            if image_names["ext"] is None:
                image_names["ext"] = ext
    image_names["fid"].sort()
    return image_names


def main():
    args = parse_args()

    if os.path.isdir(args.path):
        files = get_image_list(args.path)
    else:
        RuntimeError("{} is not a valid dir path".format(args.path))

    for fid in files["fid"]:
        imgl_file = files["dir"] + "/" + str(fid) + "_l" + files["ext"]
        imgr_file = files["dir"] + "/" + str(fid) + "_r" + files["ext"]
        imgl = cv2.imread(imgl_file, 0)
        imgr = cv2.imread(imgr_file, 0)
        cv2.imshow("imgl", imgl)
        cv2.imshow("imgr", imgr)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


if __name__ == "__main__":
    main()
