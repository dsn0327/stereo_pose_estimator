import torch
import torch.nn as nn

import cv2


class ContoursDetector(nn.Module):
    r"""
    Detector for LEDs capture in the image.
    It performs detection on the raw (distorted) image
    and returns LEDs' positions in the rectified (undistorted) image.
    """
    def __init__(
        self,
        detect_cfg):
        super(ContoursDetector, self).__init__()

        self.thresh = detect_cfg.thresh
        self.gauss_sigma = detect_cfg.gauss_sigma
        self.min_blob_area = detect_cfg.min_blob_area
        self.max_blob_area = detect_cfg.max_blob_area
        self.show_results = detect_cfg.show_results

    @torch.no_grad()
    def inference(self, meta):
        img = meta["img"]
        height, width = img.shape
        if meta["roi"] is None:
            meta["roi"] = (
                (range(0, self.width), range(0, self.height)),
                (range(self.width, self.width << 2), range(0, self.height))
            )
        roi_l, roi_r = meta["roi"][0], meta["roi"][1]

        # threshold the image
        _, imgl_thresh = cv2.threshold(img[roi_l[0], roi_l[1]], self.thresh, 255, cv2.THRESH_TOZERO)
        _, imgr_thresh = cv2.threshold(img[roi_r[0], roi_r[1]], self.thresh, 255, cv2.THRESH_TOZERO)

        if self.show_results:
            meta["imgl_detect"] = imgl_thresh
            meta["imgr_detect"] = imgr_thresh
        return meta