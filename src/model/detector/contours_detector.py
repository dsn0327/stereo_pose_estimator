import torch
import torch.nn as nn

import copy

import numpy
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
        self.ratio_circularity = detect_cfg.ratio_circularity
        self.ratio_solidity = detect_cfg.ratio_solidity
        self.show_results = detect_cfg.show_results
    
    @torch.no_grad()
    def inference(self, meta):
        img = meta["img"]
        height, width = img.shape
        if meta["roi"] is None:
            meta["roi"] = [
                [0, height, 0, width//2],
                [0, height, width//2, width]
            ]
        roi_l, roi_r = meta["roi"][0], meta["roi"][1]
        imgl = img[roi_l[0]:roi_l[1], roi_l[2]:roi_l[3]]
        imgr = img[roi_r[0]:roi_r[1], roi_r[2]:roi_r[3]]

        # run detection
        meta["imgl_detect"], meta["imgl_feat_points"] = self.findBlobCenters(imgl)
        meta["imgr_detect"], meta["imgr_feat_points"] = self.findBlobCenters(imgr)

        return meta
    
    def findBlobCenters(self, img):
        # threshold
        _, img_thresh = cv2.threshold(
            src=img,
            thresh=self.thresh,
            maxval=255,
            type=cv2.THRESH_TOZERO
        )
        # morphology
        morph_kernel = cv2.getStructuringElement(
            shape=cv2.MORPH_ELLIPSE,
            ksize=(3,3)
        )
        img_erode = cv2.morphologyEx(
            src=img_thresh,
            op=cv2.MORPH_ERODE,
            kernel=morph_kernel
        )
        # Gaussian blur
        img_blur = cv2.GaussianBlur(
            src=img_erode,
            ksize=(1,1),
            sigmaX=self.gauss_sigma,
            sigmaY=self.gauss_sigma,
            borderType=cv2.BORDER_DEFAULT
        )
        # find all contours
        contours, hierarchy = cv2.findContours(
            image=img_blur,
            mode=cv2.RETR_LIST,
            method=cv2.CHAIN_APPROX_NONE
        )

        img_detect = None
        if self.show_results:
            img_detect = copy.deepcopy(img)
            img_detect = cv2.cvtColor(
                src=img_detect,
                code=cv2.COLOR_GRAY2RGB
            )
        
        return self.identifyFeatPoints(img_detect, contours)

    def identifyFeatPoints(self, img, contours):
        r"""
        identify the feature point as the center of contour with valid area
        """
        feat_points = []
        for contour in contours:
            if self.isValidContour(contour):
                # draw valid contour
                if img is not None:
                    cv2.drawContours(
                        image=img,
                        contours=[contour],
                        contourIdx=0,
                        color=(0,255,0),
                        thickness=2,
                        lineType=cv2.LINE_8,
                        maxLevel=0
                    )
                # TODO: calculate blob center
        
        return img, feat_points

    def isValidContour(self, contour):
        # check the contour area
        area = cv2.contourArea(contour)
        if (area < self.min_blob_area or 
            area > self.max_blob_area):
            return False
        # check contour circularity
        perimeter = cv2.arcLength(
            curve=contour,
            closed=True
        )
        ratio_circularity = (4.0 * numpy.pi * float(area)) / float(perimeter * perimeter)
        if ratio_circularity < self.ratio_circularity:
            return False
        # check contour solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        ratio_solidity = float(area) / hull_area
        if ratio_solidity < self.ratio_solidity:
            return False

        return True
