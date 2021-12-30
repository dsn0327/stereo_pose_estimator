import copy

from .contours_detector import ContoursDetector


def build_detector(cfg):
    detect_cfg = copy.deepcopy(cfg)
    name = detect_cfg.pop("name")
    if name == "ContoursDetector":
        return ContoursDetector(detect_cfg)
    else:
        raise NotImplementedError
