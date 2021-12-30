import copy

# from .contours_detector import ContoursDetector


def build_matcher(cfg):
    match_cfg = copy.deepcopy(cfg)
    name = match_cfg.pop("name")
    if name == "None":
        return None
    else:
        raise NotImplementedError
