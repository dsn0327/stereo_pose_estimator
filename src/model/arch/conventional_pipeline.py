import time

import torch
import torch.nn as nn

from ..detector import build_detector
from ..matcher import build_mather


class ConventionalPipeline(nn.Module):
    def __init__(
        self,
        detect_cfg,
        match_cfg=None):
        super(ConventionalPipeline, self).__init__()

        self.detector = build_detector(detect_cfg)
        if match_cfg is not None:
            self.matcher = build_matcher(match_cfg)

    def inference(self, meta):
        with torch.no_grad():
            # torch.cuda.synchronize()
            time1 = time.time()
            meta = self.detector(meta)
            if hasattr(self, "matcher"):
                meta = self.matcher(meta)
            # torch.cuda.synchronize()
            time2 = time.time()
            print("forward time: {:.3f}s".format((time2 - time1)), end=" | ")
            # torch.cuda.synchronize()
        return meta

