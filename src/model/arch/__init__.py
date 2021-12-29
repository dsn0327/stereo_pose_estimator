import copy
import warnings

from .conventional_pipeline import ConventionalPipeline


def build_model(model_cfg):
    model_cfg = copy.deepcopy(model_cfg)
    name = model_cfg.arch.pop("name")
    if name == "ConventionalPipeline":
        warnings.warn(
            "ConventionalPipeline has no learnable module."
        )
        model = ConventionalPipeline(
            detect_cfg=model_cfg.arch.detector,
            match_cfg=model_cfg.arch.matcher if model_cfg.arch.matcher.name is not None else None)
    else:
        raise NotImplementedError
    return model
