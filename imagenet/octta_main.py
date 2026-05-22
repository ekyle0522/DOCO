import os


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import logging

import torch
from tqdm import tqdm

from conf import cfg, load_cfg_from_args
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from octta_data import load_eval_with_ood
from octta_metrics import (
    add_hscore_fields,
    get_metrics,
    init_metric_meter,
    log_final_summary,
    log_step_metrics,
    update_metric_meter,
)
from octta_methods import build_adaptation_model

logger = logging.getLogger(__name__)


def reset_model_if_needed(model, corruption_index):
    if corruption_index != 0:
        logger.warning("not resetting model")
        return

    try:
        model.reset()
        logger.info("resetting model")
    except AttributeError:
        logger.warning("model has no reset(); skip reset")
    except Exception as e:
        logger.warning("reset failed: %s", e)


def evaluate(description):
    args = load_cfg_from_args(description)

    base_model = load_model(
        cfg.MODEL.ARCH,
        cfg.CKPT_DIR,
        cfg.CORRUPTION.DATASET,
        ThreatModel.corruptions,
    ).cuda()

    model = build_adaptation_model(cfg, args, base_model)
    meter = init_metric_meter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    id_dataset_name = getattr(cfg.CORRUPTION, "ID_BENCHMARK", "imagenet_c")
    logger.info("Evaluation ID dataset variant: %s", id_dataset_name)

    for severity in tqdm(cfg.CORRUPTION.SEVERITY, desc="Overall Severity Progress"):
        for i_x, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            logger.info(
                "--> Starting evaluation for %s with severity %s",
                corruption_type,
                severity,
            )

            # Set i_x = 0 here to force single-domain evaluation without continual adaptation.
            reset_model_if_needed(model, i_x)

            test_loader = load_eval_with_ood(cfg, severity, corruption_type)
            desc_text = f"Metrics ({corruption_type} sev {severity})"

            metrics = get_metrics(
                model,
                test_loader,
                device=device,
                desc_text=desc_text,
                cfg=cfg,
            )
            metrics = add_hscore_fields(metrics)

            update_metric_meter(meter, metrics)
            log_step_metrics(logger, corruption_type, severity, metrics)

    log_final_summary(logger, meter)


if __name__ == "__main__":
    evaluate(">>>>> OCTTA >>>>>> Corruption evaluation (ImageNet-C / LAION-C).")
