import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import logging

import torch
from tqdm import tqdm

from conf import cfg, load_cfg_from_args
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from octta_data_closed import (
    load_closed_set_eval_loader,
    normalize_closed_set_dataset_name,
)
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

_CORRUPTION_STREAM_DATASETS = {"imagenet-c", "laion-c"}



def reset_model_if_needed(model, should_reset=True):
    if not should_reset:
        logger.warning("not resetting model")
        return

    try:
        model.reset()
        logger.info("resetting model")
    except AttributeError:
        logger.warning("model has no reset(); skip reset")
    except Exception as exc:
        logger.warning("reset failed: %s", exc)



def _get_model_dataset_name(_eval_dataset_name):
    return "imagenet"



def _evaluate_corruption_stream(model, device, eval_dataset_name):
    meter = init_metric_meter()

    for severity in tqdm(cfg.CORRUPTION.SEVERITY, desc="Overall Severity Progress"):
        for corruption_index, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            logger.info(
                "--> Starting closed-set evaluation for %s with severity %s on %s",
                corruption_type,
                severity,
                eval_dataset_name,
            )

            reset_model_if_needed(model, should_reset=(corruption_index == 0))

            test_loader = load_closed_set_eval_loader(
                cfg,
                severity=severity,
                corruption_type=corruption_type,
            )
            desc_text = f"Closed-set metrics ({eval_dataset_name}: {corruption_type} sev {severity})"

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



def _evaluate_single_domain(model, device, eval_dataset_name):
    meter = init_metric_meter()

    logger.info("--> Starting closed-set evaluation on %s", eval_dataset_name)
    reset_model_if_needed(model, should_reset=True)

    test_loader = load_closed_set_eval_loader(cfg)
    desc_text = f"Closed-set metrics ({eval_dataset_name})"

    metrics = get_metrics(
        model,
        test_loader,
        device=device,
        desc_text=desc_text,
        cfg=cfg,
    )
    metrics = add_hscore_fields(metrics)

    update_metric_meter(meter, metrics)
    log_step_metrics(logger, eval_dataset_name, "", metrics)
    log_final_summary(logger, meter)



def evaluate(description):
    args = load_cfg_from_args(description)
    eval_dataset_name = normalize_closed_set_dataset_name(cfg)

    base_model = load_model(
        cfg.MODEL.ARCH,
        cfg.CKPT_DIR,
        _get_model_dataset_name(eval_dataset_name),
        ThreatModel.corruptions,
    ).cuda()

    logger.info("Closed-set dataset resolved to: %s", eval_dataset_name)
    logger.info("Base classifier is loaded as dataset: imagenet")

    model = build_adaptation_model(cfg, args, base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if eval_dataset_name in _CORRUPTION_STREAM_DATASETS:
        _evaluate_corruption_stream(model, device, eval_dataset_name)
    else:
        _evaluate_single_domain(model, device, eval_dataset_name)


if __name__ == "__main__":
    evaluate(
        ">>>>> OCTTA >>>>>> Closed-set evaluation "
        "(ImageNet-C / LAION-C / ImageNet-A / ImageNet-Sketch / ImageNet-R)."
    )
