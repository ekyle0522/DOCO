"""
Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
built upon on Tent code.
"""
BATCH_COUNTER = 0

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np
import logging
logger = logging.getLogger(__name__)


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


@torch.no_grad()
def restore_sam_perturbation(optimizer, zero_grad=False):
    """Restore parameters from SAM's first-step perturbation without updating."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            old_p = optimizer.state.get(p, {}).get("old_p")
            if old_p is not None:
                p.data = old_p
    if zero_grad:
        optimizer.zero_grad()


class SAR(nn.Module):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, margin_e0=0.4*math.log(1000), reset_constant_em=0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "SAR requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.margin_e0 = margin_e0  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, ema, reset_flag = forward_and_adapt_sar(x, self.model, self.optimizer, self.margin_e0, self.reset_constant_em, self.ema)
            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_sar(x, model, optimizer, margin, reset_constant, ema):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    global BATCH_COUNTER
    BATCH_COUNTER += 1

    optimizer.zero_grad()
    outputs = model(x)
    entropys = softmax_entropy(outputs)

    # logger.info(f"\n--- [SAR] Batch: {BATCH_COUNTER} ---")
    # logger.info(f"Entropy (Softmax) Stats: "
    #       f"Min={entropys.min().item():.4f}, "
    #       f"Mean={entropys.mean().item():.4f}, "
    #       f"Max={entropys.max().item():.4f}, "
    #       f"Margin={margin:.4f}")

    filter_ids_1 = torch.where(entropys < margin)

    num_selected_1 = len(filter_ids_1[0])
    batch_size = len(x)
    # logger.info(f"Samples Selected (Step 1): {num_selected_1}/{batch_size}")
    if num_selected_1 == 0:
    #     logger.info("!!!!!! WARNING: No samples selected in SAR Step 1. Skipping update. !!!!!!")
        return outputs, ema, False

    entropys = entropys[filter_ids_1]
    loss = entropys.mean(0)

    # logger.info(f"Loss (Step 1): {loss.item():.4f}")
    if torch.isnan(loss):
        # logger.info("!!!!!! FATAL: Loss is NaN in SAR Step 1! !!!!!!")
        return outputs, ema, True

    loss.backward()

    optimizer.first_step(zero_grad=True)
    entropys2 = softmax_entropy(model(x))
    entropys2 = entropys2[filter_ids_1]
    loss_second_value = entropys2.clone().detach().mean(0)
    filter_ids_2 = torch.where(entropys2 < margin)

    num_selected_2 = len(filter_ids_2[0])
    # logger.info(f"Samples Selected (Step 2): {num_selected_2}/{num_selected_1}")
    if num_selected_2 == 0:
        # logger.info("!!!!!! WARNING: No samples selected in SAR Step 2. Skipping update. !!!!!!")
        restore_sam_perturbation(optimizer, zero_grad=True)
        return outputs, ema, False

    loss_second = entropys2[filter_ids_2].mean(0)

    # logger.info(f"Loss (Step 2): {loss_second.item():.4f}")
    if torch.isnan(loss_second):
        # logger.info("!!!!!! FATAL: Loss is NaN in SAR Step 2! !!!!!!")
        return outputs, ema, True

    if not np.isnan(loss_second.item()):
        ema = update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

    # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
    loss_second.backward()
    optimizer.second_step(zero_grad=True)

    for name, param in model.named_parameters():
        if param.requires_grad and torch.isnan(param).any():
            logger.info(f"!!!!!! FATAL: NaN detected in parameter: {name} !!!!!!")
            return outputs, ema, True

    # perform model recovery
    reset_flag = False
    if ema is not None:
        if ema < 0.2:
            logger.info("ema < 0.2, now reset the model")
            reset_flag = True

    return outputs, ema, reset_flag


def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what SAR updates
    model.requires_grad_(False)
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with SAR."""
    is_training = model.training
    assert is_training, "SAR needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "SAR needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "SAR should not update all params: " \
                               "check which require grad"
    has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules()])
    assert has_norm, "SAR needs normalization layer parameters for its optimization"
