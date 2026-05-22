import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

SCORE_NAMES = ("ent", "mls", "energy", "msp")


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def move_to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [item.to(device) for item in batch]
    return batch.to(device)


def unpack_model_output(raw_output):
    if not isinstance(raw_output, tuple):
        output = raw_output
        return output, -softmax_entropy(output)

    output = raw_output[0]
    if len(raw_output) == 2 and torch.is_tensor(raw_output[1]):
        score = raw_output[1]
        if score.ndim == 1 and score.shape[0] == output.shape[0]:
            return output, score

    return output, -softmax_entropy(output)


def compute_oscr(score_ind, score_ood, pred_ind, y_ind, fpr_thresh=0.95):
    """
    Computes CCR at a target FPR on the OSCR curve.

    This uses the Open-Set Classification Rate (OSCR) curve definition from
    Dhamija et al., "Reducing Network Agnostophobia", NeurIPS 2018
    (arXiv:1811.04110): for a threshold on an ID-like score, FPR is the
    fraction of OOD samples accepted as known and CCR is the fraction of ID
    samples that are both accepted and correctly classified. 
    This implementation reports the paper-style operating point: CCR at a
    specified FPR.
    """
    if len(score_ind) == 0 or len(score_ood) == 0:
        return 0.0

    score_all = np.concatenate((score_ind, score_ood))

    def get_ccr_at_t(t):
        return ((score_ind > t) & (pred_ind == y_ind)).sum() / len(y_ind)

    def get_fpr_at_t(t):
        return (score_ood >= t).sum() / len(score_ood)

    fpr_list = [0.0]
    ccr_list = [0.0]

    for s in np.sort(np.unique(score_all))[::-1]:
        fpr_list.append(get_fpr_at_t(s))
        ccr_list.append(get_ccr_at_t(s))

    closed_set_acc = (pred_ind == y_ind).sum() / len(y_ind)
    fpr_list.append(1.0)
    ccr_list.append(closed_set_acc)

    fpr_array = np.asarray(fpr_list)
    ccr_array = np.asarray(ccr_list)

    unique_fpr = np.unique(fpr_array)
    max_ccr_at_fpr = np.asarray([
        ccr_array[fpr_array == fpr].max() for fpr in unique_fpr
    ])

    if fpr_thresh <= unique_fpr[0]:
        return float(max_ccr_at_fpr[0])
    if fpr_thresh >= unique_fpr[-1]:
        return float(max_ccr_at_fpr[-1])

    return float(np.interp(fpr_thresh, unique_fpr, max_ccr_at_fpr))


def compute_auoscr(score_ind, score_ood, pred_ind, y_ind):
    """
    Computes AUOSCR, the Area Under the Open-Set Classification Rate curve.

    The Open-Set Classification Rate (OSCR) curve, introduced by Dhamija et al.
    (NeurIPS 2018, arXiv:1811.04110), plots Correct Classification Rate (CCR)
    against False Positive Rate (FPR) as the acceptance threshold is varied from
    high to low.

    This function computes the area under that curve using an ID-like score
    (larger means more likely in-distribution). It follows UniEnt's OSCR area
    computation except that the endpoint is aligned with the original OSCR
    paper: when every sample is accepted, CCR is the closed-set accuracy on ID
    samples.
    """
    if len(score_ind) == 0 or len(score_ood) == 0:
        return 0.0

    score_all = np.concatenate((score_ind, score_ood))

    def get_ccr_at_t(t):
        return ((score_ind > t) & (pred_ind == y_ind)).sum() / len(y_ind)

    def get_fpr_at_t(t):
        return (score_ood >= t).sum() / len(score_ood)

    ccr_list = [0.0]
    fpr_list = [0.0]

    for s in np.sort(np.unique(score_all))[::-1]:
        ccr_list.append(get_ccr_at_t(s))
        fpr_list.append(get_fpr_at_t(s))

    closed_set_acc = (pred_ind == y_ind).sum() / len(y_ind)
    ccr_list.append(closed_set_acc)
    fpr_list.append(1.0)
    # UniEnt uses the endpoint below instead.
    # ccr_list.append(1.0)
    # fpr_list.append(1.0)

    auoscr = 0.0
    for i in range(len(fpr_list) - 1):
        auoscr += (
            (fpr_list[i + 1] - fpr_list[i]) * (ccr_list[i + 1] + ccr_list[i]) / 2.0
        )

    return auoscr


def get_metrics(model, test_loader, device, desc_text="Evaluating", cfg=None):
    """
    Calculates accuracy and multiple OOD metrics.
    """
    model.eval()
    outputs_all, labels_all = [], []
    ood_scores_ent_all, ood_scores_mls_all = [], []
    ood_scores_energy_all, ood_scores_msp_all = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=desc_text, leave=False):
            imgs, labels = move_to_device(imgs, device), labels.to(device)

            raw_output = model(imgs)
            output, score_ent = unpack_model_output(raw_output)

            score_mls, _ = output.max(1)
            score_energy = torch.logsumexp(output, dim=1)
            score_msp, _ = output.softmax(1).max(1)

            outputs_all.append(output.cpu())
            labels_all.append(labels.cpu())
            ood_scores_ent_all.append(score_ent.cpu())
            ood_scores_mls_all.append(score_mls.cpu())
            ood_scores_energy_all.append(score_energy.cpu())
            ood_scores_msp_all.append(score_msp.cpu())

    outputs_all = torch.cat(outputs_all)
    labels_all = torch.cat(labels_all)
    all_scores = {
        "ent": torch.cat(ood_scores_ent_all),
        "mls": torch.cat(ood_scores_mls_all),
        "energy": torch.cat(ood_scores_energy_all),
        "msp": torch.cat(ood_scores_msp_all),
    }

    id_mask_bool = (labels_all >= 0) & (labels_all < cfg.num_classes)

    id_outputs = outputs_all[id_mask_bool]
    id_labels = labels_all[id_mask_bool]

    acc = 0.0
    if len(id_labels) > 0:
        preds = id_outputs.argmax(1)
        acc = (preds == id_labels).float().mean().item()

    id_label_np = id_mask_bool.cpu().numpy().astype(int)
    id_pred_np = outputs_all.argmax(dim=1)[id_mask_bool].cpu().numpy()
    id_true_label_np = labels_all[id_mask_bool].cpu().numpy()

    results = {"acc": acc}

    for name, scores_tensor in all_scores.items():
        scores_np = scores_tensor.cpu().numpy()

        if len(np.unique(id_label_np)) < 2:
            auc = 0.0
            fpr95 = 1.0
            oscr_at_fpr95 = 0.0
            auoscr = 0.0
        else:
            auc = roc_auc_score(id_label_np, scores_np)

            fpr, tpr, _ = roc_curve(id_label_np, scores_np)
            target_tpr = 0.95
            if tpr.max() >= target_tpr:
                f = interp1d(tpr, fpr, fill_value="extrapolate")
                fpr95 = float(f(target_tpr))
            else:
                fpr95 = 1.0

            score_ind = scores_np[id_label_np == 1]
            score_ood = scores_np[id_label_np == 0]

            oscr_at_fpr95 = compute_oscr(
                score_ind, score_ood, id_pred_np, id_true_label_np
            )
            auoscr = compute_auoscr(
                score_ind, score_ood, id_pred_np, id_true_label_np
            )

        results[f"auc_{name}"] = auc
        results[f"fpr95_{name}"] = fpr95
        results[f"oscr_at_fpr95_{name}"] = oscr_at_fpr95
        results[f"auoscr_{name}"] = auoscr

    return results


def add_hscore_fields(metrics):
    acc = metrics["acc"]
    for name in SCORE_NAMES:
        auc = metrics[f"auc_{name}"]
        denom = acc + auc
        metrics[f"h_score_{name}"] = 2 * acc * auc / denom if denom > 1e-8 else 0.0
    return metrics


def init_metric_meter():
    meter = {"acc": []}
    for name in SCORE_NAMES:
        meter[f"auc_{name}"] = []
        meter[f"fpr95_{name}"] = []
        meter[f"oscr_at_fpr95_{name}"] = []
        meter[f"auoscr_{name}"] = []
        meter[f"h_score_{name}"] = []
    return meter


def update_metric_meter(meter, metrics):
    for key in meter:
        meter[key].append(metrics[key])


def _safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else 0.0


def log_step_metrics(logger, corruption_type, severity, metrics):
    logger.info(f"[{corruption_type}{severity}]: ACC={metrics['acc']:.2%}")
    logger.info(
        f"  [Entropy]: AUC={metrics['auc_ent']:.2%}, "
        f"FPR@95={metrics['fpr95_ent']:.2%}, "
        f"OSCR@FPR95={metrics['oscr_at_fpr95_ent']:.2%}, "
        f"AUOSCR={metrics['auoscr_ent']:.2%}, "
        f"H-Score={metrics['h_score_ent']:.2%}"
    )
    logger.info(
        f"  [MLS]:     AUC={metrics['auc_mls']:.2%}, "
        f"FPR@95={metrics['fpr95_mls']:.2%}, "
        f"OSCR@FPR95={metrics['oscr_at_fpr95_mls']:.2%}, "
        f"AUOSCR={metrics['auoscr_mls']:.2%}, "
        f"H-Score={metrics['h_score_mls']:.2%}"
    )
    logger.info(
        f"  [Energy]:  AUC={metrics['auc_energy']:.2%}, "
        f"FPR@95={metrics['fpr95_energy']:.2%}, "
        f"OSCR@FPR95={metrics['oscr_at_fpr95_energy']:.2%}, "
        f"AUOSCR={metrics['auoscr_energy']:.2%}, "
        f"H-Score={metrics['h_score_energy']:.2%}"
    )
    logger.info(
        f"  [MSP]:     AUC={metrics['auc_msp']:.2%}, "
        f"FPR@95={metrics['fpr95_msp']:.2%}, "
        f"OSCR@FPR95={metrics['oscr_at_fpr95_msp']:.2%}, "
        f"AUOSCR={metrics['auoscr_msp']:.2%}, "
        f"H-Score={metrics['h_score_msp']:.2%}"
    )


def log_final_summary(logger, meter):
    logger.info("======== Overall Mean Results ========")
    logger.info(f"Mean Accuracy:            {_safe_mean(meter['acc']):.2%}")

    logger.info("--- Using Entropy Score ---")
    logger.info(f"  Mean AUC_Ent:           {_safe_mean(meter['auc_ent']):.2%}")
    logger.info(f"  Mean FPR@95_Ent:        {_safe_mean(meter['fpr95_ent']):.2%}")
    logger.info(f"  Mean OSCR@FPR95_Ent:    {_safe_mean(meter['oscr_at_fpr95_ent']):.2%}")
    logger.info(f"  Mean AUOSCR_Ent:        {_safe_mean(meter['auoscr_ent']):.2%}")
    logger.info(f"  Mean H-Score_Ent:       {_safe_mean(meter['h_score_ent']):.2%}")

    logger.info("--- Using MLS ---")
    logger.info(f"  Mean AUC_MLS:           {_safe_mean(meter['auc_mls']):.2%}")
    logger.info(f"  Mean FPR@95_MLS:        {_safe_mean(meter['fpr95_mls']):.2%}")
    logger.info(f"  Mean OSCR@FPR95_MLS:    {_safe_mean(meter['oscr_at_fpr95_mls']):.2%}")
    logger.info(f"  Mean AUOSCR_MLS:        {_safe_mean(meter['auoscr_mls']):.2%}")
    logger.info(f"  Mean H-Score_MLS:       {_safe_mean(meter['h_score_mls']):.2%}")

    logger.info("--- Using Energy Score ---")
    logger.info(f"  Mean AUC_Energy:        {_safe_mean(meter['auc_energy']):.2%}")
    logger.info(f"  Mean FPR@95_Energy:     {_safe_mean(meter['fpr95_energy']):.2%}")
    logger.info(f"  Mean OSCR@FPR95_Energy: {_safe_mean(meter['oscr_at_fpr95_energy']):.2%}")
    logger.info(f"  Mean AUOSCR_Energy:     {_safe_mean(meter['auoscr_energy']):.2%}")
    logger.info(f"  Mean H-Score_Energy:    {_safe_mean(meter['h_score_energy']):.2%}")

    logger.info("--- Using MSP ---")
    logger.info(f"  Mean AUC_MSP:           {_safe_mean(meter['auc_msp']):.2%}")
    logger.info(f"  Mean FPR@95_MSP:        {_safe_mean(meter['fpr95_msp']):.2%}")
    logger.info(f"  Mean OSCR@FPR95_MSP:    {_safe_mean(meter['oscr_at_fpr95_msp']):.2%}")
    logger.info(f"  Mean AUOSCR_MSP:        {_safe_mean(meter['auoscr_msp']):.2%}")
    logger.info(f"  Mean H-Score_MSP:       {_safe_mean(meter['h_score_msp']):.2%}")
    logger.info("======================================")
