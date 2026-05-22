import logging
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from method import cotta
from method import doco
from method import dpcore
from method import eata_come
from method import eata_unient
from method import ostta
from method import sar
from method import sar_come
from method import tent
from method import vida
from method.sam import SAM
from method.selectedRotateImageFolder import prepare_test_data
from method.stamp import STAMP
from torch.utils.data import DataLoader, SubsetRandomSampler

logger = logging.getLogger(__name__)


def build_adaptation_model(cfg, args, base_model):
    method = cfg.MODEL.ADAPTATION.lower()

    if method == "source":
        logger.info("test-time adaptation: NONE")
        return setup_source(base_model, cfg)

    elif method == "tent":
        logger.info("test-time adaptation: TENT")
        return setup_tent(base_model, cfg)

    elif method == "cotta":
        logger.info("test-time adaptation: CoTTA")
        return setup_cotta(base_model, cfg)

    elif method == "vida":
        logger.info("test-time adaptation: ViDA")
        return setup_vida(args, base_model, cfg)

    elif method == "doco":
        logger.info("test-time adaptation: DOCO")
        return setup_doco(args, base_model, cfg)

    elif method == "sarcome":
        logger.info("test-time adaptation: SAR_COME")
        return setup_sar_come(args, base_model, cfg)

    elif method == "eatacome":
        logger.info("test-time adaptation: EATA_COME")
        return setup_eata_come(args, base_model, cfg)

    elif method == "ostta":
        logger.info("test-time adaptation: OSTTA")
        return setup_ostta(args, base_model, cfg)

    elif method == "eata":
        logger.info("test-time adaptation: EATA")
        return setup_eata(args, base_model, cfg)

    elif method == "eataunient":
        logger.info("test-time adaptation: EATA_UniEnt")
        return setup_eata_unient(args, base_model, cfg)

    elif method == "stamp":
        logger.info("test-time adaptation: STAMP")
        return setup_stamp(args, base_model, cfg)

    elif method == "sar":
        logger.info("test-time adaptation: SAR")
        return setup_sar(args, base_model, cfg)

    elif method == "dpcore":
        logger.info("test-time adaptation: DPCore")
        return setup_dpcore(args, base_model, cfg)

    else:
        raise NotImplementedError(f"Unknown adaptation method: {cfg.MODEL.ADAPTATION}")


def setup_source(model, cfg):
    model.eval()
    return model


def setup_tent(model, cfg):
    model = tent.configure_model(model)
    params, _ = tent.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    return tent.Tent(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
    )


def setup_cotta(model, cfg):
    model = cotta.configure_model(model)
    params, _ = cotta.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    return cotta.CoTTA(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
    )


def setup_vida(args, model, cfg):
    model = vida.configure_model(model, cfg)
    model_param, vida_param = vida.collect_params(model)
    optimizer = setup_optimizer_vida(
        model_param,
        vida_param,
        cfg.OPTIM.LR,
        cfg.OPTIM.ViDALR,
        cfg,
    )
    vida_model = vida.ViDA(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        unc_thr=args.unc_thr,
        ema=cfg.OPTIM.MT,
        ema_vida=cfg.OPTIM.MT_ViDA,
    )
    logger.info("model for adaptation: %s", model)
    logger.info("optimizer for adaptation: %s", optimizer)
    return vida_model


def setup_doco(args, model, cfg):
    reg = cfg.OPTIM.DOCO_BETA
    logger.info("Current hyperparameter cfg.OPTIM.DOCO_BETA value: %s", reg)

    model = doco.configure_model(model, cfg)
    prompt_params = doco.collect_params(model)
    optimizer = setup_optimizer(prompt_params, cfg)
    doco_model = doco.DOCO(
        model,
        optimizer,
        ema_alpha=cfg.OPTIM.DOCO_EMA_ALPHA,
        warmup_step=cfg.OPTIM.STEPS,
        gmm_pool_size=cfg.OPTIM.DOCO_GMM_POOL_SIZE,
        reg_beta=reg,
        lr=cfg.OPTIM.LR,
    )
    logger.info("-----<<<<<<<<<<<  Current call uses the DOCO algorithm ⭐  >>>>>>>>>>>>----------")

    source_num = cfg.SRC_NUM_SAMPLES
    logger.info(">>>>>>>>>>>>  Select %s source-domain samples to build train_info   >>>>>>>>>>>>>>>", source_num)
    doco_model.obtain_src_stat(data_path=args.data_dir, num_samples=source_num)
    return doco_model


def setup_sar_come(args, model, cfg):
    model = sar_come.configure_model(model)
    params, _ = sar_come.collect_params(model)

    base_optimizer = torch.optim.SGD
    lr_for_vit = _resolve_vit_lr(cfg)
    logger.info("Current learning rate for SARCOME: %s", lr_for_vit)

    optimizer = SAM(params, base_optimizer, lr=lr_for_vit, momentum=0.9)

    margin = math.log(1000) * 0.4
    logger.info("Preset margin for this SAR_COME run: %s", margin)

    return sar_come.SAR_COME(model, optimizer, margin_e0=margin)


def setup_eata_come(args, net, cfg):
    args.exp_type = "open-world"
    args.data = f"{args.data_dir}/ImageNet"
    args.method = "EAT_COME"
    args.corruption = "original"
    args.fisher_size = cfg.FISHER_SIZE

    fisher_dataset, fisher_loader = prepare_test_data(args)
    fisher_dataset.set_dataset_size(args.fisher_size)
    fisher_dataset.switch_mode(True, False)

    logger.info("Fisher_size ablation value for EATACOME: %s", args.fisher_size)

    net = eata_come.configure_model(net)
    params, _ = eata_come.collect_params(net)

    fishers = _compute_fishers(net, params, fisher_loader)
    logger.info("compute fisher matrices finished")

    lr_for_vit = _resolve_vit_lr(cfg)
    logger.info("lr_for_vit of EATA_COME is %s", lr_for_vit)

    optimizer = torch.optim.SGD(params, lr_for_vit, momentum=0.9)
    args.fisher_beta = 2000
    args.e_margin = math.log(1000) * 0.40
    args.d_margin = 0.05

    logger.info(
        "args.fisher_beta = %s, args.e_margin = %s, args.d_margin = %s",
        args.fisher_beta,
        args.e_margin,
        args.d_margin,
    )

    return eata_come.EATA_COME(
        net,
        optimizer,
        fishers,
        args.fisher_beta,
        e_margin=args.e_margin,
        d_margin=args.d_margin,
    )


def setup_ostta(args, base_model, cfg):
    base_model = ostta.configure_model(base_model)
    params, _ = ostta.collect_params(base_model)
    optimizer = setup_optimizer(params, cfg)
    return ostta.OSTTA(
        base_model,
        optimizer,
        steps=1,
        episodic=False,
        alpha=[0.5],
        criterion="ent",
    )


def setup_stamp(args, base_model, cfg):
    params, _ = STAMP.collect_params(base_model)
    base_optimizer = optim.SGD
    optimizer = SAM(params, base_optimizer, lr=cfg.OPTIM.LR, rho=0.05)
    return STAMP(base_model, optimizer, cfg.STAMP.ALPHA, cfg.num_classes)


def setup_sar(args, base_model, cfg):
    model = sar.configure_model(base_model)
    params, _ = sar.collect_params(model)
    base_optimizer = optim.SGD
    optimizer = SAM(params, base_optimizer, lr=cfg.OPTIM.LR, momentum=0.9)
    return sar.SAR(model, optimizer, margin_e0=math.log(1000) * 0.40)


def setup_eata(args, base_model, cfg):
    return setup_eata_unient(
        args,
        base_model,
        cfg,
        criterion_override="ent",
        method_label="EATA",
        original_eata=True,
    )


def setup_eata_unient(
    args,
    base_model,
    cfg,
    criterion_override=None,
    method_label="EATA UniEnt",
    original_eata=False,
):
    fisher_size = cfg.FISHER_SIZE
    logger.info("%s Fisher sample size: %s", method_label, fisher_size)

    fisher_alpha = cfg.OPTIM.EATA_UNIENT_FISHER_ALPHA
    e_margin = cfg.OPTIM.EATA_UNIENT_E_MARGIN
    d_margin = cfg.OPTIM.EATA_UNIENT_D_MARGIN
    fisher_data_root = f"{args.data_dir}/ImageNet/train"

    fisher_dataset = datasets.ImageFolder(
        fisher_data_root,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    )
    logger.info("%s Fisher data root: %s", method_label, fisher_data_root)
    sampled_indices = torch.randperm(len(fisher_dataset))[:fisher_size]
    sampler = SubsetRandomSampler(sampled_indices)
    fisher_loader = DataLoader(
        fisher_dataset,
        batch_size=cfg.TEST.BATCH_SIZE * 2,
        sampler=sampler,
    )

    base_model = eata_unient.configure_model(base_model)
    params, _ = eata_unient.collect_params(base_model)
    fishers = _compute_fishers(base_model, params, fisher_loader)

    optimizer = setup_optimizer(params, cfg)
    unient_alpha = cfg.OPTIM.EATA_UNIENT_ALPHA
    eata_or_eataunient = criterion_override or cfg.OPTIM.EATA_UNIENT_CRITERION

    eata_unient_model = eata_unient.EATA(
        cfg.MODEL.ARCH,
        base_model,
        optimizer,
        fishers,
        fisher_alpha,
        e_margin=e_margin,
        d_margin=d_margin,
        alpha=unient_alpha,
        criterion=eata_or_eataunient,
        original_eata=original_eata,
    )

    if original_eata:
        logger.info("criterion is ent; using the original EATA")
    elif eata_or_eataunient == "ent":
        logger.info("criterion is ent; using EATA with UniEnt batch-level entropy")
    elif eata_or_eataunient == "ent_ind":
        logger.info("criterion is ent_ind; using EATA with hard pseudo-csID entropy")
    elif eata_or_eataunient == "ent_ind_ood":
        logger.info("criterion is ent_ind_ood; using EATA with hard UniEnt")
    elif eata_or_eataunient == "ent_unf":
        logger.info("criterion is ent_unf; using EATA with UniEnt+")
    else:
        raise NotImplementedError

    logger.info(
        "fisher_size = %s, fisher_alpha = %s, e_margin = %s, d_margin = %s, unient_alpha = %s, criterion = %s",
        fisher_size,
        fisher_alpha,
        e_margin,
        d_margin,
        unient_alpha,
        eata_or_eataunient,
    )

    return eata_unient_model


def setup_dpcore(args, model, cfg):
    model = dpcore.configure_model(model, cfg)
    prompt_params = dpcore.collect_params(model)
    optimizer = setup_optimizer(prompt_params, cfg)

    thr = cfg.OPTIM.DPCORE_THR_RHO
    logger.info("Standard threshold for the original dpcore algorithm: %s", thr)

    prompt_length = cfg.OPTIM.DPCORE_PROMPT_NUM
    logger.info("Prompt length for the dpcore algorithm uses the standard length: %s", prompt_length)

    dpcore_model = dpcore.DPCore(
        model,
        optimizer,
        temp_tau=cfg.OPTIM.DPCORE_TEMP_TAU,
        thr_rho=thr,
        ema_alpha=cfg.OPTIM.DPCORE_EMA_ALPHA,
        E_OOD=cfg.OPTIM.STEPS,
    )
    

    dpcore_model.obtain_src_stat(
        data_path=args.data_dir,
        num_samples=cfg.SRC_NUM_SAMPLES,
    )
    return dpcore_model


def setup_optimizer(params, cfg):
    if cfg.OPTIM.METHOD == "Adam":
        return optim.Adam(
            params,
            lr=cfg.OPTIM.LR,
            betas=(cfg.OPTIM.BETA, 0.999),
            weight_decay=cfg.OPTIM.WD,
        )
    elif cfg.OPTIM.METHOD == "SGD":
        return optim.SGD(
            params,
            lr=cfg.OPTIM.LR,
            momentum=0.9,
            dampening=0,
            weight_decay=cfg.OPTIM.WD,
            nesterov=True,
        )
    elif cfg.OPTIM.METHOD == "AdamW":
        return optim.AdamW(
            params,
            lr=cfg.OPTIM.LR,
            betas=(cfg.OPTIM.BETA, 0.999),
            weight_decay=cfg.OPTIM.WD,
        )
    else:
        raise NotImplementedError(f"Unsupported optimizer: {cfg.OPTIM.METHOD}")


def setup_optimizer_vida(params, params_vida, model_lr, vida_lr, cfg):
    if cfg.OPTIM.METHOD == "Adam":
        return optim.Adam(
            [
                {"params": params, "lr": model_lr},
                {"params": params_vida, "lr": vida_lr},
            ],
            lr=1e-5,
            betas=(cfg.OPTIM.BETA, 0.999),
            weight_decay=cfg.OPTIM.WD,
        )

    elif cfg.OPTIM.METHOD == "SGD":
        return optim.SGD(
            [
                {"params": params, "lr": model_lr},
                {"params": params_vida, "lr": vida_lr},
            ],
            momentum=cfg.OPTIM.MOMENTUM,
            dampening=cfg.OPTIM.DAMPENING,
            nesterov=cfg.OPTIM.NESTEROV,
            lr=1e-5,
            weight_decay=cfg.OPTIM.WD,
        )
    else:
        raise NotImplementedError(f"Unsupported optimizer for ViDA: {cfg.OPTIM.METHOD}")


def _resolve_vit_lr(cfg):
    if cfg.MODEL.ARCH == "Standard_R50":
        return cfg.OPTIM.LR
    elif cfg.MODEL.ARCH == "Standard_VITB":
        return (0.001 / 64) * cfg.TEST.BATCH_SIZE
    else:
        raise NotImplementedError(f"Unsupported model arch: {cfg.MODEL.ARCH}")


def _compute_fishers(model, params, loader):
    ewc_optimizer = optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = nn.CrossEntropyLoss().cuda()

    for iter_, (images, targets) in enumerate(loader, start=1):
        images = images.cuda()
        targets = targets.cuda()

        outputs = model(images)
        _, targets = outputs.max(1)
        loss = train_loss_fn(outputs, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            if iter_ > 1:
                fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
            else:
                fisher = param.grad.data.clone().detach() ** 2

            if iter_ == len(loader):
                fisher = fisher / iter_

            fishers[name] = [fisher, param.data.clone().detach()]

        ewc_optimizer.zero_grad()

    del ewc_optimizer
    return fishers
