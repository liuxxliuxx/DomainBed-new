# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from random import random

import numpy as np


def _hparams(algorithm, dataset, random_state):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ["Debug28", "RotatedMNIST", "ColoredMNIST"]

    hparams = {}

    hparams["data_augmentation"] = (True, True)
    hparams["val_augment"] = (False, False)  # augmentation for in-domain validation set
    hparams["resnet18"] = (False, False)
    hparams["resnet_dropout"] = (0.0, random_state.choice([0.0, 0.1, 0.5]))
    hparams["class_balanced"] = (False, False)
    hparams["optimizer"] = ("adam", "adam")

    hparams["freeze_bn"] = (True, True)
    hparams["pretrained"] = (True, True)  # only for ResNet

    if dataset not in SMALL_IMAGES:
        hparams["lr"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        if dataset == "DomainNet":
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5)))
        else:
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5.5)))
        if algorithm == "ARM":
            hparams["batch_size"] = (8, 8)
    else:
        hparams["lr"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
        hparams["batch_size"] = (64, int(2 ** random_state.uniform(3, 9)))

    if dataset in SMALL_IMAGES:
        hparams["weight_decay"] = (0.0, 0.0)
    else:
        hparams["weight_decay"] = (0.0, 10 ** random_state.uniform(-6, -2))

    if algorithm in ["DANN", "CDANN"]:
        if dataset not in SMALL_IMAGES:
            hparams["lr_g"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
            hparams["lr_d"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        else:
            hparams["lr_g"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
            hparams["lr_d"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))

        if dataset in SMALL_IMAGES:
            hparams["weight_decay_g"] = (0.0, 0.0)
        else:
            hparams["weight_decay_g"] = (0.0, 10 ** random_state.uniform(-6, -2))

        hparams["lambda"] = (1.0, 10 ** random_state.uniform(-2, 2))
        hparams["weight_decay_d"] = (0.0, 10 ** random_state.uniform(-6, -2))
        hparams["d_steps_per_g_step"] = (1, int(2 ** random_state.uniform(0, 3)))
        hparams["grad_penalty"] = (0.0, 10 ** random_state.uniform(-2, 1))
        hparams["beta1"] = (0.5, random_state.choice([0.0, 0.5]))
        hparams["mlp_width"] = (256, int(2 ** random_state.uniform(6, 10)))
        hparams["mlp_depth"] = (3, int(random_state.choice([3, 4, 5])))
        hparams["mlp_dropout"] = (0.0, random_state.choice([0.0, 0.1, 0.5]))
    elif algorithm in ['Fish', 'Fish_GGA']:
        hparams["meta_lr"] = (0.5, random_state.choice([0.05, 0.1, 0.5]))
    elif algorithm in ["RSC", "RSC_GGA"]:
        hparams["rsc_f_drop_factor"] = (1 / 3, random_state.uniform(0, 0.5))
        hparams["rsc_b_drop_factor"] = (1 / 3, random_state.uniform(0, 0.5))
    elif algorithm in ["SagNet", "SagNet_GGA"]:
        hparams["sag_w_adv"] = (0.1, 10 ** random_state.uniform(-2, 1))
    elif algorithm in ["IRM", "IRM_GGA"]:
        hparams["irm_lambda"] = (1e2, 10 ** random_state.uniform(-1, 5))
        hparams["irm_penalty_anneal_iters"] = (
            500,
            int(10 ** random_state.uniform(0, 4)),
        )
    elif algorithm in ["Mixup", "OrgMixup", "Mixup_GGA"]:
        hparams["mixup_alpha"] = (0.2, 10 ** random_state.uniform(-1, -1))
    elif algorithm == "GroupDRO":
        hparams["groupdro_eta"] = (1e-2, 10 ** random_state.uniform(-3, -1))
    elif algorithm in ("MMD", "MMD_GGA", "CORAL", "CORAL_GGA"):
        hparams["mmd_gamma"] = (1.0, 10 ** random_state.uniform(-1, 1))
    elif algorithm in ("MLDG", "SOMLDG", "MLDG_GGA"):
        hparams["mldg_beta"] = (1.0, 10 ** random_state.uniform(-1, 1))
    elif algorithm in ["MTL", "MTL_GGA"]:
        hparams["mtl_ema"] = (0.99, random_state.choice([0.5, 0.9, 0.99, 1.0]))
    elif algorithm in ["VREx", "VREx_GGA"]:
        hparams["vrex_lambda"] = (1e1, 10 ** random_state.uniform(-1, 5))
        hparams["vrex_penalty_anneal_iters"] = (
            500,
            int(10 ** random_state.uniform(0, 4)),
        )
    elif algorithm in ["SAGM_DG", "SAGM_DG_CUSTOM", "SAGM_DG_CUSTOM_ALL",
                       "SAM", "SAM_CUSTOM", "SAM_CUSTOM_ALL", "SAM_CALC_LOSS",
                       "SAGM_GGA", "SAM_CUSTOM_ALL_SIM","GSAM", "GSAM_GGA", "SAM_GGA"]:
        hparams["rho"] = (0.05, random_state.choice([0.01, 0.02, 0.05, 0.1]))
        hparams["alpha"] = (0.001, random_state.choice([0.01, 0.02, 0.05, 0.1]))

    elif algorithm == "CutMix":
        hparams["beta"] = (1.0, 1.0)
        # cutmix_prob is set to 1.0 for ImageNet and 0.5 for CIFAR100 in the original paper.
        hparams["cutmix_prob"] = (1.0, 1.0)

    elif algorithm in ["ANDMask", "ANDMask_GGA"]:
        hparams['tau'] = (1.0, 1.0 ** random_state.uniform(0.5, 1))

    elif algorithm in ["GroupDRO", "GroupDRO_GGA"]:
        hparams['groupdro_eta'] = (1e-2, 10 ** random_state.uniform(-3, -1))

    elif algorithm in ['DANN', 'DANN_GGA', 'CDANN','CDANN_GGA']:
        hparams['lr_g'] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        hparams['lr_d'] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        hparams['weight_decay_g'] = (0., 10 ** random_state.uniform(-6, -2))
        hparams['lambda'] = (1.0, 10 ** random_state.uniform(-2, 2))
        hparams['weight_decay_d'] = (0., 10 ** random_state.uniform(-6, -2))
        hparams['d_steps_per_g_step'] = (1, int(2 ** random_state.uniform(0, 3)))
        hparams['grad_penalty'] = (0., 10 ** random_state.uniform(-2, 1))
        hparams['beta1'] = (0.5, random_state.choice([0., 0.5]))
        hparams['mlp_width'] = (256, int(2 ** random_state.uniform(6, 10)))
        hparams['mlp_depth'] = (3, int(random_state.choice([3, 4, 5])))
        hparams['mlp_dropout'] = (0., random_state.choice([0., 0.1, 0.5]))
    elif algorithm == "Arith":
        hparams["arith_meta_lr"] = (1e-2,10 ** random_state.uniform(-3, -1))
    elif algorithm in ["ALOFT_E", "ALOFT_S", "ALOFT_DG"]:
        # 论文 4.2 节："we set the perturbation strength alpha ... to 1.0 in
        # ALOFT-E and 0.9 in ALOFT-S"
        default_alpha = 0.9 if algorithm == "ALOFT_S" else 1.0
        hparams["aloft_alpha"] = (default_alpha, random_state.choice([0.5, 0.7, 0.9, 1.0]))
        # 论文 4.2 节：r = 0.5 for PACS/VLCS/Digits-DG, 0.25 for OfficeHome
        default_r = 0.25 if dataset == "OfficeHome" else 0.5
        hparams["aloft_mask_ratio"] = (default_r, random_state.choice([0.25, 0.5, 0.75]))
        hparams["aloft_perturb_prob"] = (1.0, random_state.choice([0.5, 1.0]))
        # 论文 Fig.3 里 ALOFT 出现在每个 core block；ResNet 的对应位置是每个 stage 之后
        hparams["aloft_positions"] = (["layer1", "layer2", "layer3"],) * 2
    elif algorithm == "iDAG":
        # LightEncoder 结构：2048 -> hidden_size -> out_dim
        hparams["hidden_size"] = (512, 512)
        hparams["out_dim"] = (512, 512)
        hparams["num_hidden_layers"] = (0, 0)
        # DAG 相关损失的预热步数，之前只训分类 + 跨域对比
        hparams["dag_anneal_steps"] = (200, int(random_state.choice([200, 400, 600, 800])))
        # 两个原型对比损失的温度
        hparams["temperature"] = (0.07, random_state.uniform(0.07, 0.01))
        # 原型 EMA 动量
        hparams["ema_ratio"] = (0.99, random_state.uniform(0.99, 0.999))
        # lambda1: 邻接矩阵 L1 稀疏权重；lambda2: 原型重建损失权重
        hparams["lambda1"] = (0.01, random_state.uniform(0.01, 1.0))
        hparams["lambda2"] = (0.01, random_state.uniform(0.01, 1.0))
        # 增广拉格朗日：rho 是二次罚系数，alpha 是乘子，rho_max 是上限
        hparams["rho_max"] = (100.0, 10 ** random_state.uniform(1.0, 6.0))
        hparams["rho"] = (1.0, 1.0)
        hparams["alpha"] = (1.0, 1.0)
        # weight_nu: 跨域原型对比权重；weight_mu: 不变原型对比权重
        hparams["weight_nu"] = (1.0, random_state.uniform(1.0, 2.0))
        hparams["weight_mu"] = (1.0, random_state.uniform(1.0, 2.0))
    return hparams


def default_hparams(algorithm, dataset):
    dummy_random_state = np.random.RandomState(0)
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, dummy_random_state).items()}


def random_hparams(algorithm, dataset, seed):
    random_state = np.random.RandomState(seed)
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, random_state).items()}
