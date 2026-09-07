import argparse
import os
import subprocess
import sys
from pathlib import Path


SEEDS = [0, 1, 2, 3, 4]

_ALOFT_MASK07_ARGS = [
    "--aloft_alpha", "1.0",
    "--aloft_mask_ratio", "0.7",
    "--aloft_perturb_prob", "1.0",
]

_ALOFT_STRUCT_ARGS = _ALOFT_MASK07_ARGS + [
    "--aloft_struct_head_channels", "64",
    "--aloft_struct_warmup", "200",
    "--aloft_struct_ramp", "500",
    "--aloft_struct_dir_weight", "0.05",
    "--aloft_struct_stroke_weight", "0.05",
    "--aloft_struct_closure_weight", "0.05",
    "--aloft_struct_cldice_weight", "0.02",
    "--aloft_struct_skeleton_iters", "10",
]

METHODS = {
    "ERM": {
        "algorithm": "ERM",
        "swad": "False",
        "extra_args": [],
    },
    "GGA": {
        "algorithm": "ERM_GGA",
        "swad": "LossValley",
        "extra_args": [
            "--start_step", "100",
            "--end_step", "200",
            "--neighborhoodSize", "1e-5",
        ],
    },
    "Fish": {
        "algorithm": "Fish",
        "swad": "False",
        "extra_args": [
            "--meta_lr", "0.5",
        ],
    },
    "Mixup": {
        "algorithm": "Mixup",
        "swad": "False",
        "extra_args": [
            "--mixup_alpha", "0.2",
        ],
    },
    "SagNet": {
        "algorithm": "SagNet",
        "swad": "False",
        "extra_args": [
            "--sag_w_adv", "0.1",
        ],
    },
    # SWAD is ERM trained with LossValley model averaging.
    "SWAD": {
        "algorithm": "ERM",
        "swad": "LossValley",
        "extra_args": [],
    },
    # 频带均衡：ERM+SWAD 上加逐样本径向频谱归一化。对照臂直接用上面的 SWAD。
    "BEQ": {
        "algorithm": "ERM",
        "swad": "LossValley",
        "extra_args": [
            "--band_eq", "1",
            "--band_eq_mode", "both",
            "--band_eq_preserve_total", "True",
            "--band_eq_target_mode", "source_mean",
        ],
    },
    # 训练侧保留原始多样性，只在推理时把目标域输入归一到源域均值
    "BEQ_test": {
        "algorithm": "ERM",
        "swad": "LossValley",
        "extra_args": [
            "--band_eq", "1",
            "--band_eq_mode", "test_only",
            "--band_eq_preserve_total", "True",
            "--band_eq_target_mode", "source_mean",
        ],
    },
    # 消融：只归一源域、目标域原样，用来隔离两侧各自的作用
    "BEQ_train": {
        "algorithm": "ERM",
        "swad": "LossValley",
        "extra_args": [
            "--band_eq", "1",
            "--band_eq_mode", "train_only",
            "--band_eq_preserve_total", "True",
            "--band_eq_target_mode", "source_mean",
        ],
    },
    # 消融：目标改用 1/f^2 解析剖面（不统计源域，无泄漏、无逐划分缓存）
    "BEQ_fixed": {
        "algorithm": "ERM",
        "swad": "LossValley",
        "extra_args": [
            "--band_eq", "1",
            "--band_eq_preserve_total", "True",
            "--band_eq_target_mode", "fixed",
        ],
    },
    "Arith": {
        "algorithm": "Arith",
        "swad": "LossValley",
        "extra_args": [
            "--arith_meta_lr", "0.01",
        ],
    },
    "RSC": {
        "algorithm": "RSC",
        "swad": "False",
        "extra_args": [
            "--rsc_f_drop_factor", "0.3333333333",
            "--rsc_b_drop_factor", "0.3333333333",
        ],
    },

    "DANN": {
        "algorithm": "DANN",
        "swad": "False",
        "extra_args": [
            "--lr_g", "5e-5",
            "--lr_d", "5e-5",
            "--weight_decay_g", "0",
            "--weight_decay_d", "0",
            "--lambda", "1",
            "--grad_penalty", "0",
            "--d_steps_per_g_step", "1",
            "--beta1", "0.5",
            "--mlp_width", "256",
            "--mlp_depth", "3",
            "--mlp_dropout", "0",
        ],
    },
    "CSU": {
        "algorithm": "CSU",
        "swad": "False",
        "extra_args": [
            "--csu_p", "0.5",
            "--csu_alpha", "0.3",
        ],
    },
    "CSU_SWAD": {
        "algorithm": "CSU",
        "swad": "LossValley",
        "extra_args": [
            "--csu_p", "0.5",
            "--csu_alpha", "0.3",
        ],
    },
    "ALOFT_E": {
        "algorithm": "ALOFT_E",
        "swad": "LossValley",
        "extra_args": [
            "--aloft_alpha", "1.0",
            "--aloft_mask_ratio", "0.5",
            "--aloft_perturb_prob", "1.0",
        ],
    },
    "AWWSL_rev_E": {
            "algorithm": "AWWSL_rev_E",
            "swad": "LossValley",
            "extra_args": [
                "--aloft_alpha", "1.0",
                "--aloft_mask_ratio", "0.5",
                "--aloft_perturb_prob", "1.0",
            ],
        },
    "AWWSL_E": {
        "algorithm": "AWWSL_E",
        "swad": "LossValley",
        "extra_args": [
            "--aloft_alpha", "1.0",
            "--aloft_mask_ratio", "0.5",
            "--aloft_perturb_prob", "1.0",
        ],
    },
    "ALOFT_rev_E": {
            "algorithm": "ALOFT_rev_E",
            "swad": "LossValley",
            "extra_args": [
                "--aloft_alpha", "1.0",
                "--aloft_mask_ratio", "0.5",
                "--aloft_perturb_prob", "1.0",
            ],
        },
    "ALOFT_LF_E_mask07": {
        "algorithm": "ALOFT_E",
        "swad": "LossValley",
        "extra_args": list(_ALOFT_MASK07_ARGS),
    },
    "ALOFT_CovLF_E": {
        "algorithm": "ALOFT_CovLF_E",
        "swad": "LossValley",
        "extra_args": list(_ALOFT_MASK07_ARGS),
    },
    "ALOFT_DomainLF_E": {
        "algorithm": "ALOFT_DomainLF_E",
        "swad": "LossValley",
        "extra_args": list(_ALOFT_MASK07_ARGS),
    },
    "ALOFT_HF_E_mask07": {
        "algorithm": "ALOFT_HF_E",
        "swad": "LossValley",
        "extra_args": list(_ALOFT_MASK07_ARGS),
    },
    "ALOFT_StructLF_E": {
        "algorithm": "ALOFT_StructLF_E",
        "swad": "LossValley",
        "extra_args": list(_ALOFT_STRUCT_ARGS),
    },
    "ALOFT_StructLF_Dir_E": {
        "algorithm": "ALOFT_StructLF_Dir_E",
        "swad": "LossValley",
        "extra_args": list(_ALOFT_STRUCT_ARGS),
    },
    "ALOFT_StructLF_Topo_E": {
        "algorithm": "ALOFT_StructLF_Topo_E",
        "swad": "LossValley",
        "extra_args": list(_ALOFT_STRUCT_ARGS),
    },
    "ALOFT_StructHF_E": {
        "algorithm": "ALOFT_StructHF_E",
        "swad": "LossValley",
        "extra_args": list(_ALOFT_STRUCT_ARGS),
    },
    "ALOFT_StructHF_Dir_E": {
        "algorithm": "ALOFT_StructHF_Dir_E",
        "swad": "LossValley",
        "extra_args": list(_ALOFT_STRUCT_ARGS),
    },
    "ALOFT_StructHF_Topo_E": {
        "algorithm": "ALOFT_StructHF_Topo_E",
        "swad": "LossValley",
        "extra_args": list(_ALOFT_STRUCT_ARGS),
    },
    "ALOFT_CB_rev_E": {
        "algorithm": "ALOFT_CB_rev_E",
        "swad": "LossValley",
        "extra_args": [
            "--aloft_alpha", "1.0",
            "--aloft_mask_ratio", "0.7",
            "--aloft_perturb_prob", "1.0",
            "--aloft_cb_codebook", "256",
            "--aloft_cb_group_size", "32",
            "--aloft_cb_bands", "3",
            "--aloft_cb_strength_max", "0.2",
            "--aloft_cb_decay", "0.99",
            "--aloft_cb_dead_patience", "200",
            "--aloft_cb_reservoir", "1024",
            "--fft_quant", "1",
            "--fq_steps", "2000",
            "--fq_ramp", "500",
        ],
    },
    "ALOFT_Sketch_rev_E": {
        "algorithm": "ALOFT_Sketch_rev_E",
        "swad": "LossValley",
        "extra_args": [
            "--aloft_alpha", "1.0",
            "--aloft_mask_ratio", "0.7",
            "--aloft_perturb_prob", "1.0",
            "--aloft_sketch_group_size", "32",
            "--aloft_sketch_radial_bands", "3",
            "--aloft_sketch_orientation_bins", "6",
            "--aloft_sketch_strength_max", "0.3",
            "--aloft_sketch_warmup", "500",
            "--aloft_sketch_ramp", "500",
            "--aloft_sketch_class_decay", "0.99",
            "--aloft_sketch_class_min_count", "20",
            "--aloft_sketch_ready_ratio", "0.5",
            "--aloft_sketch_gate_power", "0.5",
            "--aloft_sketch_topo_weight", "0.05",
            "--aloft_sketch_skeleton_iters", "10",
        ],
    },
    "ALOFT_SketchTopo_rev_E": {
        "algorithm": "ALOFT_SketchTopo_rev_E",
        "swad": "LossValley",
        "extra_args": [
            "--aloft_alpha", "1.0",
            "--aloft_mask_ratio", "0.7",
            "--aloft_perturb_prob", "1.0",
            "--aloft_sketch_group_size", "32",
            "--aloft_sketch_radial_bands", "3",
            "--aloft_sketch_orientation_bins", "6",
            "--aloft_sketch_strength_max", "0.3",
            "--aloft_sketch_warmup", "500",
            "--aloft_sketch_ramp", "500",
            "--aloft_sketch_class_decay", "0.99",
            "--aloft_sketch_class_min_count", "20",
            "--aloft_sketch_ready_ratio", "0.5",
            "--aloft_sketch_gate_power", "0.5",
            "--aloft_sketch_topo_weight", "0.05",
            "--aloft_sketch_skeleton_iters", "10",
        ],
    },
    "ALOFT_S": {
        "algorithm": "ALOFT_S",
        "swad": "LossValley",
        "extra_args": [
            "--aloft_alpha", "0.9",
            "--aloft_mask_ratio", "0.5",
            "--aloft_perturb_prob", "1.0",
        ],
    },
    "ALOFT_rev_S": {
            "algorithm": "ALOFT_rev_S",
            "swad": "LossValley",
            "extra_args": [
                "--aloft_alpha", "0.9",
                "--aloft_mask_ratio", "0.5",
                "--aloft_perturb_prob", "1.0",
            ],
        },
    "iDAG": {
        "algorithm": "iDAG",
        "swad": "False",
        "extra_args": [
            "--out_dim", "512",
            "--hidden_size", "512",
            "--num_hidden_layers", "0",
            "--dag_anneal_steps", "200",
            "--temperature", "0.07",
            "--ema_ratio", "0.99",
            "--lambda1", "0.01",
            "--lambda2", "0.01",
            "--rho", "1.0",
            "--alpha", "1.0",
            "--rho_max", "100.0",
            "--weight_mu", "1.0",
            "--weight_nu", "1.0",
        ],
    },
    "iDAG_SWAD": {
        "algorithm": "iDAG",
        "swad": "LossValley",
        "extra_args": [
            "--out_dim", "512",
            "--hidden_size", "512",
            "--num_hidden_layers", "0",
            "--dag_anneal_steps", "200",
            "--temperature", "0.07",
            "--ema_ratio", "0.99",
            "--lambda1", "0.01",
            "--lambda2", "0.01",
            "--rho", "1.0",
            "--alpha", "1.0",
            "--rho_max", "100.0",
            "--weight_mu", "1.0",
            "--weight_nu", "1.0",
        ],
    },
    "QTDoG": {
        "algorithm": "ERM",
        "swad": "LossValley",
        "extra_args": [
            "--quant", "1",
            "--q_steps", "2000",
        ],
    },
    "QTDoG_noswad": {
        "algorithm": "ERM",
        "swad": "False",
        "extra_args": [
            "--quant", "1",
            "--q_steps", "100",
        ],
    },
    "FQ": {
        "algorithm": "FQ",
        "swad": "LossValley",
        "extra_args": [
            "--fft_quant", "1",
            "--fq_steps", "2000",
            "--fq_ramp", "500",
            "--fq_levels","8",
        ],
    },
    "FQ_noswad": {
        "algorithm": "FQ",
        "swad": "False",
        "extra_args": [
            "--fft_quant", "1",
            "--fq_steps", "2000",
            "--fq_ramp", "500",
            "--fq_levels","8",
        ],
    },
    # ---- 机制分解：levels 32~256 四档结果几乎相同，说明起作用的可能是那个
    # 和 levels 无关的分位截断，而不是量化。这两档把它们拆开。对照臂用 SWAD。
    # 只截断不量化
    "FQ_clip": {
        "algorithm": "FQ",
        "swad": "LossValley",
        "extra_args": [
            "--fft_quant", "1",
            "--fq_steps", "2000",
            "--fq_ramp", "500",
            "--fq_mode", "clip",
            "--fq_quantile", "0.01",
        ],
    },
    # 只量化不截断
    "FQ_noclip": {
        "algorithm": "FQ",
        "swad": "LossValley",
        "extra_args": [
            "--fft_quant", "1",
            "--fq_steps", "2000",
            "--fq_ramp", "500",
            "--fq_mode", "quant",
            "--fq_levels", "256",
            "--fq_quantile", "0",
        ],
    },
    # ---- 码本：标量 round 只能表达「幅度接近」，表达不了「不同抽象度的
    # 同一结构是同一个东西」。K=256 对齐 DDG(arXiv 2504.06572)。
    # 频带内单码本
    "FQ_cbf": {
        "algorithm": "FQ",
        "swad": "LossValley",
        "extra_args": [
            "--fft_quant", "1",
            "--fq_steps", "2000",
            "--fq_ramp", "500",
            "--fq_mode", "cb_freq",
            "--fq_codebook", "1024",
            "--fq_groups", "1",
            "--fq_strength_max", "0.3",
            "--fq_mask_ratio", "0.9",
            "--fq_aux_weight", "0.05",
        ],
    },
    # 频带内乘积量化：等开销下容量大得多，单码本压太狠时的备胎
    "FQ_cbf_pq": {
        "algorithm": "FQ",
        "swad": "LossValley",
        "extra_args": [
            "--fft_quant", "1",
            "--fq_steps", "2000",
            "--fq_ramp", "500",
            "--fq_mode", "cb_freq",
            "--fq_codebook", "1024",
            "--fq_groups", "2",
            "--fq_strength_max", "0.3",
            "--fq_mask_ratio", "0.9",
            "--fq_aux_weight", "0.05",
        ],
    },
    # 低频码本，和之前 hi/lo 那组对齐
    "FQ_cbf_lo": {
        "algorithm": "FQ",
        "swad": "LossValley",
        "extra_args": [
            "--fft_quant", "1",
            "--fq_steps", "2000",
            "--fq_ramp", "500",
            "--fq_mode", "cb_freq",
            "--fq_band", "low",
            "--fq_codebook", "1024",
            "--fq_groups", "1",
            "--fq_strength_max", "0.5",
            "--fq_mask_ratio", "0.25",
            "--fq_aux_weight", "0.05",
        ],
    },
    # 全图码本，不碰频域，对标 DDG。机制最硬的一档
    "FQ_cbx": {
        "algorithm": "FQ",
        "swad": "LossValley",
        "extra_args": [
            "--fft_quant", "1",
            "--fq_steps", "2000",
            "--fq_ramp", "500",
            "--fq_mode", "cb_feat",
            "--fq_codebook", "1024",
            "--fq_groups", "1",
            "--fq_strength_max", "0.5",
            "--fq_aux_weight", "0.05",
        ],
    },
}

METHOD_NAMES = {
    method_name.lower(): method_name
    for method_name in METHODS
}


def parse_algorithm(value):
    method_name = METHOD_NAMES.get(value.lower())
    if method_name is None:
        choices = ", ".join(METHODS)
        raise argparse.ArgumentTypeError(
            f"unknown algorithm {value!r}; choose one of: {choices}"
        )
    return method_name


def parse_gpu(value):
    try:
        gpu_index = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "gpu must be a non-negative integer"
        ) from error

    if gpu_index < 0:
        raise argparse.ArgumentTypeError(
            "gpu must be a non-negative integer"
        )
    return gpu_index


def parse_backbone(value):
    from domainbed.backbones import normalize_backbone
    try:
        return normalize_backbone(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one HTP algorithm with seeds 0-4 on a selected GPU."
        )
    )
    parser.add_argument(
        "--algorithm",
        required=True,
        type=parse_algorithm,
        metavar="{" + ",".join(METHODS) + "}",
        help="algorithm to run; the value is case-insensitive",
    )
    parser.add_argument(
        "--gpu",
        required=True,
        type=parse_gpu,
        metavar="INDEX",
        help="physical CUDA GPU index, for example 0 or 1",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="batch size for training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HTP",
        help="dataset to use; default is HTP",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="5000",
        help="number of steps to train for; default is 5000",
    )
    parser.add_argument(
        "--backbone", type=parse_backbone, default="resnet",
        metavar="{resnet,vit}",
        help="backbone (case-insensitive); vit_b_16 is an alias for vit",
    )
    return parser.parse_args()


def build_command(repo_dir, method_name, seed,batch,dataset,steps, backbone="resnet"):
    method = METHODS[method_name]
    experiment_name = f"{dataset}_{method_name}_seed{seed}"

    backbone = parse_backbone(backbone)
    if backbone == "vit":
        experiment_name = f"{dataset}_{method_name}_vit_b_16_seed{seed}"

    command = [
        sys.executable,
        str(repo_dir / "train_all.py"),
        experiment_name,
        "--dataset", dataset,
        "--data_dir", str(repo_dir / "dataset"),
        "--algorithm", method["algorithm"],
        "--steps", steps,
        "--batch_size", str(batch),
        "--pretrained", "True",
        "--freeze_bn", "True",
        "--swad", method["swad"],
        "--seed", str(seed),
        "--trial_seed", str(seed),
        "--deterministic",
        "--cache", "disk",
        "--prebuild_loader",
        "--image_size", "224",
    ]
    command.extend(method["extra_args"])
    if backbone == "vit":
        command.extend(["--backbone", "vit"])
    return command


def main():
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent
    dataset_dir = repo_dir / "dataset" / args.dataset

    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"HTP dataset directory was not found: {dataset_dir}"
        )

    child_environment = os.environ.copy()
    child_environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(
        f"Algorithm: {args.algorithm}\n"
        f"Physical GPU: {args.gpu}\n"
        f"Seeds: {SEEDS}\n"
        f"Dataset: {dataset_dir}\n",
        flush=True,
    )

    print(f"Backbone: {args.backbone}", flush=True)

    for seed in SEEDS:
        command = build_command(
            repo_dir,
            args.algorithm,
            seed,
            args.batch,
            args.dataset,
            args.steps,
            args.backbone,
        )

        print("=" * 80, flush=True)
        print(
            f"Starting {args.algorithm}, seed={seed}, "
            f"physical GPU={args.gpu}",
            flush=True,
        )
        print(
            subprocess.list2cmdline(command),
            flush=True,
        )
        print("=" * 80, flush=True)

        subprocess.run(
            command,
            cwd=repo_dir,
            env=child_environment,
            check=True,
        )


if __name__ == "__main__":
    main()
