import argparse
import os
import subprocess
import sys
from pathlib import Path


SEEDS = [0, 1, 2, 3, 4]

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
        ],
    },
    "FQ_noswad": {
        "algorithm": "FQ",
        "swad": "False",
        "extra_args": [
            "--fft_quant", "1",
            "--fq_steps", "2000",
            "--fq_ramp", "500",
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
    return parser.parse_args()


def build_command(repo_dir, method_name, seed,batch,dataset,steps):
    method = METHODS[method_name]
    experiment_name = f"{dataset}_{method_name}_seed{seed}"

    command = [
        sys.executable,
        str(repo_dir / "train_all.py"),
        experiment_name,
        "--dataset", dataset,
        "--data_dir", str(repo_dir / "dataset"),
        "--algorithm", method["algorithm"],
        "--steps", steps,
        "--checkpoint_freq", "100",
        "--batch_size", str(batch),
        "--optimizer", "adam",
        "--lr", "5e-5",
        "--weight_decay", "0",
        "--resnet_dropout", "0",
        "--resnet18", "False",
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

    for seed in SEEDS:
        command = build_command(
            repo_dir,
            args.algorithm,
            seed,
            args.batch,
            args.dataset,
            args.steps,
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
