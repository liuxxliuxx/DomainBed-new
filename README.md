# GGA
The official codes of our CVPR2025 paper: [Gradient-Guided Annealing for Domain Generalization](https://arxiv.org/abs/2502.20162)

In this paper we observe that the initial iterations of model training play a key 
role in domain generalization effectiveness, since the loss landscape may be 
significantly different across the training and test distributions, contrary 
to the case of i.i.d. data. Conflicts between gradients of the loss 
components of each domain lead the optimization procedure to undesirable 
local minima that do not capture the domain-invariant features of the target 
classes. We propose alleviating domain conflicts in model optimization, by 
iteratively annealing the parameters of a model in the early stages of 
training and searching for points where gradients align between domains. By 
discovering a set of parameter values where gradients are updated towards the 
same direction for each data distribution present in the training set, the 
proposed **Gradient-Guided Annealing (GGA)** algorithm encourages models to seek 
out minima that exhibit improved robustness against domain shifts.

<p align="center">
    <img src="./assets/gga_concept.png" width="70%" />
</p>

Note that this project is built upon [SWAD](https://github.com/khanrc/swad) and [DomainBed](https://github.com/facebookresearch/DomainBed/).


## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

### Environments

Environment details used for our study.

```
Python: 3.10.12
PyTorch: 2.0.1
Torchvision: 0.15.2
CUDA: 11.8
```

## How to Run

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

```sh
python train_all.py exp_name --dataset <dataset> --data_dir /my/datasets/path --trial_seed <seed> --algorithm <algorithm> --checkpoint_freq 100 --lr <lr> --weight_decay 1e-4 --resnet_dropout 0.5 --swad False
```

### Backbone: ResNet / ViT

所有现有算法及 `run_all.py` 配置可选择 backbone，默认仍为 ResNet：

```sh
python run_all.py --algorithm ALOFT_rev_E --gpu 0 --dataset HTP
python run_all.py --algorithm ALOFT_rev_E --gpu 0 --dataset HTP --backbone vit
python run_all.py --algorithm QTDoG --gpu 0 --dataset HTP --backbone vit --batch 8
python train_all.py ViT_ERM --algorithm ERM --dataset HTP --data_dir ./dataset --backbone vit
```

- `--backbone` 不区分大小写；`vit_b_16` 是 `vit` 的别名。默认参数和显式 `resnet` 保留原实验名称及训练配置，ViT 实验名称增加 `_vit_b_16`。
- ViT 使用 torchvision ViT-B/16、ImageNet-1K V1 权重，全量微调，返回 768 维 CLS 特征。`--pretrained False` 禁止下载预训练权重。
- 沿用现有优化器、学习率、batch size、增强、`resnet_dropout` 和 SWAD 配置，不自动替换训练配方。显存不足时通过 `--batch` 自行减小每域 batch size。
- 默认输入 224×224；其他正方形边长须能被 16 整除，预训练位置编码按目标尺寸插值。ARM 的附加通道按原 ResNet 的通道复制规则处理。
- Attention 使用局部的矩阵乘法实现，支持高阶梯度，不修改全局后端开关。

中间特征模块仅处理 patch token 的二维网格，不直接修改 CLS。`layer1/2/3/4` 分别对应第 3/6/9/11 个 Transformer block 后，最后一个 block 保留用于 CLS 汇总。`conv1` 对应 patch embedding 后、位置编码前，`maxpool` 对应位置编码与 embedding dropout 后。FQ 仍逐 block 插入，默认覆盖第 1–9 个 block。

ALOFT/AWWSL/CSU/MixStyle、结构头和码本复用原有计算及 train/eval 行为。14×14 网格上的 Sketch 方向／径向划分可能出现空单元，其他小尺寸上的 ALOFT-CB 也可能出现空频带；仅 ViT 分支允许这些空单元使用有限占位统计并跳过实际匹配，保留原频率掩码、不上采样特征。ResNet 仍保留原来的空单元检查。

本次不修正已有算法行为：`ALOFT_HF_E` 仍沿用现有低频配置，真正的高频对照使用 `ALOFT_rev_E`。也没有将当前 complex 扰动改为 amplitude，或修改 `mask_ratio` 的含义。

ViT 的 QTDoG 在 `q_steps` 后量化 patch embedding 卷积、Attention 的 QKV/输出投影及 MLP 线性权重；分类头、bias、LayerNorm 和 embedding 保持浮点。不新增激活量化。新增 LSQ 尺度加入优化器，已有权重对象与优化器状态保留。ResNet 的原卷积量化分支不变。

恢复 ViT checkpoint 时，先用其中的 `model_hparams` 构建相同算法，再调用下述函数；它会按 checkpoint 元数据重建可选的 QTDoG 模块：

```python
from domainbed.quan.vit import load_vit_checkpoint
load_vit_checkpoint(algorithm, checkpoint, strict=True)
```

测试命令（不下载预训练权重、不启动长训练）：

```sh
python -m unittest discover -s tests -p "test_*.py" -v
python tests/check_backbone_compatibility.py --reference /path/to/pre-edit/source
```

兼容性脚本在独立进程中比较修改前、修改后默认、显式 ResNet 三条路径的模型、梯度、优化器、随机状态和预测哈希。CUDA 专属更新在没有 CUDA 时明确跳过。CPU 测试通过不代表完成了完整 GPU 训练结果复现。

### CSU

CSU uses the repository's ResNet-50 backbone and inserts correlated style
perturbations after `maxpool` and `layer1` by default. A single experiment can
be started with:

```sh
python train_all.py CSU0 --dataset SKET --data_dir ./dataset --algorithm CSU --csu_p 0.5 --csu_alpha 0.3 --swad False
```

The five-seed runner exposes the original objective and the SWAD variant:

```sh
python run_all.py --algorithm CSU --gpu 0 --dataset SKET
python run_all.py --algorithm CSU_SWAD --gpu 0 --dataset SKET
```

### HTP: ALOFT-E + StableNet

新增算法 `ALOFT_Stable_E`，原有 `ALOFT_E`、`StableNet` 和 runner 配方不变。
组合默认对齐 `ALOFT_LF_E_mask07`，不是 runner 中 `mask_ratio=0.5` 的 `ALOFT_E`。

在服务器仓库根目录运行，默认 ResNet50、seed 0–4，每个 seed 完成三个目标域的留一训练：

```sh
python run_all.py --algorithm ALOFT_Stable_E --gpu 0 --dataset HTP --batch 32
```

如需同配置重跑 baseline：

```sh
python run_all.py --algorithm ALOFT_LF_E_mask07 --gpu 0 --dataset HTP --batch 32
```

`--gpu` 指物理 GPU 编号。两个命令都启用 `LossValley` SWAD；计划步数为 5000，
保留原有 SWAD 早停。学习率、优化器、划分和图像增强沿用当前配置，不单独更换。
ViT 仍可追加 `--backbone vit`，但首轮建议先比较同一 ResNet50 骨干。

#### 组合实现

1. 同一批已完成图像增强的输入，先经过共享骨干的无梯度、eval-mode 前向。
   此分支关闭 ALOFT 和 dropout，不更新 BN 统计，结束后恢复每个模块的原始模式。
   “未扰动”只针对模型内部扰动，不代表输入取消了原有图像增强。
2. 保留原版 StableNet 的随机 Fourier 特征和权重内循环，只使用上述分支的
   detached 特征估权；其梯度不进入骨干或分类头。
3. 将原始权重按源域重新归一，保留该域原来的 batch 占比，再与均匀权重混合：
   `w_i = (1 - eta) / B + eta * q_i`。每域 batch 相同，因此 HTP 两个源域各占 0.5。
   实现先在每个域内对 raw logits 做 softmax，避免全局 softmax 下溢后某域权重为零。
   这是原内循环输出的后处理，不是重新求解带域约束的最优权重。
4. 恢复训练模式，再做正常 ALOFT 前向；仅对逐样本交叉熵加权求和，不新增直接
   作用于骨干的去相关损失，不使用目标域或额外标签监督。
5. 历史缓存保存未扰动特征与原始权重 logits：前 10 次缓存更新取累计平均，之后
   使用保留系数 0.9 的 EMA。无历史时使用当前未扰动特征，避免将零向量作为历史样本。
   缓存仍是固定 batch 大小的移动平均，不是按样本 ID 索引的完整记忆库。

推理沿用原 ALOFT-E 的无扰动路径，不估计权重、不增加前向分支。
训练多一次无梯度骨干前向，以及 StableNet 的权重内循环；不新增可学习网络参数。

#### 默认参数与诊断

| 参数 | 默认值 |
| --- | --- |
| `aloft_alpha` / `aloft_mask_ratio` / `aloft_perturb_prob` | `1.0 / 0.7 / 1.0` |
| `aloft_positions` | `layer1, layer2, layer3` |
| `stable_mix_max` | `0.2` |
| `stable_warmup_steps` / `stable_ramp_steps` | `100 / 200` |
| `stable_epochb` / `stable_lrbl` / `stable_lambdap` | `20 / 1.0 / 70.0` |
| `stable_presave_ratio` | `0.9` |

按从 0 开始的更新计数，step 0–99 使用普通交叉熵并积累未扰动特征，step 100
开始增加 eta，step 299 达到 0.2。计数和历史缓存写入模型 `state_dict`。
若手动恢复训练，还需恢复优化器及随机状态；当前训练入口没有新增自动断点续训功能。

日志额外记录 `loss_unweighted`、`stable_mix`、`weight_min/max`、`weight_ess`、
`weight_ess_ratio`、`stable_raw_ess`、`weight_domain_*`、`weight_class_*` 和
`batch_class_*`。`weight_ess` 基于最终参与分类的权重，`stable_raw_ess` 基于域内
归一和混合之前的原版 StableNet 权重。`weight_domain_0/1` 是本次训练的源域顺序，
不是固定的 HTP 全局域编号。类别权重不被强行平衡，应与 `batch_class_*` 对照检查。

`stable_lambdap` 位于原目标的分母：增大它会减弱去相关项。
目前仍使用本仓库的复数低频谱扰动，不切换到官方幅度扰动版本。

#### 单 seed 与服务器烟雾测试

需要调整 eta 时使用 `train_all.py`，例如同配方的 eta=0.1、seed=0：

```sh
python train_all.py HTP_ALOFT_Stable_E_eta01_seed0 --dataset HTP --data_dir ./dataset --algorithm ALOFT_Stable_E --steps 5000 --batch_size 32 --pretrained True --freeze_bn True --swad LossValley --seed 0 --trial_seed 0 --deterministic --cache disk --prebuild_loader --image_size 224 --stable_mix_max 0.1
```

`--stable_mix_max 0` 关闭额外前向和权重优化，退化为相同配置的 ALOFT-E。
不要按目标域测试结果选择 eta；正式比较使用相同划分、seed 和选模规则。
SWAD 训练中的 `iid` 单 checkpoint 结果不能当成独立关闭 SWAD 的实验。

首次上服务器可先执行下面的短检查。它关闭预热以确保真正进入权重内循环，
不用于报告准确率，也不下载预训练权重：

```sh
CUDA_VISIBLE_DEVICES=0 python train_all.py HTP_ALOFT_Stable_smoke --dataset HTP --data_dir ./dataset --algorithm ALOFT_Stable_E --test_envs 0 --steps 2 --checkpoint_freq 1 --batch_size 32 --pretrained False --freeze_bn True --swad False --stable_warmup_steps 0 --stable_ramp_steps 0
```

离线本地测试：

```sh
python -m unittest discover -s tests -p "test_aloft_stable.py" -v
python -m unittest discover -s tests -p "test_*.py" -v
```

测试覆盖原版权重内循环、域权重归一、预热/渐进开启、模式与 BN 状态恢复、
缓存更新、关闭权重时与 baseline 的逐位更新/RNG 一致性、检查点状态恢复、SWAD 推理，
以及 ResNet18、ResNet50 和小型 ViT 的有效重加权更新。不下载预训练权重；无 CUDA 时
明确跳过 CUDA 测试。本地单元测试不代表完成 HTP 正式训练或已经验证准确率提升。

### Run all experiments

We provide the instructions to reproduce the main results of the paper, Table 1 and 2.
Note that the difference in a detailed environment or uncontrolled randomness may bring a slightly different result from the paper.

- PACS

```
python train_all.py PACS0 --dataset PACS --data_dir /my/datasets/path --deterministic --trial_seed 0 --algorithm ERM_GGA --checkpoint_freq 100 --lr 3e-5 --weight_decay 1e-4 --resnet_dropout 0.5 --swad False \
--start_step 100 --end_step 200 --neighborhoodSize 0.00001
```

- VLCS

```
python train_all.py VLCS0 --dataset VLCS --data_dir /my/datasets/path --deterministic --trial_seed 0 --algorithm ERM_GGA --checkpoint_freq 100 --lr 1e-5 --weight_decay 1e-4 --resnet_dropout 0.5 --swad False \
--start_step 100 --end_step 200 --neighborhoodSize 0.000001
```

- OfficeHome

```
python train_all.py OH0 --dataset OfficeHome --data_dir /my/datasets/path --deterministic --trial_seed 0 --algorithm ERM_GGA --checkpoint_freq 100 --lr 1e-5 --weight_decay 1e-4 --resnet_dropout 0.5 --swad False \
--start_step 100 --end_step 200 --neighborhoodSize 0.00001
```

- TerraIncognita

```
python train_all.py TR0 --dataset TerraIncognita --data_dir /my/datasets/path --deterministic --trial_seed 0 --algorithm ERM_GGA --checkpoint_freq 100 --lr 1e-5 --weight_decay 1e-4 --resnet_dropout 0.5 --swad False \
--start_step 100 --end_step 200 --neighborhoodSize 0.00001
```

- DomainNet

```
python train_all.py DN0 --dataset DomainNet --data_dir /my/datasets/path --deterministic --trial_seed 0 --algorithm ERM_GGA --checkpoint_freq 100 --lr 3e-5 --weight_decay 1e-6 --resnet_dropout 0.5 --swad False \
--start_step 100 --end_step 200 --neighborhoodSize 0.00001
```

## An Alternative Method for Gradient-Guided Annealing: GGA-L
This repo also contains code for an alternative method to GGA, 
which integrates noise directly into the gradient update step rather than modifying 
weights separately. This alternative method is described in the Supplemntary Material of
the [ArXiv](https://arxiv.org/abs/2502.20162) version of the manuscript.

Specifically in GGA-L, we propose injecting dynamic noise based on domain gradient similarity directly into the 
update step, as follows:


<p align="center">
    <img src="./assets/gga_l.png" width="50%" />
</p>

where ξ is noise drawn from a Uniform distribution and

<p align="center">
    <img src="./assets/gga_l2.png" width="50%" />
</p>

is a dynamic scaling factor depending on the average gradient similarity between domains and γ is a 
hyperparameter controlling the noise intensity. 

GGA-L is able to perform similarly to GGA but adds a considerably lower computational cost as the gradients for each domain are only calculated once per batch.
The algorithm for GGA-L is the following:

<p align="center">
    <img src="./assets/gga_l_algo.png" width="50%" />
</p>

Similar to above, you can run GGA-L as follows:

```sh
python train_all.py exp_name --dataset <dataset> --data_dir /my/datasets/path --trial_seed <seed> --algorithm GGA_L --checkpoint_freq 100 --lr <lr> --gga_l_gamma <gamma> --weight_decay 1e-4 --resnet_dropout 0.5 --swad False 
```

## Improvement over baseline results
GGA is able to boost the performance of a vanilla model on all 5 datasets.

<p align="center">
    <img src="./assets/baseline_results.png" width="70%" />
</p>

## Improving the performance of SoTA algorithms
When applied on top of previously proposed algorithms, GGA is able to boost their 
performance in most cases.
<p align="center">
    <img src="./assets/sota_results.png" width="80%" />
</p>

## Our searched HPs

<p align="center">
    <img src="./assets/hp.png" width="60%" />
</p>

---
**NOTICE**

*We have identified and corrected several issues in the 
original version of this paper, including errors in 
both the manuscript and the accompanying code. We kindly 
ask that any comparisons or future references be made 
using the results and findings presented in the updated 
[ArXiv](https://arxiv.org/abs/2502.20162) version.*

---
## Citation

Please cite this paper if it helps your research:

```
@inproceedings{ballas2025gradient,
  title={Gradient-Guided Annealing for Domain Generalization},
  author={Ballas, Aristotelis and Diou, Christos},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={20558--20568},
  year={2025}
}
```


## License

This source code is released under the MIT license, included [here](./LICENSE).

This project includes some code from [DomainBed](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414), also MIT licensed.

