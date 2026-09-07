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

