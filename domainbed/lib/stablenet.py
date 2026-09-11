import math

import torch


def random_fourier_features(x, num_f=1, concat=True):
    # x: [N, D]
    # concat=True 时返回 [N, D, 2*num_f]
    d = x.shape[1]

    omega = torch.randn(num_f, 1).to(x)
    phase = (2 * math.pi * torch.rand(d, num_f)).to(x)

    z = x.unsqueeze(-1) @ omega.t() + phase

    # 保留原仓库的映射方式，增加除零保护。
    z = z - z.amin(dim=1, keepdim=True)
    z = z / z.amax(dim=1, keepdim=True).clamp_min(1e-12)
    z = z * (math.pi / 2)

    if concat:
        mapped = torch.cat([z.cos(), z.sin()], dim=-1)
    else:
        mapped = z.cos() + z.sin()

    return math.sqrt(2.0 / num_f) * mapped


def balance_loss(features, weights, num_f=1, concat=True):
    mapped = random_fourier_features(features, num_f, concat)
    loss = features.new_zeros(())

    for k in range(mapped.shape[-1]):
        z = mapped[:, :, k]

        mean = (weights * z).sum(dim=0, keepdim=True)
        covariance = (weights * z).t() @ z - mean.t() @ mean

        off_diag = covariance - torch.diag_embed(
            covariance.diagonal()
        )
        loss = loss + off_diag.square().sum()

    return loss


def learn_weights(features, pre_features, pre_logits, hp, epoch=0):
    # 内循环只优化 raw，不更新骨干网络。
    features = features.detach().float()

    all_features = torch.cat([
        features,
        pre_features.detach().to(features),
    ])

    raw = features.new_ones(
        (features.shape[0], 1),
        requires_grad=True,
    )

    optimizer = torch.optim.SGD(
        [raw],
        lr=hp["stable_lrbl"],
        momentum=0.9,
    )

    rounds = int(hp["stable_epochb"])

    if (
        rounds < 1
        or hp["stable_num_f"] < 1
        or hp["stable_lambdap"] <= 0
    ):
        raise ValueError("Invalid StableNet hyperparameters")

    lam = hp["stable_lambdap"] * max(
        hp["stable_lambda_decay_rate"]
        ** (epoch // hp["stable_lambda_decay_epoch"]),
        hp["stable_min_lambda_times"],
    )

    for k in range(rounds):
        optimizer.param_groups[0]["lr"] = (
            hp["stable_lrbl"]
            * 0.1 ** (k // (rounds * 0.5))
        )

        optimizer.zero_grad()

        all_logits = torch.cat([
            raw,
            pre_logits.detach().to(features),
        ])

        global_weights = all_logits.softmax(dim=0)

        decorrelation = balance_loss(
            all_features,
            global_weights,
            hp["stable_num_f"],
            hp["stable_concat"],
        )

        regularizer = (
            raw.softmax(dim=0)
            .pow(hp["stable_decay_pow"])
            .sum()
        )

        objective = decorrelation / lam + regularizer

        if epoch == 0:
            objective = objective * hp["stable_first_step_cons"]

        if not torch.isfinite(objective):
            raise FloatingPointError(
                "Non-finite StableNet weight objective"
            )

        objective.backward()
        optimizer.step()

    weights = raw.detach().softmax(dim=0)

    if not torch.isfinite(weights).all():
        raise FloatingPointError("Non-finite StableNet weights")

    return weights, raw.detach()