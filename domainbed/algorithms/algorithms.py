# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
from typing import List

import itertools


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

from collections import OrderedDict

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

#  import higher

from domainbed import networks
from domainbed.lib.gradient_utils import METHODS, get_method, agreement_mask
from domainbed.lib.misc import random_pairs_of_minibatches, split_meta_train_test, ParamDict
from domainbed.optimizers import get_optimizer
from domainbed.models.aloft import (
    ALOFT as ALOFTModule,
    find_aloft_modules,
    resnet_aloft,
)
from domainbed.models.aloft_cb import resnet_aloft_cb, find_band_codebooks
from domainbed.models.aloft_sketch import (
    collect_sketch_diagnostics,
    collect_sketch_topology_loss,
    find_sketch_spectrum_modules,
    resnet_aloft_sketch,
)
from domainbed.models.aloft_structure import (
    SketchStructureTargets,
    collect_structure_diagnostics,
    collect_structure_losses,
    find_structure_probes,
    resnet_aloft_structure,
    structure_scale_at_step,
)
from domainbed.models.awwsl import AWWSL as AWWSLModule, resnet_awwsl
from domainbed.models.frequant import FreqQuant, resnet_freqquant, collect_aux_loss
from domainbed.models.csu import resnet_csu

from domainbed.models.resnet_mixstyle import (
    resnet18_mixstyle_L234_p0d5_a0d1,
    resnet50_mixstyle_L234_p0d5_a0d1,
)
from domainbed.models.resnet_mixstyle2 import (
    resnet18_mixstyle2_L234_p0d5_a0d1,
    resnet50_mixstyle2_L234_p0d5_a0d1,
)

from domainbed.sagm import SAGM, LinearScheduler, SAGM_CUSTOM
from domainbed import gsam

from domainbed.lib.stablenet import learn_weights
from domainbed.lib.aloft_stable import (
    clean_features,
    domain_balanced_weights,
    stable_mix_at_step,
)
from domainbed.lib import sft as sft_utils


def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches

def flat_grad(grad_tuple):
    return torch.cat([p.view(-1) for p in grad_tuple]).unsqueeze(0)

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError



    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.grad_fn = hparams["grad_fn"]
        self.logdir = hparams["logdir"]
        self.test_env = hparams["test_env"]

        self.cos = nn.CosineSimilarity(dim=0)

    def get_grads(self):
        grads = []
        for p in self.network.parameters():
            grads.append(p.grad.data.clone().flatten())
        return torch.cat(grads)

    def set_grads(self, new_grads):
        start = 0
        for k, p in enumerate(self.network.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class SFT(Algorithm):
    """Self-Feedback Training (arXiv:2412.13573v2, Algorithm 1).

    The student takes one soft-label SAM step on a randomly selected source
    domain. Then a separate, equally sized refiner learns from all-source PCE
    and post-update, domain-specific sharpness feedback. Only the student is
    used at inference. Neither optimizer owns the other network's parameters.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if num_domains < 2:
            raise ValueError("SFT requires at least two source domains")
        if num_classes < 2:
            raise ValueError("SFT requires at least two classes")
        for key, lower in (("sft_rho", 0), ("sft_alpha", 1),
                           ("sft_lambda1", 0), ("sft_lambda2", 0)):
            value = float(hparams[key])
            if not np.isfinite(value) or value < lower:
                raise ValueError(f"{key} must be finite and >= {lower}")
            setattr(self, key, value)

        self.featurizer = networks.Featurizer(input_shape, hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        refiner_features = networks.Featurizer(input_shape, hparams)
        self.refiner = nn.Sequential(
            refiner_features, nn.Linear(refiner_features.n_outputs, num_classes))
        self.optimizer = self.new_optimizer(self.network.parameters())
        self.refiner_optimizer = self.new_optimizer(self.refiner.parameters())

    def update(self, x, y, **kwargs):
        if len(x) != self.num_domains or len(y) != self.num_domains:
            raise ValueError("SFT expects one nonempty minibatch per source domain")
        for xi, yi in zip(x, y):
            if (len(xi) == 0 or yi.shape != (len(xi),) or yi.dtype != torch.long
                    or xi.device != yi.device or xi.device != x[0].device):
                raise ValueError("SFT expects matching nonempty inputs and int64 labels")
            if (yi < 0).any() or (yi >= self.num_classes).any():
                raise ValueError("SFT labels are outside the class range")

        d, other = torch.randperm(self.num_domains)[:2].tolist()
        self.optimizer.zero_grad(set_to_none=True)
        self.refiner_optimizer.zero_grad(set_to_none=True)
        # One refiner forward updates BN once and reuses the same soft labels
        # throughout both phases. Only the SAM targets are detached.
        refiner_logits = self.refiner(torch.cat(x))
        pce = sft_utils.projection_cross_entropy(
            refiner_logits, torch.cat(y), self.sft_alpha)
        targets = refiner_logits.softmax(dim=-1).split([len(xi) for xi in x])

        _, model_loss = sft_utils.paired_losses(
            self.network, x[d], targets[d].detach(), self.sft_rho,
            update_buffers=True)
        model_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Eq. (9) must be recomputed at the updated theta, separately per domain.
        # Detached parameter leaves keep refinement out of the live model and
        # avoid unrolling the preceding optimizer step. Epsilon remains a
        # differentiable function of phi, including its global normalization.
        feedback_enabled = self.sft_lambda1 != 0 or self.sft_lambda2 != 0
        feedback_targets = targets if feedback_enabled else [t.detach() for t in targets]
        sharp_d = sft_utils.sharpness(
            self.network, x[d], feedback_targets[d], self.sft_rho,
            create_graph=feedback_enabled)
        sharp_other = sft_utils.sharpness(
            self.network, x[other], feedback_targets[other], self.sft_rho,
            create_graph=feedback_enabled)
        gap = (sharp_d - sharp_other).abs()
        refiner_loss = pce + self.sft_lambda1 * sharp_d + self.sft_lambda2 * gap
        refiner_parameters = tuple(self.refiner.parameters())
        gradients = torch.autograd.grad(refiner_loss, refiner_parameters, allow_unused=True)
        for p, gradient in zip(refiner_parameters, gradients):
            p.grad = gradient
        self.refiner_optimizer.step()

        return {"loss": model_loss.item(), "sft_pce": pce.item(),
                "sft_refiner_loss": refiner_loss.item(),
                "sft_sharpness_train": sharp_d.item(),
                "sft_sharpness_other": sharp_other.item(),
                "sft_sharpness_gap": gap.item()}

    def predict(self, x):
        return self.network(x)

    def get_swad_model(self):
        return sft_utils.SFTInferenceModel(self.network, self.training)

    def clone(self):
        # Copy both optimizers together with their parameter owners and states.
        return copy.deepcopy(self)


class CSU(Algorithm):
    """Correlated Style Uncertainty with the standard ERM objective."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[0] == 3, "CSU supports three-channel images only"
        assert networks.is_vit(hparams) or not hparams["resnet18"], "This CSU integration uses ResNet-50"
        super().__init__(input_shape, num_classes, num_domains, hparams)

        if networks.is_vit(hparams):
            network = networks.Featurizer(input_shape, hparams)
        else:
            network = torchvision.models.resnet50(pretrained=hparams["pretrained"])
        network = resnet_csu(
            network,
            positions=tuple(hparams["csu_positions"]),
            p=hparams["csu_p"],
            alpha=hparams["csu_alpha"],
        )
        self.featurizer = networks.wrap_backbone(input_shape, self.hparams, network)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)

class PrototypePLoss(nn.Module):
    def __init__(self, num_classes, temperature):
        super(PrototypePLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.label = torch.arange(num_classes)
        self.temperature = temperature

    def forward(self, feature, prototypes, labels):
        feature = F.normalize(feature, p=2, dim=1)
        feature_prototype = torch.einsum('nc,mc->nm', feature, prototypes)

        feature_pairwise = torch.einsum('ic,jc->ij', feature, feature)
        mask_neg = torch.not_equal(labels, labels.T)
        l_neg = feature_pairwise * mask_neg
        l_neg = l_neg.masked_fill(l_neg < 1e-6, -np.inf)

        # [N, C+N]
        logits = torch.cat([feature_prototype, l_neg], dim=1)
        loss = F.nll_loss(F.log_softmax(logits / self.temperature, dim=1), labels)
        return loss


class MultiDomainPrototypePLoss(nn.Module):
    def __init__(self, num_classes, num_domains, temperature):
        super(MultiDomainPrototypePLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.num_classes = num_classes
        self.label = torch.arange(num_classes)
        self.domain_label = torch.arange(num_domains)
        self.temperature = temperature

    def forward(self, feature, prototypes, labels, domain_labels):
        feature = F.normalize(feature, p=2, dim=1)
        feature_prototype = torch.einsum('nc,mc->nm', feature, prototypes.reshape(-1, prototypes.size(-1)))

        feature_pairwise = torch.einsum('ic,jc->ij', feature, feature)
        mask_neg = torch.logical_or(torch.not_equal(labels, labels.T), torch.not_equal(domain_labels, domain_labels))
        l_neg = feature_pairwise * mask_neg
        l_neg = l_neg.masked_fill(l_neg < 1e-6, -np.inf)

        # [N, C*D + N]
        logits = torch.cat([feature_prototype, l_neg], dim=1)
        loss = F.nll_loss(F.log_softmax(logits / self.temperature, dim=1), domain_labels * self.num_classes + labels)
        return loss


class iDAG(Algorithm):
    """
    DAG domain generalization methods
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(iDAG, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.encoder = networks.LightEncoder(self.featurizer.n_outputs,
                                             hparams['out_dim'],
                                             hparams['hidden_size'],
                                             hparams["num_hidden_layers"])

        self.dag_mlp = networks.NotearsClassifier(hparams['out_dim'], num_classes)
        self.dag_mlp.weight_pos.data[:-1, -1].fill_(1.0)

        self.inv_classifier = nn.Linear(hparams['out_dim'], num_classes)
        self.rec_classifier = nn.Linear(hparams['out_dim'], num_classes)
        self.network = nn.Sequential(self.featurizer, self.encoder, self.dag_mlp, self.inv_classifier)

        self.proto_m = self.hparams["ema_ratio"]
        self.lambda1 = self.hparams["lambda1"]
        self.lambda2 = self.hparams["lambda2"]
        self.rho_max = self.hparams["rho_max"]
        self.alpha = self.hparams["alpha"]
        self.rho = self.hparams["rho"]
        self._h_val = np.inf

        self.register_buffer(
            "prototypes_y",
            torch.zeros(num_classes, hparams['out_dim']))
        self.register_buffer(
            "prototypes",
            torch.zeros(num_domains, num_classes, hparams['out_dim']))
        self.register_buffer(
            "prototypes_label",
            torch.arange(num_classes).repeat(num_domains))

        params = [
            {"params": self.network.parameters()},
            {"params": self.rec_classifier.parameters()},
        ]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.loss_proto_con = PrototypePLoss(num_classes, hparams['temperature'])
        self.loss_multi_proto_con = MultiDomainPrototypePLoss(num_classes, num_domains, hparams['temperature'])

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        domain_labels = torch.cat([torch.ones(len(_y)) * i for i, _y in enumerate(y)]).long().to(all_x.device)

        all_f = self.featurizer(all_x)
        all_f = self.encoder(all_f)
        all_masked_f = self.dag_mlp(all_f)

        for f, masked_f, label_y, label_d in zip(F.normalize(all_f, dim=1),
                                                 F.normalize(all_masked_f, dim=1),
                                                 all_y,
                                                 domain_labels):
            self.prototypes[label_d, label_y] = self.prototypes[label_d, label_y] * self.proto_m + (1 - self.proto_m) * f.detach()
            self.prototypes_y[label_y] = self.prototypes_y[label_y] * self.proto_m + (1 - self.proto_m) * masked_f.detach()
        self.prototypes = F.normalize(self.prototypes, p=2, dim=2)
        self.prototypes_y = F.normalize(self.prototypes_y, p=2, dim=1)

        prototypes = self.prototypes.detach().clone()
        prototypes_y = self.prototypes_y.detach().clone()

        proto_rec, masked_proto = self.dag_mlp(
            x=prototypes.view(self.num_domains * self.num_classes, -1),
            y=self.prototypes_label)

        # reconstruction loss
        loss_rec = F.cosine_embedding_loss(
            proto_rec,
            prototypes.view(self.num_domains * self.num_classes, -1),
            torch.ones(self.num_domains * self.num_classes, device=all_x.device))
        loss_rec += F.cross_entropy(
            self.rec_classifier(masked_proto),
            self.prototypes_label)
        loss_rec = self.lambda2 * loss_rec
        h_val = self.dag_mlp.h_func()
        penalty = 0.5 * self.rho * h_val * h_val + self.alpha * h_val
        l1_reg = self.lambda1 * self.dag_mlp.w_l1_reg()

        # update the DAG hyper-parameters
        if kwargs['step'] % 100 == 0:
            if self.rho < self.rho_max and h_val > 0.25 * self._h_val:
                self.rho *= 10
                self.alpha += self.rho * h_val.item()
            self._h_val = h_val.item()

        loss_dag = loss_rec + penalty + l1_reg

        loss_inv_ce = F.cross_entropy(self.inv_classifier(all_masked_f), all_y)

        loss_contr_mu = self.hparams["weight_mu"] * self.loss_proto_con(all_masked_f, prototypes_y, all_y)
        loss_contr_nu = self.hparams["weight_nu"] * self.loss_multi_proto_con(all_f, prototypes, all_y, domain_labels)
        loss_contr = loss_contr_mu + loss_contr_nu

        if kwargs['step'] == self.hparams["dag_anneal_steps"]:
            # avoid the gradient jump
            params = [
                {"params": self.network.parameters()},
            ]
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                params,
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        if kwargs['step'] >= self.hparams["dag_anneal_steps"]:
            loss = loss_inv_ce + loss_dag + loss_contr_mu + loss_contr_nu
        else:
            loss = loss_inv_ce + loss_contr_nu

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # constraint DAG weights
        self.dag_mlp.projection()

        return {"loss": loss.item(),
                "inv_ce": loss_inv_ce.item(),
                "l2": loss_rec.item(),
                "penalty": penalty.item(),
                "l1": l1_reg.item(),
                "cl": loss_contr.item()}

    def predict(self, x):
        f = self.featurizer(x)
        f = self.encoder(f)
        masked_f = self.dag_mlp(f)
        return self.inv_classifier(masked_f)

    def clone(self):
        clone = copy.deepcopy(self)
        params = [
            {"params": clone.network.parameters()},
        ]
        clone.optimizer = self.new_optimizer(params)
        clone.optimizer.load_state_dict(self.optimizer.state_dict())
        return clone

class GSAM(Algorithm):
    """
    Surrogate Gap Guided Sharpness-Aware Minimization
    """

    # def __init__(self, input_shape, num_classes, num_domains, hparams):
    #     assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
    #     super().__init__(input_shape, num_classes, num_domains, hparams)
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.lr_scheduler = gsam.LinearScheduler(T_max=5000, max_value=self.hparams["lr"],
                                            min_value=self.hparams["lr"], optimizer=self.optimizer)

        self.rho_scheduler = gsam.LinearScheduler(T_max=5000, max_value=0.05,
                                             min_value=0.05)

        self.GSAM_optimizer = gsam.GSAM(params=self.network.parameters(), base_optimizer=self.optimizer, model=self.network,
                                        gsam_alpha=self.hparams["alpha"], rho_scheduler=self.rho_scheduler, adaptive=False)

        self.grad_fn = hparams["grad_fn"]
        self.logdir = hparams["logdir"]
        self.test_env = hparams["test_env"]

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        def loss_fn(predictions, targets):
            return F.cross_entropy(predictions, targets)

        self.GSAM_optimizer.set_closure(loss_fn, all_x, all_y)
        predictions, loss = self.GSAM_optimizer.step()
        self.lr_scheduler.step()
        self.GSAM_optimizer.update_rho_t()


        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class SAGM_DG(Algorithm):
    """
    Sharpness-Aware Gradient Matching for Domain Generalization
    """

    # def __init__(self, input_shape, num_classes, num_domains, hparams):
    #     assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
    #     super().__init__(input_shape, num_classes, num_domains, hparams)
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.lr_scheduler = LinearScheduler(T_max=5000, max_value=self.hparams["lr"],
                                            min_value=self.hparams["lr"], optimizer=self.optimizer)

        self.rho_scheduler = LinearScheduler(T_max=5000, max_value=0.05,
                                             min_value=0.05)

        self.SAGM_optimizer = SAGM(params=self.network.parameters(), base_optimizer=self.optimizer, model=self.network,
                                   alpha=self.hparams["alpha"], rho_scheduler=self.rho_scheduler, adaptive=False)

        self.grad_fn = hparams["grad_fn"]
        self.logdir = hparams["logdir"]
        self.test_env = hparams["test_env"]

        self.cos = nn.CosineSimilarity(dim=0)

        self.neighborhoodSize = hparams["neighborhoodSize"]
        self.out_dir = hparams["logdir"]
        self.patience_limit = hparams["annealing_patience"]
        self.patience_step = 0
        self.best_loss = np.inf

        # save start state
        torch.save(self.network.state_dict(), self.out_dir / "network_start.pt")
        # save best state
        torch.save(self.network.state_dict(), self.out_dir / "network_best.pt")

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        def loss_fn(predictions, targets):
            return F.cross_entropy(predictions, targets)

        self.SAGM_optimizer.set_closure(loss_fn, all_x, all_y)
        predictions, loss = self.SAGM_optimizer.step()
        self.lr_scheduler.step()
        self.SAGM_optimizer.update_rho_t()


        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class Mixstyle(Algorithm):
    """MixStyle w/o domain label (random shuffle)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[0] == 3, "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if networks.is_vit(hparams):
            from domainbed.models.mixstyle import MixStyle
            network = networks.Featurizer(input_shape, hparams)
            operation = MixStyle(p=1.0, alpha=0.1)
            for position in ("layer1", "layer2", "layer3"):
                network.add_stage_op(position, operation, "mixstyle")
        elif hparams["resnet18"]:
            network = resnet18_mixstyle_L234_p0d5_a0d1()
        else:
            # network = resnet50_mixstyle_L234_p0d5_a0d1(postion=["conv2_x", "conv3_x", "conv4_x"])
            network = resnet50_mixstyle_L234_p0d5_a0d1(postion=[])

        self.featurizer = networks.wrap_backbone(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class Mixstyle2(Algorithm):
    """MixStyle w/ domain label"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[0] == 3, "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if networks.is_vit(hparams):
            from domainbed.models.mixstyle import MixStyle2
            network = networks.Featurizer(input_shape, hparams)
            operation = MixStyle2(p=0.5, alpha=0.1)
            for position in ("layer1", "layer2", "layer3"):
                network.add_stage_op(position, operation, "mixstyle")
        elif hparams["resnet18"]:
            network = resnet18_mixstyle2_L234_p0d5_a0d1()
        else:
            network = resnet50_mixstyle2_L234_p0d5_a0d1()
        self.featurizer = networks.wrap_backbone(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def pair_batches(self, xs, ys):
        xs = [x.chunk(2) for x in xs]
        ys = [y.chunk(2) for y in ys]
        N = len(xs)
        pairs = []
        for i in range(N):
            j = i + 1 if i < (N - 1) else 0
            xi, yi = xs[i][0], ys[i][0]
            xj, yj = xs[j][1], ys[j][1]

            pairs.append(((xi, yi), (xj, yj)))

        return pairs

    def update(self, x, y, **kwargs):
        pairs = self.pair_batches(x, y)
        loss = 0.0

        for (xi, yi), (xj, yj) in pairs:
            #  Mixstyle2:
            #  For the input x, the first half comes from one domain,
            #  while the second half comes from the other domain.
            x2 = torch.cat([xi, xj])
            y2 = torch.cat([yi, yj])
            loss += F.cross_entropy(self.predict(x2), y2)

        loss /= len(pairs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class ARM(ERM):
    """Adaptive Risk Minimization (ARM)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams["batch_size"]

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class SAM(ERM):
    """Sharpness-Aware Minimization
    """
    @staticmethod
    def norm(tensor_list: List[torch.tensor], p=2):
        """Compute p-norm for tensor list"""
        return torch.cat([x.flatten() for x in tensor_list]).norm(p)

    def update_anneal(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        grads_i_v = []
        i = 0
        for x_i, y_i in zip(x, y):
            grads_i = []
            # 1. Compute grad of domain batch
            loss_i = F.cross_entropy(self.predict(x_i), y_i)

            # with open(f"{self.logdir}/loss_{self.test_env}_{i}.txt", "a") as f:
            #     f.write(f"{str(loss_i.item())}\n")
            # f.close()

            # 2. Flat and add domain grads to list
            grad_i = autograd.grad(loss_i, self.network.parameters())
            for g in grad_i:
                grads_i.append(g.flatten())
            grads_i_v.append(torch.cat(grads_i))
            i += 1

        c = list(itertools.combinations(list(range(len(grads_i_v))), 2))

        sims = []
        for i, j in c:
            sims.append(self.cos(grads_i_v[i], grads_i_v[j]))

        if any(s < 0.5 for s in sims):
            # self.anneal()
            # print(f"Loss before: {F.cross_entropy(self.predict(all_x), all_y).item()}")
            start = 10
            finish = 160
            i = 0
            with torch.no_grad():
                for param in self.network.parameters():
                    if start <= i <= finish:
                        # param.add_(torch.randn(param.size()).cuda() * 0.1)

                        param.add_(torch.cuda.FloatTensor.normal_(0, 1))

                        param.add_(torch.Tensor(np.random.uniform(low=self.neighborhoodSize * -1, high=self.neighborhoodSize,
                                                                  size=param.shape)))
                i += 1
            # params = list(self.network.parameters())

            # eps = [torch.Tensor(np.random.uniform(low=self.neighborhoodSize * -1, high=self.neighborhoodSize,
            #                                       size=p.shape)) for p in params]
            #
            # start = 50
            # finish = 155
            # i = 0
            # with torch.no_grad():
            #     for p, v in zip(self.network.parameters(), eps):
            #         if start <= i <= finish:
            #             p.add_(v)
            #     i += 1

        loss = F.cross_entropy(self.predict(all_x), all_y)
        # print(f"Loss after: {loss.item()}")
        self.optimizer.zero_grad()
        loss.backward()

        # grad_all = self.get_grads()

        # for j in range(len(grads_i_v)):
        #     sim_j = self.cos(grad_all, grads_i_v[j])
        #     with open(f"{self.logdir}/cosine_{self.test_env}_all_{j}.txt", "a") as f:
        #         f.write(f"{str(sim_j.item())}\n")
        #     f.close()

        self.optimizer.step()


        return {"loss": loss.item()}


    def update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])

        loss = F.cross_entropy(self.predict(all_x), all_y)

        # 1. eps(w) = rho * g(w) / g(w).norm(2)
        #           = (rho / g(w).norm(2)) * g(w)
        grad_w = autograd.grad(loss, self.network.parameters())
        scale = self.hparams["rho"] / self.norm(grad_w)
        eps = [g * scale for g in grad_w]

        # 2. w' = w + eps(w)
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.add_(v)

        # 3. w = w - lr * g(w')
        loss = F.cross_entropy(self.predict(all_x), all_y)
        with open("sam_0.txt", "a") as f:
            f.write(str(loss.item()))
            f.write("\n")

        self.optimizer.zero_grad()
        loss.backward()
        # restore original network params
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.sub_(v)
        self.optimizer.step()

        return {"loss": loss.item()}


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.register_buffer("update_count", torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.discriminator = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = get_optimizer(
            hparams["optimizer"],
            (list(self.discriminator.parameters()) + list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams["weight_decay_d"],
            betas=(self.hparams["beta1"], 0.9),
        )

        self.gen_opt = get_optimizer(
            hparams["optimizer"],
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams["weight_decay_g"],
            betas=(self.hparams["beta1"], 0.9),
        )

    def update(self, x, y, **kwargs):
        self.update_count += 1
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])
        minibatches = to_minibatch(x, y)
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat(
            [
                torch.full((x.shape[0],), i, dtype=torch.int64, device="cuda")
                for i, (x, y) in enumerate(minibatches)
            ]
        )

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1.0 / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction="none")
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(
            disc_softmax[:, disc_labels].sum(), [disc_input], create_graph=True
        )[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams["grad_penalty"] * grad_penalty

        d_steps_per_g = self.hparams["d_steps_per_g_step"]
        if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {"disc_loss": disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = classifier_loss + (self.hparams["lambda"] * -disc_loss)
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {"gen_loss": gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=False,
            class_balance=False,
        )


class CDANN(AbstractDANN):
    """Conditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=True,
            class_balance=True,
        )


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.0).cuda().requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        penalty_weight = (
            self.hparams["irm_lambda"]
            if self.update_count >= self.hparams["irm_penalty_anneal_iters"]
            else 1.0
        )
        nll = 0.0
        penalty = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams["irm_penalty_anneal_iters"]:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams["vrex_penalty_anneal_iters"]:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class OrgMixup(ERM):
    """
    Original Mixup independent with domains
    """

    def update(self, x, y, **kwargs):
        x = torch.cat(x)
        y = torch.cat(y)

        indices = torch.randperm(x.size(0))
        x2 = x[indices]
        y2 = y[indices]

        lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

        x = lam * x + (1 - lam) * x2
        predictions = self.predict(x)

        objective = lam * F.cross_entropy(predictions, y)
        objective += (1 - lam) * F.cross_entropy(predictions, y2)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class CutMix(ERM):
    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def update(self, x, y, **kwargs):
        # cutmix_prob is set to 1.0 for ImageNet and 0.5 for CIFAR100 in the original paper.
        x = torch.cat(x)
        y = torch.cat(y)

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(x.size()[0]).cuda()
            target_a = y
            target_b = y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
            # compute output
            output = self.predict(x)
            objective = F.cross_entropy(output, target_a) * lam + F.cross_entropy(
                output, target_b
            ) * (1.0 - lam)
        else:
            output = self.predict(x)
            objective = F.cross_entropy(output, y)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q) / len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, x, y, **kwargs):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        minibatches = to_minibatch(x, y)
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        # for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
        for (xi, yi), (xj, yj) in split_meta_train_test(minibatches):

            inner_net = copy.deepcopy(self.network)

            inner_opt = get_optimizer(
                self.hparams["optimizer"],
                #  "SGD",
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # 1. Compute supervised loss for meta-train set
            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # 2. Compute meta loss for meta-val set
            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams["mldg_beta"] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(self.hparams["mldg_beta"] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {"loss": objective}


#  class SOMLDG(MLDG):
#      """Second-order MLDG"""
#      # This commented "update" method back-propagates through the gradients of
#      # the inner update, as suggested in the original MAML paper.  However, this
#      # is twice as expensive as the uncommented "update" method, which does not
#      # compute second-order derivatives, implementing the First-Order MAML
#      # method (FOMAML) described in the original MAML paper.

#      def update(self, x, y, **kwargs):
#          minibatches = to_minibatch(x, y)
#          objective = 0
#          beta = self.hparams["mldg_beta"]
#          inner_iterations = self.hparams.get("inner_iterations", 1)

#          self.optimizer.zero_grad()

#          with higher.innerloop_ctx(
#              self.network, self.optimizer, copy_initial_weights=False
#          ) as (inner_network, inner_optimizer):
#              for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
#                  for inner_iteration in range(inner_iterations):
#                      li = F.cross_entropy(inner_network(xi), yi)
#                      inner_optimizer.step(li)

#                  objective += F.cross_entropy(self.network(xi), yi)
#                  objective += beta * F.cross_entropy(inner_network(xj), yj)

#              objective /= len(minibatches)
#              objective.backward()

#          self.optimizer.step()

#          return {"loss": objective.item()}


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(
            x1_norm
        )
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs * 2, num_classes)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.register_buffer("embeddings", torch.zeros(num_domains, self.featurizer.n_outputs))

        self.ema = self.hparams["mtl_ema"]

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding + (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))


class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains, hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = nn.Linear(self.network_f.n_outputs, num_classes)
        # style network
        self.network_s = nn.Linear(self.network_f.n_outputs, num_classes)

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return get_optimizer(
                hparams["optimizer"], p, lr=hparams["lr"], weight_decay=hparams["weight_decay"]
            )

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).cuda()

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {
            "loss_c": loss_c.item(),
            "loss_s": loss_s.item(),
            "loss_adv": loss_adv.item(),
        }

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.drop_f = (1 - hparams["rsc_f_drop_factor"]) * 100
        self.drop_b = (1 - hparams["rsc_b_drop_factor"]) * 100
        self.num_classes = num_classes

    def update(self, x, y, **kwargs):
        # inputs
        all_x = torch.cat([xi for xi in x])
        # labels
        all_y = torch.cat([yi for yi in y])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.cuda()).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                            weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, x, y, **kwargs):
        self.create_clone(x[0].device)

        for xi, yi in zip(x, y):
            loss = F.cross_entropy(self.network_inner(xi), yi)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


##### Imported from DomainBed
class SelfReg(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SelfReg, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.num_classes = num_classes
        self.MSEloss = nn.MSELoss()
        # self.featurizer = networks.Featurizer(input_shape, self.hparams)
        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        input_feat_size = self.featurizer.n_outputs
        hidden_size = input_feat_size if input_feat_size==2048 else input_feat_size*2

        self.cdpl = nn.Sequential(
                            nn.Linear(input_feat_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, input_feat_size),
                            nn.BatchNorm1d(input_feat_size)
        )

    def update(self, x, y, **kwargs):

        all_x = torch.cat(x)
        all_y = torch.cat(y)

        lam = np.random.beta(0.5, 0.5)

        batch_size = all_y.size()[0]

        # cluster and order features into same-class group
        with torch.no_grad():
            sorted_y, indices = torch.sort(all_y)
            sorted_x = torch.zeros_like(all_x)
            for idx, order in enumerate(indices):
                sorted_x[idx] = all_x[order]
            intervals = []
            ex = 0
            for idx, val in enumerate(sorted_y):
                if ex==val:
                    continue
                intervals.append(idx)
                ex = val
            intervals.append(batch_size)

            all_x = sorted_x
            all_y = sorted_y

        feat = self.featurizer(all_x)
        proj = self.cdpl(feat)

        output = self.classifier(feat)

        # shuffle
        output_2 = torch.zeros_like(output)
        feat_2 = torch.zeros_like(proj)
        output_3 = torch.zeros_like(output)
        feat_3 = torch.zeros_like(proj)
        ex = 0
        for end in intervals:
            shuffle_indices = torch.randperm(end-ex)+ex
            shuffle_indices2 = torch.randperm(end-ex)+ex
            for idx in range(end-ex):
                output_2[idx+ex] = output[shuffle_indices[idx]]
                feat_2[idx+ex] = proj[shuffle_indices[idx]]
                output_3[idx+ex] = output[shuffle_indices2[idx]]
                feat_3[idx+ex] = proj[shuffle_indices2[idx]]
            ex = end

        # mixup
        output_3 = lam*output_2 + (1-lam)*output_3
        feat_3 = lam*feat_2 + (1-lam)*feat_3

        # regularization
        L_ind_logit = self.MSEloss(output, output_2)
        L_hdl_logit = self.MSEloss(output, output_3)
        L_ind_feat = 0.3 * self.MSEloss(feat, feat_2)
        L_hdl_feat = 0.3 * self.MSEloss(feat, feat_3)

        cl_loss = F.cross_entropy(output, all_y)
        C_scale = min(cl_loss.item(), 1.)
        loss = cl_loss + C_scale*(lam*(L_ind_logit + L_ind_feat)+(1-lam)*(L_hdl_logit + L_hdl_feat))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]

    def update(self, x, y, **kwargs):
        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        # minibatches = (x, y)
        for xi, yi in zip(x, y):
            logits = self.network(xi)

            env_loss = F.cross_entropy(logits, yi)
            mean_loss += env_loss.item() / len(x)

            env_grads = autograd.grad(env_loss, self.network.parameters())
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        self.optimizer.step()

        return {'loss': mean_loss}

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

        return 0



class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]
        self.cos = nn.CosineSimilarity(dim=0)

    def update(self, x, y, **kwargs):
        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        # minibatches = (x, y)
        for xi, yi in zip(x, y):
            logits = self.network(xi)

            env_loss = F.cross_entropy(logits, yi)
            mean_loss += env_loss.item() / len(x)

            env_grads = autograd.grad(env_loss, self.network.parameters())
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        self.optimizer.step()

        return {'loss': mean_loss}

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

        return 0



######################################################

class ERM_GGA(Algorithm):
    """
    Empirical Risk Minimization (ERM) with Gradient-Guided Annealing
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_GGA, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.grad_fn = hparams["grad_fn"]
        self.logdir = hparams["logdir"]
        self.test_env = hparams["test_env"]

        self.cos = nn.CosineSimilarity(dim=0)

        self.neighborhoodSize = hparams["neighborhoodSize"]

        self.T = 1.0
        self.T_min = 0.01
        self.P_T = 1.0
        self.P_T_min = 0.01
        self.best_params = None
        self.best_loss = float('inf')


    def perturb_weights(self, scale=0.0001):
        # Randomly perturb model parameters
        with torch.no_grad():
            for param in self.featurizer.parameters():
                if param.requires_grad:
                    noise = (torch.rand_like(param) * 2 - 1) * self.neighborhoodSize * self.P_T
                    param.data += noise

    def save_best_weights(self):
        # Save the current best parameters
        self.best_params = [param.data.clone() for param in self.network.parameters()]

    def restore_best_weights(self):
        # Restore the best parameters
        with torch.no_grad():
            for param, best_param in zip(self.network.parameters(), self.best_params):
                param.data.copy_(best_param)

    def calculate_domain_gradients(self, x, y):
        """
        Compute gradients for each domain batch.
        """
        grads_i_v = []
        for x_i, y_i in zip(x, y):
            loss_i = F.cross_entropy(self.predict(x_i), y_i)
            grad_i = autograd.grad(loss_i, self.network.parameters(), retain_graph=True)
            grads_i_v.append(torch.cat([g.flatten() for g in grad_i]))
        return grads_i_v

    def calculate_minimum_similarity(self, grads_i_v):
        """
        Compute the average cosine similarity between domain gradients.
        """
        pairwise_combinations = list(itertools.combinations(range(len(grads_i_v)), 2))

        all_sims = [self.cos(grads_i_v[i], grads_i_v[j]) for i, j in pairwise_combinations]

        return min(all_sims)

    def get_grads(self):
        grads = []
        for p in self.network.parameters():
            grads.append(p.grad.data.clone().flatten())
        return torch.cat(grads)

    def set_grads(self, new_grads):
        start = 0
        for k, p in enumerate(self.network.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end

    def update_simulated_annealing(self, x, y, **kwargs):
        """
        Search for better weights by adding noise and accepting them based on the improvement of
        the minimum domain pairwise gradient cosine similarity. The criterion can change.
        For i in range(iterations):
            # Step 1 --> Randomly add noise to weights to model parameters
            # Step 2 --> Calculate domain grads
            # Step 3 --> Calculate minimum pairwise cos similarity between domain grads and save
            # Step 4 --> Save model weights (state_dict()) if improvement
        # Step 5 --> load weights of optimal parameters
        # Step 6 --> update step
        """

        all_x = torch.cat(x)
        all_y = torch.cat(y)

        grads_i_v = self.calculate_domain_gradients(x, y)

        best_min_sim = self.calculate_minimum_similarity(grads_i_v)
        current_loss = F.cross_entropy(self.predict(all_x), all_y).item()

        # Save current state as best state before search
        current_params = [param.data.clone() for param in self.network.parameters()]
        self.save_best_weights()

        ##################
        search_steps = 250
        best_search_loss = current_loss

        for search_step in range(search_steps):
            # Perturb weights
            self.perturb_weights()

            # Calculate domain grad sim in search
            grads_i_v = self.calculate_domain_gradients(x, y)

            step_min_sim = self.calculate_minimum_similarity(grads_i_v)

            step_loss = F.cross_entropy(self.predict(all_x), all_y)
            step_loss = step_loss.item()

            # Calculate change in loss
            delta_loss = step_loss - best_search_loss
            delta_sim = best_min_sim - step_min_sim

            # Accept or reject new weights
            if delta_loss < 0.1 and delta_sim < 0:
                # Accept perturbed weights
                best_min_sim = step_min_sim
                best_search_loss = step_loss
                self.save_best_weights()
                with torch.no_grad():
                    for param, original_param in zip(self.network.parameters(), current_params):
                        param.data.copy_(original_param)

        # Restore best weights (they may be original the params if search failed)
        self.restore_best_weights()

        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return {"loss": loss.item()}

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class GGA_L(Algorithm):
    """
    Empirical Risk Minimization (ERM) with GGA-L update.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GGA_L, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.grad_fn = hparams["grad_fn"]
        self.logdir = hparams["logdir"]
        self.test_env = hparams["test_env"]

        self.cos = nn.CosineSimilarity(dim=0)

        self.neighborhoodSize = hparams["neighborhoodSize"]

        self.T = 1.0
        self.T_min = 0.01
        self.cooling_rate = 0.99
        self.P_T = 1.0
        self.P_T_min = 0.01
        self.perturb_cooling_rate = 0.99
        self.best_params = None
        self.best_loss = float('inf')

        self.gga_l_gamma = hparams["gga_l_gamma"]

    def save_best_weights(self):
        # Save the current best parameters
        self.best_params = [param.data.clone() for param in self.network.parameters()]

    def restore_best_weights(self):
        # Restore the best parameters
        with torch.no_grad():
            for param, best_param in zip(self.network.parameters(), self.best_params):
                param.data.copy_(best_param)

    def calculate_domain_gradients(self, x, y):
        """
        Compute gradients for each domain batch.
        """
        grads_i_v = []
        for x_i, y_i in zip(x, y):
            loss_i = F.cross_entropy(self.predict(x_i), y_i)
            grad_i = autograd.grad(loss_i, self.network.parameters(), retain_graph=True)
            grads_i_v.append(torch.cat([g.flatten() for g in grad_i]))
        return grads_i_v

    def calculate_similarity(self, grads_i_v):
        """
        Compute the average cosine similarity between domain gradients.
        """
        pairwise_combinations = list(itertools.combinations(range(len(grads_i_v)), 2))

        all_sims = [self.cos(grads_i_v[i], grads_i_v[j]) for i, j in pairwise_combinations]
        avg_sim = sum(all_sims) / len(pairwise_combinations)

        return avg_sim

    def get_grads(self):
        grads = []
        for p in self.network.parameters():
            grads.append(p.grad.data.clone().flatten())
        return torch.cat(grads)

    def set_grads(self, new_grads):
        start = 0
        for k, p in enumerate(self.network.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end

    def update_perturbed(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # Compute domain gradients
        grads_i_v = self.calculate_domain_gradients(x, y)

        # Compute min gradient similarity across domains
        min_sim = self.calculate_similarity(grads_i_v)

        # Compute loss and gradients
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()

        # Compute perturbation scale dynamically based on gradient similarity
        alpha = self.gga_l_gamma * (1 - min_sim)   # Adjust noise based on gradient alignment
        # Apply noise directly to gradients
        with torch.no_grad():
            for param in self.network.parameters():
                if param.grad is not None:
                    noise = torch.rand_like(param.grad) * alpha  # Scale noise adaptively
                    param.grad.add_(noise)

        # Perform optimizer step with modified gradients
        self.optimizer.step()

        return {"loss": loss.item(),}

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)

class Arith(ERM):
    """
    Arithmetic Meta-Learning for Domain Generalization.

    Inner loop:
        sequential SGD without momentum on randomly ordered domains

    Outer loop:
        Adam with arithmetically decreasing domain-gradient weights
    """

    @staticmethod
    def _make_step_weights(num_steps):
        if num_steps < 1:
            raise ValueError("Arith requires at least one source domain")

        denominator = num_steps * (num_steps + 1) / 2
        return [
            (num_steps - step_index) / denominator
            for step_index in range(num_steps)
        ]

    def _forward_with_parameters(self, parameters, x):
        # functional_call can replace both parameters and buffers.
        # Buffer references are included so BatchNorm continues to work.
        parameters_and_buffers = OrderedDict(
            self.network.named_buffers()
        )
        parameters_and_buffers.update(parameters)

        return functional_call(
            self.network,
            parameters_and_buffers,
            (x,)
        )

    def update(self, x,y, **kwargs):
        minibatches = to_minibatch(x,y)
        num_steps = len(minibatches)
        if num_steps == 0:
            raise ValueError("Arith received no source-domain minibatches")

        # Random order prevents one fixed domain from always receiving
        # the largest arithmetic weight.
        permutation = torch.randperm(num_steps).tolist()
        ordered_minibatches = [
            minibatches[index] for index in permutation
        ]

        step_weights = self._make_step_weights(num_steps)
        meta_lr = self.hparams["arith_meta_lr"]

        if meta_lr <= 0:
            raise ValueError("arith_meta_lr must be positive")

        base_parameters = OrderedDict(
            self.network.named_parameters()
        )
        fast_parameters = OrderedDict(base_parameters)

        meta_gradients = OrderedDict(
            (name, torch.zeros_like(parameter))
            for name, parameter in base_parameters.items()
        )

        losses = []

        for (x, y), step_weight in zip(
                ordered_minibatches, step_weights):
            logits = self._forward_with_parameters(
                fast_parameters, x
            )
            loss = F.cross_entropy(logits, y)

            gradients = torch.autograd.grad(
                loss,
                tuple(fast_parameters.values()),
                create_graph=False,
                allow_unused=True
            )

            next_fast_parameters = OrderedDict()

            for (name, parameter), gradient in zip(
                    fast_parameters.items(), gradients):
                if gradient is None:
                    next_fast_parameters[name] = parameter
                    continue

                gradient = gradient.detach()

                # Inner-loop SGD without momentum.
                next_fast_parameters[name] = (
                    parameter - meta_lr * gradient
                )

                # Matches MEDIC-plus accumulate_meta_grads_arith().
                # 1 / meta_lr is common to all domains and therefore
                # does not alter the arithmetic direction.
                meta_gradients[name].add_(
                    gradient,
                    alpha=step_weight / meta_lr
                )

            fast_parameters = next_fast_parameters
            losses.append(loss.detach())

        # Outer-loop Adam updates only the original parameters.
        self.optimizer.zero_grad()

        for name, parameter in base_parameters.items():
            parameter.grad = meta_gradients[name]

        self.optimizer.step()

        return {
            "loss": torch.stack(losses).mean().item()
        }

class ALOFT_DG(Algorithm):
    """ALOFT (CVPR'23) 的 ResNet 版：把频域扰动插进 ResNet 的 stage 之间。

    训练目标就是普通交叉熵，ALOFT 不引入任何损失项、也不引入可学习参数。
    """

    MODE = "E"          # 由子类覆盖
    REV = False         # 由子类覆盖
    NOISE_MODE = "iid"  # 由定向扰动子类覆盖

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[0] == 3, "ALOFT supports R18/R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)

        if networks.is_vit(hparams):
            network = networks.Featurizer(input_shape, hparams)
        elif hparams["resnet18"]:
            network = torchvision.models.resnet18(pretrained=hparams["pretrained"])
        else:
            network = torchvision.models.resnet50(pretrained=hparams["pretrained"])

        # 必须在预训练权重加载完之后再包 ALOFT，否则 state_dict 键名对不上
        network = resnet_aloft(
            network,
            positions=tuple(hparams["aloft_positions"]),
            mode=self.MODE,
            alpha=hparams["aloft_alpha"],
            mask_ratio=hparams["aloft_mask_ratio"],
            perturb_prob=hparams["aloft_perturb_prob"],
            rev=self.REV,
            noise_mode=self.NOISE_MODE,
        )

        self.featurizer = networks.wrap_backbone(input_shape, self.hparams, network)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def update(self, x, y, **kwargs):
        # 所有源域拼成一个 batch：ALOFT 的 Sigma 是沿 batch 维统计的，
        # 混合多域样本才能让方差真正反映域间差异（论文 3.3 的出发点）
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        aloft_modules = []
        if self.NOISE_MODE == "domain":
            domain_ids = torch.cat([
                torch.full(
                    (minibatch.shape[0],), index,
                    device=minibatch.device, dtype=torch.long)
                for index, minibatch in enumerate(x)
            ])
            aloft_modules = find_aloft_modules(self)
            for module in aloft_modules:
                module.set_domain_context(domain_ids)
        try:
            logits = self.predict(all_x)
        finally:
            for module in aloft_modules:
                module.clear_domain_context()
        loss = F.cross_entropy(logits, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class ALOFT_E(ALOFT_DG):
    """按元素建模低频分布，论文 Eq.(6)(7)，论文 4.2 节给的 alpha = 1.0。"""

    MODE = "E"


class ALOFT_Stable_E(ALOFT_E):
    """Unchanged low-frequency ALOFT-E with clean-view, weak reweighting.

    StableNet learns detached weights from an eval-mode feature forward. The
    gradient-bearing forward retains all baseline ALOFT perturbations. Source
    domain mass is preserved; no label-conditional balancing is performed.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        stable_mix_at_step(
            0, hparams["stable_mix_max"], hparams["stable_warmup_steps"],
            hparams["stable_ramp_steps"])
        if not 0.0 <= hparams["stable_presave_ratio"] < 1.0:
            raise ValueError("stable_presave_ratio must be in [0, 1)")
        if num_domains < 1 or hparams["batch_size"] < 1:
            raise ValueError("ALOFT_Stable_E requires non-empty source batches")

        size = hparams["batch_size"] * num_domains
        self.register_buffer("pre_features", torch.zeros(size, self.featurizer.n_outputs))
        self.register_buffer("pre_logits", torch.ones(size, 1))
        self.register_buffer("history_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("update_count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _save_clean_history(self, features, raw):
        count = int(self.history_count.item())
        # Populate with actual clean features, never synthetic zero observations.
        rate = 1.0 / (count + 1) if count < 10 else 1.0 - self.hparams["stable_presave_ratio"]
        self.pre_features.mul_(1 - rate).add_(features.detach(), alpha=rate)
        self.pre_logits.mul_(1 - rate).add_(raw.detach(), alpha=rate)
        self.history_count.add_(1)

    def update(self, x, y, **kwargs):
        sizes = [batch.shape[0] for batch in x]
        if (len(x) != self.num_domains or len(y) != self.num_domains
                or any(size != self.hparams["batch_size"] for size in sizes)
                or any(labels.shape[0] != size for labels, size in zip(y, sizes))):
            raise ValueError(
                "ALOFT_Stable_E expects batch_size examples per source domain")

        all_x, all_y = torch.cat(x), torch.cat(y)
        step = int(self.update_count.item())
        mix = stable_mix_at_step(
            step, self.hparams["stable_mix_max"],
            self.hparams["stable_warmup_steps"], self.hparams["stable_ramp_steps"])
        raw = self.pre_logits.new_ones((all_x.shape[0], 1))
        original_weights = raw / all_x.shape[0]
        clean = None

        # mix_max=0 is an exact baseline ablation: no extra forward or RNG draws.
        # During warmup the clean history is populated, but weights are uniform.
        if self.hparams["stable_mix_max"] > 0.0:
            clean = clean_features(self.featurizer, all_x)
            if clean.shape != self.pre_features.shape or not torch.isfinite(clean).all():
                raise ValueError("Invalid clean features for ALOFT_Stable_E")
            if mix > 0.0:
                ready = int(self.history_count.item()) > 0
                original_weights, raw = learn_weights(
                    clean, self.pre_features if ready else clean,
                    self.pre_logits if ready else torch.ones_like(raw),
                    self.hparams, epoch=int(kwargs.get("epoch", 0)))

        weights = domain_balanced_weights(raw, sizes, mix).flatten()
        # clean_features has already restored the original module modes.
        logits = self.predict(all_x)
        per_sample = F.cross_entropy(logits, all_y, reduction="none")
        if mix == 0.0:
            # Match the baseline reduction exactly, including floating-point order.
            loss = F.cross_entropy(logits, all_y)
        else:
            loss = (weights * per_sample).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if clean is not None:
            self._save_clean_history(clean, raw)
        self.update_count.add_(1)

        with torch.no_grad():
            ess = 1.0 / weights.square().sum()
            result = {
                "loss": loss.item(),
                "loss_unweighted": per_sample.mean().item(),
                "stable_mix": mix,
                "weight_min": weights.min().item(),
                "weight_max": weights.max().item(),
                "weight_ess": ess.item(),
                "weight_ess_ratio": (ess / weights.numel()).item(),
                "stable_raw_ess": (1.0 / original_weights.square().sum()).item(),
            }
            for index, domain_weights in enumerate(weights.split(sizes)):
                result[f"weight_domain_{index}"] = domain_weights.sum().item()
            for label in range(self.num_classes):
                selected = all_y == label
                result[f"weight_class_{label}"] = weights[selected].sum().item()
                result[f"batch_class_{label}"] = selected.float().mean().item()
        return result


class ALOFT_S(ALOFT_DG):
    """按通道统计量建模低频分布，论文 Eq.(8)-(14)，论文 4.2 节给的 alpha = 0.9。"""

    MODE = "S"
class ALOFT_rev_E(ALOFT_DG):
    MODE = "E"
    REV = True
class ALOFT_rev_S(ALOFT_DG):
    MODE = "S"
    REV = True


class ALOFT_HF_E(ALOFT_DG):
    """Unambiguous high-frequency ALOFT-E baseline."""

    MODE = "E"


class ALOFT_CovLF_E(ALOFT_DG):
    """ALOFT-E with full batch-covariance low-frequency directions."""

    MODE = "E"
    NOISE_MODE = "covariance"


class ALOFT_DomainLF_E(ALOFT_DG):
    """ALOFT-E with source-domain low-frequency mean directions."""

    MODE = "E"
    NOISE_MODE = "domain"


class ALOFT_CB_rev_E(Algorithm):
    """High-frequency ALOFT-E on layer1/2 and a band codebook on layer3."""

    MODE = "E"
    REV = True

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[0] == 3, "ALOFT_CB_rev_E supports R18/R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)

        if networks.is_vit(hparams):
            network = networks.Featurizer(input_shape, hparams)
        elif hparams["resnet18"]:
            network = torchvision.models.resnet18(pretrained=hparams["pretrained"])
        else:
            network = torchvision.models.resnet50(pretrained=hparams["pretrained"])

        network = resnet_aloft_cb(
            network,
            positions=tuple(hparams["aloft_positions"]),
            mode=self.MODE,
            alpha=hparams["aloft_alpha"],
            mask_ratio=hparams["aloft_mask_ratio"],
            perturb_prob=hparams["aloft_perturb_prob"],
            rev=self.REV,
            codebook=hparams["aloft_cb_codebook"],
            group_size=hparams["aloft_cb_group_size"],
            n_bands=hparams["aloft_cb_bands"],
            strength_max=hparams["aloft_cb_strength_max"],
            decay=hparams["aloft_cb_decay"],
            dead_patience=hparams["aloft_cb_dead_patience"],
            reservoir_size=hparams["aloft_cb_reservoir"],
        )

        self.featurizer = networks.wrap_backbone(input_shape, self.hparams, network)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        result = {"loss": loss.item()}
        codebooks = find_band_codebooks(self)
        if codebooks:
            result.update(codebooks[0].diagnostics())
        return result

    def predict(self, x):
        return self.network(x)


class _ALOFTSketchBase(Algorithm):
    """Direction/radial high-frequency perturbation for sketch features."""

    TOPOLOGY = False

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[0] == 3, "ALOFT sketch variants support R18/R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)

        if networks.is_vit(hparams):
            network = networks.Featurizer(input_shape, hparams)
        elif hparams["resnet18"]:
            network = torchvision.models.resnet18(pretrained=hparams["pretrained"])
        else:
            network = torchvision.models.resnet50(pretrained=hparams["pretrained"])

        network = resnet_aloft_sketch(
            network,
            num_classes=num_classes,
            positions=("layer1", "layer2"),
            alpha=hparams["aloft_alpha"],
            mask_ratio=hparams["aloft_mask_ratio"],
            perturb_prob=hparams["aloft_perturb_prob"],
            group_size=hparams["aloft_sketch_group_size"],
            radial_bands=hparams["aloft_sketch_radial_bands"],
            orientation_bins=hparams["aloft_sketch_orientation_bins"],
            strength_max=hparams["aloft_sketch_strength_max"],
            warmup_steps=hparams["aloft_sketch_warmup"],
            ramp_steps=hparams["aloft_sketch_ramp"],
            class_decay=hparams["aloft_sketch_class_decay"],
            class_min_count=hparams["aloft_sketch_class_min_count"],
            ready_ratio=hparams["aloft_sketch_ready_ratio"],
            gate_power=hparams["aloft_sketch_gate_power"],
            topology=self.TOPOLOGY,
            skeleton_iters=hparams["aloft_sketch_skeleton_iters"],
        )

        self.featurizer = networks.wrap_backbone(input_shape, self.hparams, network)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())
        self.topology_weight = (
            hparams["aloft_sketch_topo_weight"] if self.TOPOLOGY else 0.0)

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        step = int(kwargs.get("step", 0))
        modules = find_sketch_spectrum_modules(self)
        for module in modules:
            module.set_batch_context(all_y, step)
        try:
            logits = self.predict(all_x)
        finally:
            for module in modules:
                module.clear_batch_context()

        classification_loss = F.cross_entropy(logits, all_y)
        topology_loss = collect_sketch_topology_loss(self)
        if topology_loss is None:
            topology_loss = classification_loss.new_zeros(())
        loss = classification_loss + self.topology_weight * topology_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        result = {
            "loss": loss.item(),
            "ce_loss": classification_loss.item(),
        }
        result.update(collect_sketch_diagnostics(self))
        return result

    def predict(self, x):
        return self.network(x)


class ALOFT_Sketch_rev_E(_ALOFTSketchBase):
    """Sketch-spectrum perturbation with phase/orientation hard constraints."""


class ALOFT_SketchTopo_rev_E(_ALOFTSketchBase):
    """Sketch-spectrum perturbation plus soft-skeleton consistency."""

    TOPOLOGY = True


class _ALOFTStructureBase(Algorithm):
    """Original ALOFT-E plus train-only sketch-structure recovery."""

    REV = False
    DIRECTION = True
    TOPOLOGY = True

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[0] == 3, "ALOFT structure variants support R18/R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)

        if networks.is_vit(hparams):
            network = networks.Featurizer(input_shape, hparams)
        elif hparams["resnet18"]:
            network = torchvision.models.resnet18(pretrained=hparams["pretrained"])
        else:
            network = torchvision.models.resnet50(pretrained=hparams["pretrained"])

        network = resnet_aloft_structure(
            network,
            direction=self.DIRECTION,
            topology=self.TOPOLOGY,
            hidden_channels=hparams["aloft_struct_head_channels"],
            skeleton_iters=hparams["aloft_struct_skeleton_iters"],
            positions=tuple(hparams["aloft_positions"]),
            mode="E",
            alpha=hparams["aloft_alpha"],
            mask_ratio=hparams["aloft_mask_ratio"],
            perturb_prob=hparams["aloft_perturb_prob"],
            rev=self.REV,
        )

        self.featurizer = networks.wrap_backbone(input_shape, self.hparams, network)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.structure_targets = SketchStructureTargets()
        self.optimizer = self.new_optimizer(self.network.parameters())

        self.struct_warmup = int(hparams["aloft_struct_warmup"])
        self.struct_ramp = int(hparams["aloft_struct_ramp"])
        self.struct_weights = {
            "dir": float(hparams["aloft_struct_dir_weight"]),
            "stroke": float(hparams["aloft_struct_stroke_weight"]),
            "closure": float(hparams["aloft_struct_closure_weight"]),
            "cldice": float(hparams["aloft_struct_cldice_weight"]),
        }

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        step = int(kwargs.get("step", 0))
        scale = structure_scale_at_step(
            step, self.struct_warmup, self.struct_ramp)
        probes = find_structure_probes(self)

        targets = self.structure_targets(
            all_x, direction=self.DIRECTION, topology=self.TOPOLOGY
        ) if scale > 0.0 else None
        for probe in probes:
            probe.set_batch_context(targets, scale)
        try:
            logits = self.predict(all_x)
        finally:
            for probe in probes:
                probe.clear_batch_context()

        classification_loss = F.cross_entropy(logits, all_y)
        auxiliary = collect_structure_losses(self)
        zero = classification_loss.new_zeros(())
        direction_loss = auxiliary.get("dir", zero)
        stroke_loss = auxiliary.get("stroke", zero)
        closure_loss = auxiliary.get("closure", zero)
        cldice_loss = auxiliary.get("cldice", zero)
        weighted_auxiliary = (
            self.struct_weights["dir"] * direction_loss
            + self.struct_weights["stroke"] * stroke_loss
            + self.struct_weights["closure"] * closure_loss
            + self.struct_weights["cldice"] * cldice_loss
        )
        structure_loss = weighted_auxiliary * scale
        loss = classification_loss + structure_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        result = {
            "loss": loss.item(),
            "ce_loss": classification_loss.item(),
            "struct_scale": float(scale),
            "struct_loss": structure_loss.item(),
            "struct_dir_loss": direction_loss.item(),
            "struct_stroke_loss": stroke_loss.item(),
            "struct_closure_loss": closure_loss.item(),
            "struct_cldice_loss": cldice_loss.item(),
        }
        result.update(collect_structure_diagnostics(self))
        return result

    def predict(self, x):
        return self.network(x)


class ALOFT_StructLF_E(_ALOFTStructureBase):
    """Observed strongest low-frequency ALOFT-E plus both structure heads."""


class ALOFT_StructLF_Dir_E(_ALOFTStructureBase):
    """Low-frequency ALOFT-E with direction recovery only."""

    TOPOLOGY = False


class ALOFT_StructLF_Topo_E(_ALOFTStructureBase):
    """Low-frequency ALOFT-E with stroke/topology recovery only."""

    DIRECTION = False


class ALOFT_StructHF_E(_ALOFTStructureBase):
    """True high-frequency ALOFT-E plus both structure heads."""

    REV = True


class ALOFT_StructHF_Dir_E(ALOFT_StructLF_Dir_E):
    """True high-frequency ALOFT-E with direction recovery only."""

    REV = True


class ALOFT_StructHF_Topo_E(ALOFT_StructLF_Topo_E):
    """True high-frequency ALOFT-E with stroke/topology recovery only."""

    REV = True

class AWWSL_DG(Algorithm):
    """AWWSL (CVPR'23) 的 ResNet 版：把频域扰动插进 ResNet 的 stage 之间。

    训练目标就是普通交叉熵，AWWSL 不引入任何损失项、也不引入可学习参数。
    """

    MODE = "E"          # 由子类覆盖
    REV = False         # 由子类覆盖

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[0] == 3, "AWWSL supports R18/R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)

        if networks.is_vit(hparams):
            network = networks.Featurizer(input_shape, hparams)
        elif hparams["resnet18"]:
            network = torchvision.models.resnet18(pretrained=hparams["pretrained"])
        else:
            network = torchvision.models.resnet50(pretrained=hparams["pretrained"])

        # 必须在预训练权重加载完之后再包 ALOFT，否则 state_dict 键名对不上
        network = resnet_awwsl(
            network,
            positions=tuple(hparams["aloft_positions"]),
            mode=self.MODE,
            alpha=hparams["aloft_alpha"],
            mask_ratio=hparams["aloft_mask_ratio"],
            perturb_prob=hparams["aloft_perturb_prob"],
            rev=self.REV
        )

        self.featurizer = networks.wrap_backbone(input_shape, self.hparams, network)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def update(self, x, y, **kwargs):
        # 所有源域拼成一个 batch：ALOFT 的 Sigma 是沿 batch 维统计的，
        # 混合多域样本才能让方差真正反映域间差异（论文 3.3 的出发点）
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class AWWSL_E(AWWSL_DG):
    """按元素建模低频分布，论文 Eq.(6)(7)，论文 4.2 节给的 alpha = 1.0。"""

    MODE = "E"


class AWWSL_S(AWWSL_DG):
    """按通道统计量建模低频分布，论文 Eq.(8)-(14)，论文 4.2 节给的 alpha = 0.9。"""

    MODE = "S"
class AWWSL_rev_E(AWWSL_DG):
    MODE = "E"
    REV = True
class AWWSL_rev_S(AWWSL_DG):
    MODE = "S"
    REV = True


class FQ(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if networks.is_vit(hparams):
            network = networks.Featurizer(input_shape, hparams)
        elif hparams["resnet18"]:
            network = torchvision.models.resnet18(pretrained=hparams["pretrained"])
        else:
            network = torchvision.models.resnet50(pretrained=hparams["pretrained"])

        network = resnet_freqquant(              # 权重加载之后再包
            network,
            layers=tuple(hparams["fq_layers"]),
            mode=hparams["fq_mode"],
            band=hparams["fq_band"],
            levels=hparams["fq_levels"],
            mask_ratio=hparams["fq_mask_ratio"],
            low_gain=hparams["fq_low_gain"],
            quantile=hparams["fq_quantile"],
            codebook=hparams["fq_codebook"],
            groups=hparams["fq_groups"],
            strength_max=hparams["fq_strength_max"],
        )
        self.featurizer = networks.wrap_backbone(input_shape, self.hparams, network)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())
        self.aux_weight = hparams["fq_aux_weight"]

    def update(self, x, y, **kwargs):
        all_x, all_y = torch.cat(x), torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        aux = collect_aux_loss(self.network)
        total = loss + self.aux_weight * aux
        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()
        return {"loss": loss.item(),
                "aux": float(aux) if torch.is_tensor(aux) else aux}

    def predict(self, x):
        return self.network(x)

class StableNet(Algorithm):
    def __init__(
        self,
        input_shape,
        num_classes,
        num_domains,
        hparams,
    ):
        super().__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
        )

        self.featurizer = networks.Featurizer(input_shape, hparams)
        self.classifier = nn.Linear(
            self.featurizer.n_outputs,
            num_classes,
        )

        self.network = nn.Sequential(
            self.featurizer,
            self.classifier,
        )

        self.optimizer = self.new_optimizer(
            self.network.parameters()
        )

        # DomainBed 的 batch_size 是每个源域的 batch size。
        size = hparams["batch_size"] * num_domains
        dim = self.featurizer.n_outputs

        self.register_buffer(
            "pre_features",
            torch.zeros(size, dim),
        )
        self.register_buffer(
            "pre_logits",
            torch.ones(size, 1),
        )
        self.register_buffer(
            "update_count",
            torch.tensor(0, dtype=torch.long),
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        features = self.featurizer(all_x)
        logits = self.classifier(features)

        if features.shape != self.pre_features.shape:
            raise ValueError(
                "StableNet expects a fixed combined source batch"
            )

        step = int(self.update_count.item())
        epoch = int(kwargs.get("epoch", 0))

        self.optimizer.zero_grad()

        weights, raw = learn_weights(
            features,
            self.pre_features,
            self.pre_logits,
            self.hparams,
            epoch,
        )

        per_sample = F.cross_entropy(
            logits,
            all_y,
            reduction="none",
        )

        # weights 已经归一化，使用 sum，不能再 mean。
        loss = (per_sample * weights.flatten()).sum()

        loss.backward()
        self.optimizer.step()

        # 历史状态不保留计算图。
        with torch.no_grad():
            if epoch == 0 and step < 10:
                rate = 1.0 / (step + 1)
            else:
                rate = 1.0 - self.hparams["stable_presave_ratio"]

            self.pre_features.mul_(1 - rate).add_(
                features.detach().float(),
                alpha=rate,
            )

            self.pre_logits.mul_(1 - rate).add_(
                raw,
                alpha=rate,
            )

            self.update_count.add_(1)

        return {
            "loss": loss.item(),
            "weight_min": weights.min().item(),
            "weight_max": weights.max().item(),
            "weight_ess": (
                1 / weights.square().sum()
            ).item(),
        }

    def predict(self, x):
        return self.network(x)
