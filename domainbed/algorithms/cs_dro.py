# Adapted from CS-DRO (KDD 2026), https://github.com/gyeomo/CS-DRO.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# See domainbed/models/CS_DRO_LICENSE.txt for the upstream MIT license.

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .algorithms import Algorithm
from domainbed import networks
from domainbed.optimizers import get_optimizer
from domainbed.models.cs_dro import LightEncoder, NotearsClassifier


def adjust_lr_zt(optimizer, lr0, epoch):
    lr = lr0 * (1.0 / np.sqrt(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def _rbf_gram(x: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-8):
    """
    x: (n, 1) or (n,)
    sigma: scalar tensor
    returns K: (n, n)
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    # squared pairwise distance
    d2 = torch.cdist(x, x, p=2).pow(2)  # (n, n)
    gamma = 1.0 / (2.0 * (sigma.pow(2) + eps))
    K = torch.exp(-gamma * d2)
    return K

def _median_heuristic_sigma(x: torch.Tensor, eps: float = 1e-8):
    """
    x: (n,) or (n,1)
    returns sigma (scalar tensor)
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    with torch.no_grad():
        d = torch.cdist(x, x, p=2)  # (n, n)
        # remove diagonal
        n = d.size(0)
        mask = ~torch.eye(n, device=d.device, dtype=torch.bool)
        vals = d[mask]
        med = vals.median()
        # sigma=median distance (common practical choice)
        sigma = med.clamp_min(eps)
    return sigma

def hsic_rbf(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8):
    """
    HSIC with RBF kernels (biased estimator).
    x, y: (n,) or (n,1)
    """
    n = x.size(0)
    sigma_x = _median_heuristic_sigma(x, eps=eps)
    sigma_y = _median_heuristic_sigma(y, eps=eps)

    K = _rbf_gram(x, sigma_x, eps=eps)
    L = _rbf_gram(y, sigma_y, eps=eps)

    # Centering: Kc = HKH, Lc = HLH
    H = torch.eye(n, device=K.device) - (1.0 / n) * torch.ones((n, n), device=K.device)
    Kc = H @ K @ H
    Lc = H @ L @ H

    hsic = (Kc * Lc).sum() / ((n - 1) ** 2 + eps)  # trace(Kc @ Lc) == sum elementwise
    return hsic

def hsic_dim_penalty(r: torch.Tensor, num_pairs: int = 64, eps: float = 1e-8):
    """
    r: (n, d)
    """
    n, d = r.shape
    if d < 2:
        return r.new_tensor(0.0)

    pairs = []
    for _ in range(num_pairs):
        i = torch.randint(0, d, (1,), device=r.device).item()
        j = torch.randint(0, d - 1, (1,), device=r.device).item()
        if j >= i:
            j += 1
        pairs.append((i, j))

    penalty = r.new_tensor(0.0)
    for i, j in pairs:
        penalty = penalty + hsic_rbf(r[:, i], r[:, j], eps=eps)

    return penalty / max(len(pairs), 1)

def variance_floor_penalty(r: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4):
    """
    r: (n, d)
    """
    std = torch.sqrt(r.var(dim=0, unbiased=False) + eps)  # (d,)
    # VICReg
    return torch.relu(target_std - std).mean()

class CS_DRO(Algorithm):
    """Reference CS-DRO; trainer supplies radius refreshes and DAG updates.

    The DAG/mask is learned only at steps 101..299. SWAD must start at 301,
    after the warmup and with the final mask already copied into its buffers.
    """

    SWAD_START_STEP = 301

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CS_DRO, self).__init__(input_shape,
                                   num_classes,
                                   num_domains,
                                   hparams)
        self.num_classes = num_classes
        self.num_domains = num_domains
        if num_domains < 2:
            raise ValueError("CS_DRO requires at least two source domains")
        if num_classes < 2:
            raise ValueError("CS_DRO requires at least two classes")
        if hparams['hidden_size'] != hparams['out_dim']:
            raise ValueError("CS_DRO requires hidden_size == out_dim")
        if hparams['num_hidden_layers'] != 0:
            raise ValueError("CS_DRO's reference encoder has no hidden layers")
        self.feature_dim = hparams['out_dim']
        self.featurizer = networks.Featurizer(input_shape, self.hparams)

        # An additional layer for dimensionality reduction
        self.encoder = LightEncoder(self.featurizer.n_outputs,
                                             hparams['out_dim'],
                                             hparams['hidden_size'],
                                             hparams["num_hidden_layers"])

        self.dag_mlp = NotearsClassifier(hparams['out_dim'], num_classes)
        self.dag_mlp.weight_pos.data[:-1, -1].fill_(1.0)
        self.dag_mlp.projection()

        self.clf = nn.Linear(hparams['out_dim'], num_classes)

        # Classifier for DAG learning
        self.rec_classifier = nn.Linear(hparams['out_dim'], num_classes)

        # CS_DRO hyperparameter
        self.tau = hparams['hp_tau']
        self.lambda_G = hparams['lambda_G']
        self.kappa = hparams['kappa']
        self.lambda_r = hparams['lambda_r']

        self.adv_steps = self.hparams.get("adv_steps", 15)
        if self.adv_steps < 1:
            raise ValueError("adv_steps must be at least 1")
        self.adv_step_size = self.hparams.get("adv_step_size", 0.05)
        self.adv_gamma = self.hparams.get("adv_gamma", 0.001)
        self.target_rho = self.hparams.get("target_rho", 0.5)

        self.adv_beta = self.hparams.get("adv_beta", self.kappa)

        self.mask_m_min = self.hparams.get("m_min", 1e-6)
        self.mask_m_max = self.hparams.get("m_max", 1-1e-6)
        self.mask_temp = self.hparams.get("temp", 0.1)


        # DAG hyperparameter (from iDAG)
        self.proto_m = self.hparams["ema_ratio"]
        self.lambda1 = self.hparams["lambda1"]
        self.rho_max = self.hparams["rho_max"]
        self.alpha = self.hparams["alpha"]
        self.rho = self.hparams["rho"]
        self._h_val = np.inf


        self.counter = 0

        self.full_cov = hparams.get('full_cov', False)
        self.gauss_min_count = hparams.get('gauss_min_count', 5)
        k = int(1000.0/(num_classes*num_domains))
        self.gauss_k = hparams.get('gauss_k', max(200,k))
        self.gauss_eps = hparams.get('gauss_eps', 1e-4)
        self.gauss_shrink = hparams.get('gauss_shrink', 0.1)

        self.register_buffer(
            "m_ema",
            torch.ones(hparams['out_dim']))
        self.flag = True

        self.register_buffer(
            "prototypes",
            torch.zeros(num_domains, num_classes, hparams['out_dim']))
        self.register_buffer(
            "prototypes_label",
            torch.arange(num_classes).repeat(num_domains))
        S, C, Dz = num_domains, num_classes, hparams['out_dim']
        self.register_buffer("gauss_count", torch.zeros(S, C, dtype=torch.long))
        self.register_buffer("gauss_mean", torch.zeros(S, C, Dz))
        if self.full_cov:
            self.register_buffer("gauss_M2", torch.zeros(S, C, Dz, Dz))
        else:
            self.register_buffer("gauss_M2_diag", torch.zeros(S, C, Dz))

        self.gauss_decay = float(hparams.get('gauss_decay', 0.9))
        hl = hparams.get('gauss_half_life', None)
        if hl is not None and hl > 0:
            self.gauss_decay = math.exp(-math.log(2.0) / float(hl))

        S, C, Dz = num_domains, num_classes, hparams['out_dim']
        self.register_buffer("gauss_mass", torch.zeros(S, C))       # [S, C], float



        dag_params  = list(self.dag_mlp.parameters()) + \
              list(self.rec_classifier.parameters())
        main_params = list(self.featurizer.parameters()) + list(self.encoder.parameters()) + list(self.clf.parameters())
        self.optimizer = get_optimizer(hparams["optimizer"], [{"params": main_params}],
                                    lr=self.hparams["lr"], weight_decay=1e-6)
        self.opt_dag  = get_optimizer(hparams["optimizer"], [{"params": dag_params}],
                                    lr=1e-4, weight_decay=1e-6)

    @torch.no_grad()
    def _get_total_effect_masks(self, update_ema: bool = True):
        mc = self.dag_mlp._adj_sub()[:self.feature_dim, -1]
        c = mc.median()
        mad = (mc - c).abs().median().clamp(min=1e-6)
        m = torch.sigmoid((mc - c) / mad)
        m_new = m.clamp(self.mask_m_min, self.mask_m_max).detach()
        mu = self.hparams.get("m_ema_mu", 0.1)
        if self.flag:
            self.m_ema.copy_(m_new)
            self.flag = False
        if update_ema and self.training:
            self.m_ema.copy_((1.0 - mu) * self.m_ema + mu * m_new)
        m = self.m_ema
        return m.unsqueeze(0)

    def structure_preserving(self, z, y, d):
        W_base = self.dag_mlp._adj().detach()
        grads = []
        dmax = int(d.max().item())
        for dd in range(dmax + 1):
            index = (d == dd)
            if index.sum() == 0:
                continue
            zz = z[index]
            yy = y[index].unsqueeze(1).float()
            alpha = torch.ones_like(W_base, requires_grad=True)
            W_eff = W_base * alpha
            x_aug = torch.cat((zz, yy), dim=1)
            proto_rec = (x_aug @ W_eff)[:, :self.dag_mlp.dims]
            loss_ = 0.5 * (zz - proto_rec).pow(2).sum(-1).mean()
            (grad,) = torch.autograd.grad(loss_, alpha, retain_graph=True, create_graph=True)
            grads.append(grad)
        grads_flat = torch.stack([g.reshape(-1) for g in grads], dim=0)  # [E, P]
        loss_inv = (grads_flat ** 2).sum()
        mean_grad = grads_flat.mean(0, keepdim=True)
        loss_inv = loss_inv + ((grads_flat - mean_grad) ** 2).sum()
        return loss_inv

    def inner_maximization(self, z, y, d,  G_y):
        for p in self.clf.parameters(): p.requires_grad_(False)
        for p in self.dag_mlp.parameters(): p.requires_grad_(False)
        B = z.size(0)
        step_size = self.adv_step_size
        d_data = []
        for dd in range(int(d.max().item()) + 1):
            index = d==dd
            zz = z[index]
            d_data.append(zz.mean())
        l2_sum = 0.0
        cnt = 0
        for i in range(len(d_data)):
            for j in range(i + 1, len(d_data)):
                l2_sum = l2_sum + (d_data[i] - d_data[j]).norm(p=2)
                cnt += 1
        norm = l2_sum / cnt
        norm = torch.clamp(norm, min=1.0, max=100.0)
        step_size = step_size * norm

        delta = torch.empty_like(z).uniform_(-1e-4, 1e-4).requires_grad_(True)
        optimizer_zt = torch.optim.Adam([delta], lr=step_size)
        for nnn in range(self.adv_steps):
            optimizer_zt.zero_grad()
            logits = self.clf(z * G_y + delta)
            CE = F.cross_entropy(logits, y)

            z_pair = torch.cat([z, z + delta], dim=0)
            y_pair = torch.cat([y, y], dim=0)
            d_pair = torch.cat([torch.zeros(B, device=z.device, dtype=d.dtype),
                                torch.ones (B, device=z.device, dtype=d.dtype)],dim=0)
            D_G = self.structure_preserving(z_pair, y_pair, d_pair)

            W_G = (delta.pow(2) * G_y).sum(dim=-1).mean()

            loss_zt = - (CE - self.adv_beta * D_G - self.adv_gamma * W_G)
            loss_zt.backward()
            optimizer_zt.step()
            adjust_lr_zt(optimizer_zt, step_size, nnn+1)

        for p in self.clf.parameters(): p.requires_grad_(True)
        for p in self.dag_mlp.parameters(): p.requires_grad_(True)
        with torch.no_grad():
            self.adv_gamma = torch.clamp(self.adv_gamma + 1e-4 * (W_G.detach() - self.target_rho), min=1e-6, max=1.0).item()
            self.adv_beta = torch.clamp(self.adv_beta + 1e-4 * (D_G.detach() - self.kappa), min=1e-2, max=10.0).item()
        return delta.detach().clone()

    def update_dag(self):
        X_syn, y_syn, _ = self.sample_from_gaussians(K=self.gauss_k)
        if X_syn is None:
            raise RuntimeError(
                "CS_DRO has no valid domain/class Gaussian statistics. "
                "Collect source batches at steps 50..301 before DAG updates; "
                "check gauss_min_count and batch_size."
            )
        proto_rec, masked_proto_c = self.dag_mlp(X_syn, y_syn)

        loss_rec = 0.5*(proto_rec - X_syn).pow(2).sum(-1).mean()
        loss_cls_aux = F.cross_entropy(self.rec_classifier(masked_proto_c), y_syn)

        h_val = self.dag_mlp.h_func()
        penalty = 0.5 * self.rho * h_val * h_val + self.alpha * h_val
        l1_reg = self.lambda1 * self.dag_mlp.w_l1_reg()

        if self.counter % 50 == 0:
            if self.rho < self.rho_max and h_val > 0.25 * self._h_val:
                self.rho *= 10
                self.alpha += self.rho * h_val.item()
            self._h_val = h_val.item()
        self.counter += 1

        loss_dag = loss_rec + loss_cls_aux + penalty + l1_reg

        loss = loss_dag
        self.opt_dag.zero_grad()
        loss.backward()
        self.opt_dag.step()
        self.dag_mlp.projection()
        self._get_total_effect_masks(update_ema=True)

        return {
            "rec": loss_rec.item(),
            "aux_cls": loss_cls_aux.item(),
            "l1": l1_reg.item(),
            "h": h_val.item(),
        }

    def update(self, x, y, **kwargs):
        device = self.prototypes.device
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        domain_labels = torch.cat([torch.ones(len(_y)) * i for i, _y in enumerate(y)]).long().to(device)

        z = self.encoder(self.featurizer(all_x))
        if kwargs["step"] >= 50 and kwargs["step"] <= 301:
            self.update_stats_and_prototypes(z.detach(), all_y, domain_labels)
        # Warm-Up
        if kwargs["step"] <= 300:
            logits_clean = self.clf(z)
            loss_ce = F.cross_entropy(logits_clean, all_y)
            weight = torch.linspace(0.001,0.1,301)[kwargs["step"]].item()
            loss_hsic = hsic_dim_penalty(z, num_pairs=64)
            loss_var = variance_floor_penalty(z, target_std=1.0)
            loss = loss_ce + weight*(0.1*loss_hsic + loss_var)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {"ce": loss.item(),"kl" : 0,"inv": 0,}

        G_y = self.m_ema.unsqueeze(0).clone().detach()
        delta_adv = self.inner_maximization(z.detach(), all_y, domain_labels, G_y)

        # outer minimization
        z_cls = z * G_y
        logits_clean = self.clf(z_cls)
        logits_adv = self.clf(z_cls + delta_adv)

        loss_G = self.structure_preserving(z+delta_adv, all_y, domain_labels)

        loss_adv = F.cross_entropy(logits_adv, all_y)

        tau = self.tau
        p0_c = F.softmax(logits_clean/tau, dim=-1).detach()   # (B, C)
        p0_a_k = F.softmax(logits_adv/tau, dim=-1).detach()
        log_q_c = F.log_softmax(logits_clean, dim=-1)         # (B, C)
        log_q_a_k = F.log_softmax(logits_adv, dim=-1)

        loss_reg = F.kl_div(log_q_a_k, p0_c, reduction="none").sum(dim=-1).mean()
        loss_reg = loss_reg + F.kl_div(log_q_c, p0_a_k, reduction="none").sum(dim=-1).mean()

        loss = loss_adv + self.lambda_r * loss_reg + self.lambda_G * loss_G
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"ce": loss_adv.item(), "reg" : loss_reg.item(), "inv": loss_G.item()}

    def predict(self, x):
        z = self.encoder(self.featurizer(x))
        G_y = self.m_ema.unsqueeze(0).clone().detach()
        return self.clf(z * G_y)

    def classify(self, z):
        G_y = self.m_ema.unsqueeze(0).clone().detach()
        return self.clf(z * G_y)

    @torch.no_grad()
    def update_stats_and_prototypes(self, z, y, d):
        device = self.prototypes.device
        S, C, Dz = self.prototypes.shape
        B = S * C

        z = z.to(device)
        y = y.to(device).long()
        d = d.to(device).long()

        # --- (a) ---
        idx = (d * C + y).view(-1)                   # [N]
        nb = torch.bincount(idx, minlength=B)        # [B]
        nb_sc = nb.clamp_min(1).unsqueeze(-1)        # [B,1]

        sum_x  = torch.zeros(B, Dz, device=device, dtype=z.dtype)
        sum_x.scatter_add_(0, idx[:, None].expand(-1, Dz), z)
        sum_x2 = torch.zeros(B, Dz, device=device, dtype=z.dtype)
        sum_x2.scatter_add_(0, idx[:, None].expand(-1, Dz), z * z)

        mb = sum_x / nb_sc                           # [B, Dz]
        M2_b_diag = sum_x2 - nb_sc * (mb * mb)       # [B, Dz]

        if self.full_cov:
            outer = torch.einsum('ni,nj->nij', z, z)           # [N, Dz, Dz]
            sum_outer = torch.zeros(B, Dz, Dz, device=device, dtype=z.dtype)
            sum_outer.index_add_(0, idx, outer)
            mb_outer = torch.einsum('bi,bj->bij', mb, mb)
            M2_b_full = sum_outer - nb.view(B, 1, 1) * mb_outer  # [B, Dz, Dz]

        # --- (b) ---
        n1_long = self.gauss_count.view(B)                 # [B]
        m1      = self.gauss_mean.view(B, Dz)              # [B, Dz]
        mass1   = self.gauss_mass.view(B)                  # [B]

        gamma = float(self.gauss_decay)
        mass1 = mass1 * gamma

        if self.full_cov:
            M2_1 = self.gauss_M2.view(B, Dz, Dz)
            M2_1 = M2_1 * gamma
        else:
            M2_1 = self.gauss_M2_diag.view(B, Dz)
            M2_1 = M2_1 * gamma

        # --- (c) ---
        nb_f    = nb.float()
        mass_new = mass1 + nb_f                          # [B]
        mass_new_sc = mass_new.clamp_min(1.0).unsqueeze(-1)  # [B,1]
        nb_f_sc = nb_f.unsqueeze(-1)                     # [B,1]
        mass1_sc = mass1.unsqueeze(-1)                   # [B,1]

        delta = mb - m1                                  # [B, Dz]
        new_m = (mass1_sc * m1 + nb_f_sc * mb) / mass_new_sc  # [B, Dz]
        alpha = (mass1_sc * nb_f_sc / mass_new_sc)            # [B,1]

        if self.full_cov:
            cross = alpha.view(B, 1, 1) * torch.einsum('bi,bj->bij', delta, delta)
            new_M2 = M2_1 + M2_b_full + cross
        else:
            cross = alpha * (delta * delta)
            new_M2 = M2_1 + M2_b_diag + cross

        has_batch = (nb > 0)

        # --- (d) ---
        m1[has_batch] = new_m[has_batch]
        if self.full_cov:
            M2_1[has_batch] = new_M2[has_batch]
            self.gauss_M2.copy_(M2_1.view(S, C, Dz, Dz))
        else:
            M2_1[has_batch] = new_M2[has_batch]
            self.gauss_M2_diag.copy_(M2_1.view(S, C, Dz))

        self.gauss_mean.copy_(m1.view(S, C, Dz))

        self.gauss_mass.view(B)[has_batch] = mass_new[has_batch]

        self.gauss_count.view(B)[has_batch] = (n1_long[has_batch] + nb[has_batch])

        # --- (e) ---
        proto = self.prototypes.view(B, Dz)
        first_seen = (n1_long == 0) & has_batch
        update_seen = (n1_long > 0) & has_batch
        proto[first_seen] = mb[first_seen]
        if update_seen.any():
            proto[update_seen] = proto[update_seen] * self.proto_m + (1 - self.proto_m) * mb[update_seen]
        self.prototypes.copy_(proto.view(S, C, Dz))



    @torch.no_grad()
    def sample_from_gaussians(self, K: int = None):
        """
        return: X_syn [M, Dz], y_syn [M], d_syn [M]
        """
        S, C, Dz = self.prototypes.shape
        B = S * C
        K = self.gauss_k if K is None else K

        # === (1) ===
        mass_all = self.gauss_mass.view(B).float()                    # [B]
        valid = (mass_all >= float(self.gauss_min_count))             # [B]
        if valid.sum() == 0:
            return None, None, None

        mu_all = self.gauss_mean.view(B, Dz)                          # [B, Dz]
        mu = mu_all[valid]                                            # [Bv, Dz]

        if self.full_cov:
            M2_all = self.gauss_M2.view(B, Dz, Dz)                    # [B, Dz, Dz]
            M2 = M2_all[valid]                                        # [Bv, Dz, Dz]
            n_eff = mass_all[valid].clamp_min(2.0)                    # [Bv]

            cov = M2 / (n_eff - 1.0).view(-1, 1, 1)
            # shrinkage + jitter SPD
            diag = torch.diagonal(cov, dim1=-2, dim2=-1)
            cov = (1.0 - self.gauss_shrink) * cov + self.gauss_shrink * torch.diag_embed(diag)
            cov = cov + torch.eye(Dz, device=cov.device, dtype=cov.dtype).unsqueeze(0) * self.gauss_eps

            L = torch.linalg.cholesky(cov)                            # [Bv, Dz, Dz] (lower)
            eps = torch.randn(mu.shape[0], K, Dz, device=mu.device, dtype=mu.dtype)  # [Bv, K, Dz]
            X_syn = mu.unsqueeze(1) + torch.matmul(eps, L.transpose(-2, -1))         # [Bv, K, Dz]
        else:
            M2_all = self.gauss_M2_diag.view(B, Dz)                   # [B, Dz]
            M2 = M2_all[valid]                                        # [Bv, Dz]
            n_eff = mass_all[valid].clamp_min(2.0).unsqueeze(-1)      # [Bv, 1]

            var = (M2 / (n_eff - 1.0)).clamp_min(self.gauss_eps)      # [Bv, Dz]
            std = var.sqrt()                                          # [Bv, Dz]
            eps = torch.randn(mu.shape[0], K, Dz, device=mu.device, dtype=mu.dtype)  # [Bv, K, Dz]
            X_syn = mu.unsqueeze(1) + eps * std.unsqueeze(1)          # [Bv, K, Dz]

        b_idx = torch.nonzero(valid, as_tuple=False).view(-1)         # [Bv]
        c = (b_idx % C).to(mu.device)
        d = (b_idx // C).to(mu.device)
        y_syn = c.repeat_interleave(K)
        d_syn = d.repeat_interleave(K)

        X_syn = X_syn.reshape(-1, Dz).detach()
        return X_syn, y_syn.long(), d_syn.long()


    def get_emb(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        return self.encoder(self.featurizer(all_x)), all_y

    def cross_entropy_loss(self, logits, y):
        return F.cross_entropy(logits, y, reduction='none')  # per-sample

    def cw_margin_loss(self, logits, y, kappa: float = 0.0):
        """
        CW-style non-targeted margin loss per-sample:
        max(max_{j!=y} logit_j - logit_y, -kappa)
        """
        B, C = logits.shape
        real = logits.gather(1, y.view(-1, 1)).squeeze(1)
        idx = torch.arange(B, device=logits.device)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[idx, y] = False
        max_other = logits.masked_fill(~mask, float('-inf')).max(dim=1).values
        return torch.clamp(max_other - real, min=-kappa)

    def weighted_l2_sq(self, z, z0, w_vec = None):
        disp = z - z0
        if w_vec is None:
            return (disp ** 2).sum(dim=1)
        return (w_vec * (disp ** 2)).sum(dim=1)

    def _cw_penalty_objective(self,
        z: torch.Tensor,
        z0: torch.Tensor,
        y: torch.Tensor,
        c: float,
        w_vec,
        use_margin: bool,
        kappa: float
    ) -> torch.Tensor:
        logits = self.classify(z)
        if use_margin:
            loss_adv = self.cw_margin_loss(logits, y, kappa=kappa)
        else:
            loss_adv = self.cross_entropy_loss(logits, y)
        disp2_vec = self.weighted_l2_sq(z, z0, w_vec=w_vec)
        return (disp2_vec + c * loss_adv).mean()


    def _misclassified(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.classify(z)
        pred = logits.argmax(dim=1)
        return (pred != y)


    def cw_min_weighted_norm_adversary(
        self,
        z0: torch.Tensor,           # (B,d)
        y: torch.Tensor,            # (B,)
        w_vec,
        steps: int = 100,
        lr: float = 1e-2,
        c_init: float = 1e-3,
        c_mul: float = 10.0,
        binary_steps: int = 5,
        restarts: int = 1,
        use_margin: bool = True,
        kappa: float = 0.0,
    ):
        """
        Penalty-form CW in embedding space with binary search on c.
        Returns:
        z_best:    (B,d) closest misclassified embeddings found
        dist_best: (B,) weighted squared distances
        """
        device = z0.device
        B, d = z0.shape
        if w_vec is not None:
            w_vec = w_vec.to(device)

        # Maintain best across binary search & restarts
        dist_best = torch.full((B,), float('inf'), device=device)
        z_best = z0.clone()

        for _ in range(restarts if restarts >= 1 else 1):
            # Binary search on c (shared c across batch for simplicity)
            c_low = torch.full((B,), 0.0, device=device)      # not used in shared-c variant
            c_high = torch.full((B,), float('inf'), device=device)  # not used; we use scalar c
            c = c_init

            for _bs in range(binary_steps):
                # Optimize penalty objective for current c
                z = z0.detach().clone().requires_grad_(True)
                opt = torch.optim.Adam([z], lr=lr)

                for _ in range(steps):
                    opt.zero_grad(set_to_none=True)
                    loss = self._cw_penalty_objective(z, z0, y, c, w_vec, use_margin, kappa)
                    loss.backward()
                    opt.step()

                # Check misclassification and distances
                mis = self._misclassified(z, y)
                d2 = self.weighted_l2_sq(z, z0, w_vec=w_vec).detach()

                # Update best where misclassified and closer
                improved = (mis) & (d2 < dist_best)
                z_best[improved] = z.detach()[improved]
                dist_best[improved] = d2[improved]

                # Adjust c (shared policy): if enough misclassified, decrease c; else increase
                mis_ratio = mis.float().mean().item()
                if mis_ratio > 0.5:
                    c = c / c_mul
                else:
                    c = c * c_mul

        # For samples never misclassified, keep original z0 and inf distance
        return z_best, dist_best

    def recompute_rho_adv_ref(
        self,
        z0,
        y):
        """
        Recompute rho_adv_ref by CW (closest misclassification) in embedding space,
        aggregated over up to `max_batches` batches.
        """
        for p in self.clf.parameters():
            p.requires_grad_(False)
        _, d2 = self.cw_min_weighted_norm_adversary(
            z0=z0, y=y, w_vec=None,
            steps=60,
            lr=1e-2,
            c_init=1e-3,
            c_mul=10.0,
            binary_steps=4,
            restarts=1,
            use_margin=True,
            kappa=0.0,
        )
        for p in self.clf.parameters():
            p.requires_grad_(True)
        d2 = d2.detach().float()
        mask = torch.isfinite(d2)
        if mask.any():
            return d2[mask].cpu().numpy()
        return d2.cpu().numpy()

    def clone(self):
        # Preserve both optimizers and their disjoint parameter groups.
        return copy.deepcopy(self)
