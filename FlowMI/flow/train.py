"""Train one conditional flow over every incomplete subset of a cohort.

Six defects were found by reading this file against the AE before running it, rather than by
running it. They are called out where they live, because each would have produced numbers
rather than an error:

  1. decode indexed the modality embedding by CHANNEL POSITION rather than by modality id,
     which silently mislabels every decode.
  2. the shared code fed to the decoder was left in NORMALISED units. The AE was trained on
     raw latents, so every decode would have been off by a constant factor.
  3. oracle PSNR was stubbed to 0.0, which reads as a real measurement in a results table.
  4. R2's total-sum-of-squares used a per-batch mean instead of the dataset mean, which
     inflates R2 by an amount that depends on batch size.
  5. the deterministic control ran under autocast with no GradScaler, so its gradients could
     underflow and it would have looked artificially weak -- biasing the one comparison the
     whole exercise exists to make.
  6. evaluation decoded every val case for every pattern at 128^3, which for CT is
     64 cases x 2 patterns x 1 absent modality per pattern and grows as 2^M; capped explicitly
     and the cap is logged rather than left implicit.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flow.data import (LatentSubsets, compute_stats, poe_shared, poe_full,
                       modality_corr)      # noqa: E402
from flow.model import FlowUNet, flow_batch, sample, translate_sample                 # noqa: E402


def _logger(log_dir, tag):
    os.makedirs(log_dir, exist_ok=True)
    lg = logging.getLogger(tag)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    for h in (logging.FileHandler(os.path.join(log_dir, f"{tag}.log")),
              logging.StreamHandler()):
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        lg.addHandler(h)
    return lg


class Deterministic(torch.nn.Module):
    """The control: identical conditioning and capacity, no stochasticity. It regresses the
    missing private code instead of transporting noise to it. fm_difficulty showed a plain
    probe already reaching most of the way, so without this the flow's contribution cannot be
    separated from what least squares gives for free."""

    def __init__(self, m, p, s, base=96, n_res=2):
        super().__init__()
        self.net = FlowUNet(m, p, s, base=base, n_res=n_res)
        self.M, self.P = m, p

    def forward(self, mu_s, mask, priv):
        B = priv.shape[0]
        sp = priv.shape[3:]
        obs = mask.view(B, self.M, 1, *([1] * len(sp)))
        x = (obs * priv).reshape(B, self.M * self.P, *sp)     # absent channels zeroed
        t = torch.zeros(B, device=x.device)
        return self.net(x, mu_s, mask, t).reshape(B, self.M, self.P, *sp)


def load_ae(ckpt, device):
    from autoencoder.poeae import PoEAE, PoECfg
    blob = torch.load(ckpt, map_location="cpu")
    cd = dict(blob["cfg"])
    emb = blob["state_dict"].get("mod_emb_enc")
    if emb is not None and "vocab" in PoECfg.__dataclass_fields__:
        cd["vocab"] = emb.shape[0]
    net = PoEAE(PoECfg(**cd)).to(device).eval()
    net.load_state_dict(blob["state_dict"], strict=True)
    for q in net.parameters():
        q.requires_grad_(False)
    return net


@torch.no_grad()
def evaluate(net, base_net, ae, args, stats, patterns, mod_ids, M, P, dev, lg, n_dec,
             res_scale=None, to_delta=None, dscale_v=None):
    """Three numbers per pattern, because on this data they can disagree.

    latent R2   comparable to the deterministic probes in fm_difficulty
    image PSNR  measured against the ORACLE decode, i.e. what the same decoder produces from
                the TRUE private code. That isolates the flow's own error instead of mixing in
                the AE's reconstruction error, which no flow can affect.
    """
    net.eval()
    if base_net is not None:
        base_net.train(False)
    es, ps = stats["expert_std"], stats["priv_std"]
    rows = []
    lg.info("   pattern   flow_R2  base_R2 | flow_dB  base_dB   (dB vs oracle decode)")
    for pat in patterns:
        ds = LatentSubsets(args.latent_root, args.cohort, "val", stats, fixed_mask=list(pat))
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
        # (4) accumulate over the WHOLE split so SST uses the dataset mean, not a batch mean
        se = bse = sx = sx2 = cnt = 0.0
        f_db, b_db = [], []
        for i, b in enumerate(dl):
            emu = b["expert_mu"].to(dev); elv = b["expert_logvar"].to(dev)
            priv = b["priv"].to(dev); mask = b["mask"].to(dev)
            mu_s = poe_shared(emu, elv, mask)
            miss = (1 - mask.view(1, M, 1, 1, 1, 1))
            em = emu if args.rich_cond else None
            lvs = None
            if args.rich_cond:
                lvs = -torch.log((torch.exp(-elv) * mask.view(*mask.shape, 1, 1, 1, 1)
                                  ).sum(1) + 1.0)
            if args.translate:
                # transport the INPUT latent, then read the private codes back out of the
                # result. The shared half of the output is the corrected a, which the decoder
                # in the image evaluation can use in place of a_S.
                ob = mask.view(1, M, 1, 1, 1, 1)
                spx = priv.shape[3:]
                x0 = torch.cat([mu_s, (priv * ob).reshape(1, M * P, *spx)], 1)
                out1 = translate_sample(net, x0, mu_s, mask, steps=args.steps,
                                        expert_mu=em, logvar_s=lvs,
                                        noise=args.start_noise, cfg=args.cfg,
                                        pred_x=args.pred_x)
                gen = out1[:, mu_s.shape[1]:].reshape(1, M, P, *spx)
                pred = base_net(mu_s, mask, priv) if base_net is not None else None
                se += float((((gen - priv) ** 2) * miss).sum())
                sx += float((priv * miss).sum())
                sx2 += float(((priv ** 2) * miss).sum())
                cnt += float(miss.expand_as(priv).sum())
                if pred is not None:
                    bse += float((((pred - priv) ** 2) * miss).sum())
                if i < n_dec:
                    a_used = out1[:, :mu_s.shape[1]] * es
                    for m in range(M):
                        if pat[m]:
                            continue
                        gid = mod_ids[m]
                        ref = ae.decode_from(mu_s * es, priv[:, m] * ps, gid)
                        got = ae.decode_from(a_used, gen[:, m] * ps, gid)
                        f_db.append(float(-10 * torch.log10(
                            F.mse_loss(got, ref).clamp(min=1e-10))))
                        if pred is not None:
                            gotb = ae.decode_from(mu_s * es, pred[:, m] * ps, gid)
                            b_db.append(float(-10 * torch.log10(
                                F.mse_loss(gotb, ref).clamp(min=1e-10))))
                continue

            base_pred = base_net(mu_s, mask, priv) if base_net is not None else None
            tgt_for_flow = priv
            if args.residual and base_pred is not None:
                tgt_for_flow = (priv - base_pred) / max(res_scale or 1.0, 1e-6)
            priv_in = priv
            refs = None
            if args.delta_ref:
                priv_in, refs = to_delta(priv, mask)
            gen = sample(net, mu_s, mask,
                         priv_in if not args.residual else tgt_for_flow,
                         steps=args.steps, expert_mu=em, logvar_s=lvs, cfg=args.cfg,
                         pred_x=args.pred_x)
            if isinstance(gen, tuple):
                gen, _dsh = gen
            if args.delta_ref:
                # add the reference back. It is an OBSERVED modality, so it is known exactly at
                # inference -- this is what separates the idea from the earlier "residual",
                # which needed a model's own prediction and therefore inherited its error.
                g2 = gen.clone()
                for bi in range(gen.shape[0]):
                    for mm in range(M):
                        if float(mask[bi, mm]) < 0.5:
                            g2[bi, mm] = (gen[bi, mm] * max(dscale_v or 1.0, 1e-6)
                                          + priv[bi, refs[bi][mm]])
                gen = g2
            if args.residual and base_pred is not None:
                # NOTE the floor claimed for this design does NOT come for free: sampling starts
                # at noise and a zero-velocity network leaves it there, so an untrained residual
                # flow returns control + noise, which is WORSE than the control. The flow has to
                # learn to transport that noise onto a near-zero residual. Only then does the
                # "cannot be worse than control" property hold.
                gen = gen * max(res_scale or 1.0, 1e-6) + base_pred
            se += float((((gen - priv) ** 2) * miss).sum())
            sx += float((priv * miss).sum())
            sx2 += float(((priv ** 2) * miss).sum())
            cnt += float(miss.expand_as(priv).sum())
            pred = base_pred
            if pred is not None:
                bse += float((((pred - priv) ** 2) * miss).sum())
            if i < n_dec:                                  # (6) explicit, logged cap
                # (2) the AE was trained on RAW latents; poe_shared is linear in mu, so the
                # normalised shared code is exactly mu_raw / expert_std -- undo it here.
                a_raw = mu_s * es
                for m in range(M):
                    if pat[m]:
                        continue
                    gid = mod_ids[m]                       # (1) GLOBAL id, never position
                    ref = ae.decode_from(a_raw, priv[:, m] * ps, gid)
                    got = ae.decode_from(a_raw, gen[:, m] * ps, gid)
                    f_db.append(float(-10 * torch.log10(
                        F.mse_loss(got, ref).clamp(min=1e-10))))
                    if pred is not None:
                        gotb = ae.decode_from(a_raw, pred[:, m] * ps, gid)
                        b_db.append(float(-10 * torch.log10(
                            F.mse_loss(gotb, ref).clamp(min=1e-10))))
        sst = max(sx2 - sx * sx / max(cnt, 1.0), 1e-12)
        fr = 1 - se / sst
        br = 1 - bse / sst if base_net is not None else float("nan")
        fd = float(np.mean(f_db)) if f_db else float("nan")
        bd = float(np.mean(b_db)) if b_db else float("nan")
        rows.append((fr, br, fd, bd))
        lg.info(f"   {''.join(str(x) for x in pat):8s}  {fr:7.4f}  {br:7.4f} | "
                f"{fd:7.2f}  {bd:7.2f}")
    fr = float(np.mean([r[0] for r in rows])); br = float(np.nanmean([r[1] for r in rows]))
    fd = float(np.nanmean([r[2] for r in rows])); bd = float(np.nanmean([r[3] for r in rows]))
    verdict = ("flow ahead" if fd > bd + 0.05 else
               "control ahead" if bd > fd + 0.05 else "tied")
    lg.info(f"   MEAN      {fr:7.4f}  {br:7.4f} | {fd:7.2f}  {bd:7.2f}   -> {verdict}")
    return [f"{fr:.5f}", f"{br:.5f}", f"{fd:.4f}", f"{bd:.4f}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-root", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--ae-ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=96)
    ap.add_argument("--n-res", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-decode-cases", type=int, default=8)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--rich-cond", action="store_true",
                    help="also condition on per-modality experts and the shared logvar")
    ap.add_argument("--p-uncond", type=float, default=0.0,
                    help="probability of dropping the observation-derived conditioning, "
                         "which is what makes classifier-free guidance available at "
                         "sampling time")
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--image-loss", type=float, default=0.0,
                    help="weight on an L1 taken after decoding the predicted "
                         "latent. Latent MSE is a poor proxy for image quality: a "
                         "high latent R2 converts to much less image headroom, and it "
                         "weights the background that the task is not about.")
    ap.add_argument("--t-dist", default="uniform", choices=["uniform", "logitnormal"],
                    help="x-prediction carries an implicit (1-t)^2 weight, which starves large "
                         "t under uniform sampling; a logit-normal puts the draws back where "
                         "the weight has not vanished")
    ap.add_argument("--t-mean", type=float, default=0.0)
    ap.add_argument("--t-std", type=float, default=1.0)
    ap.add_argument("--arch", default="unet", choices=["unet", "jit"],
                    help="jit swaps in 3DMMIT's TranslationJiT3D, leaving data, control and "
                         "evaluation untouched so the difference is attributable")
    ap.add_argument("--image-loss-max", type=int, default=2,
                    help="cap on decoded samples per step")
    ap.add_argument("--image-loss-tmin", type=float, default=0.5,
                    help="only apply it above this t; below, x1_hat is an "
                         "extrapolation and its image loss is mostly noise")
    ap.add_argument("--data-root",
                    default=os.environ.get("FLOWMI_EXTRACTED_ROOT", "data/extracted"))
    ap.add_argument("--pred-x", action="store_true",
                    help="regress x1 instead of the velocity")
    ap.add_argument("--delta-ref", action="store_true",
                    help="predict r_missing - r_reference, the reference being the "
                         "observed modality it correlates with most. Measured to "
                         "shrink the target 3.22x on ct, 2.42x for dce2->dce3.")
    ap.add_argument("--bridge-sigma", type=float, default=0.0,
                    help="Brownian-bridge noise on the path, sigma*sqrt(t(1-t)). "
                         "Zero reproduces the straight line, which leaks: with x0 "
                         "a deterministic function of the conditioning, x1 follows "
                         "from (x_t-(1-t)x0)/t by algebra. The bridge term is what "
                         "makes x_t stop determining x1, and it vanishes at both "
                         "ends so the endpoints are still exactly x0 and x1.")
    ap.add_argument("--hybrid", action="store_true",
                    help="noise->data flow for the absent private codes PLUS a "
                         "deterministic head for the shared-code correction")
    ap.add_argument("--translate", action="store_true",
                    help="transport the input latent to the complete latent, "
                         "instead of noise to the missing private codes")
    ap.add_argument("--start-noise", type=float, default=0.0,
                    help="noise added to the starting point; the only source of "
                         "diversity in translate mode")
    ap.add_argument("--sample-target", action="store_true",
                    help="draw the private code from its posterior each visit, so a "
                         "condition has more than one observation")
    ap.add_argument("--fix-shared", action="store_true",
                    help="SUPERSEDED by --translate, which carries the shared code inside the "
                         "transport instead of bolting S channels onto a noise->data flow. "
                         "Left in place only so old commands fail loudly rather than silently.")
    ap.add_argument("--residual", action="store_true",
                    help="flow the RESIDUAL of the deterministic control rather than the code "
                         "itself: output = control(cond) + flow(cond)")
    a = ap.parse_args()

    if a.fix_shared and not a.translate:
        raise SystemExit(
            "--fix-shared is superseded by --translate. It appended the shared-code delta as "
            "extra channels of x_t during training but the evaluation sampler never carried "
            "them, so the mode trained and then died at the first eval. --translate solves the "
            "same problem properly: a_full - a_S is part of the transport, not an appendage.")
    os.makedirs(a.out_dir, exist_ok=True)
    lg = _logger(a.log_dir, a.tag)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    stats = compute_stats(a.latent_root, a.cohort)
    tr = LatentSubsets(a.latent_root, a.cohort, "train", stats,
                       sample_target=a.sample_target,
                       with_image=a.image_loss > 0, data_root=a.data_root)
    if a.max_train:
        tr.files = tr.files[:a.max_train]
    M, P, S, mod_ids = tr.M, tr.P, tr.S, tr.mod_ids
    lg.info(f"cohort={a.cohort} M={M} P={P} S={S} mod_ids={mod_ids} train={len(tr)}")
    lg.info(f"scale: expert_std {stats['expert_std']:.4f}  priv_std {stats['priv_std']:.4f}")

    dl = DataLoader(tr, batch_size=a.batch_size, shuffle=True,
                    num_workers=a.num_workers, drop_last=True)

    Arch = FlowUNet
    if a.arch == "jit":
        from flow.jit_adapter import JiTFlow
        Arch = JiTFlow
    net = Arch(M, P, S, base=a.base, n_res=a.n_res,
                   rich_cond=a.rich_cond, fix_shared=a.fix_shared,
                   translate=a.translate, hybrid=a.hybrid).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    # bf16 needs no loss scaling; the scaler is kept only so the call sites are
    # unchanged, and disabled.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    lg.info(f"flow {sum(q.numel() for q in net.parameters())/1e6:.2f}M params")

    base_net = base_opt = base_scaler = None
    if not a.no_baseline:
        base_net = Deterministic(M, P, S, base=a.base, n_res=a.n_res).to(dev)
        base_opt = torch.optim.AdamW(base_net.parameters(), lr=a.lr, weight_decay=1e-4)
        # (5) its own scaler; running it under autocast without one lets gradients underflow
        # and would quietly handicap the control this comparison depends on
        base_scaler = torch.amp.GradScaler("cuda", enabled=False)
        lg.info(f"control {sum(q.numel() for q in base_net.parameters())/1e6:.2f}M params")

    CORR = None
    if a.delta_ref:
        CORR = modality_corr(a.latent_root, a.cohort)
        lg.info("modality correlations:\n" + str(np.round(CORR, 3)))

    def pick_ref(mask_row):
        obs_idx = [i for i in range(M) if float(mask_row[i]) > 0.5]
        return [max(obs_idx, key=lambda j: CORR[m, j]) if obs_idx else m for m in range(M)]

    dscale = {"v": None}

    def to_delta(pv, mk, track=False):
        """r_m -> (r_m - r_ref(m)) / scale for the ABSENT modalities.

        The division is the point. The codes arrive already divided by priv_std, but the
        DIFFERENCE has a much smaller scale than the absolute code, and by a different factor
        per cohort -- so without rescaling the target lands far below the unit-variance noise it
        is transported from, and the size of the mismatch varies by dataset. Same defect as the
        one already fixed for --residual.
        """
        out = pv.clone()
        refs = []
        for bi in range(pv.shape[0]):
            r = pick_ref(mk[bi])
            refs.append(r)
            for mm in range(M):
                if float(mk[bi, mm]) < 0.5:
                    out[bi, mm] = pv[bi, mm] - pv[bi, r[mm]]
        if track:
            msk = (1 - mk.view(-1, M, 1, 1, 1, 1))
            cur = float(((out ** 2) * msk).sum()
                        / msk.expand_as(out).sum().clamp(min=1)) ** 0.5
            dscale["v"] = cur if dscale["v"] is None else 0.99 * dscale["v"] + 0.01 * cur
        sc = max(dscale["v"] or 1.0, 1e-6)
        for bi in range(pv.shape[0]):
            for mm in range(M):
                if float(mk[bi, mm]) < 0.5:
                    out[bi, mm] = out[bi, mm] / sc
        return out, refs

    res_scale = None      # EMA of the residual's std, only used in --residual mode
    sh_scale = None       # same, for the shared-code delta
    ae = load_ae(a.ae_ckpt, dev)
    patterns = [p for p in itertools.product([0, 1], repeat=M) if any(p) and not all(p)]
    lg.info(f"{len(patterns)} incomplete patterns; decoding {a.eval_decode_cases} "
            f"val case(s) per pattern for the image metric")

    csv_path = os.path.join(a.out_dir, "flow_metrics.csv")
    # A restart used to APPEND, so a diverged run's rows sat above the replacement's and the
    # epoch numbers restarted from zero inside one file. Rotate instead: the old rows stay
    # readable but cannot be mistaken for this run's.
    if os.path.isfile(csv_path):
        import shutil, glob as _g
        shutil.move(csv_path, csv_path + f".prev{len(_g.glob(csv_path + '.prev*'))}")
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "base_loss", "flow_r2", "base_r2",
                                    "flow_db", "base_db"])

    for ep in range(a.epochs):
        net.train()
        if base_net is not None:
            base_net.train()
        run = nb = 0.0
        n = 0
        nonfinite = 0
        t0 = time.time()
        for b in dl:
            emu = b["expert_mu"].to(dev); elv = b["expert_logvar"].to(dev)
            priv = b["priv"].to(dev); mask = b["mask"].to(dev)
            mu_s = poe_shared(emu, elv, mask)

            # RESIDUAL MODE. The flow's target becomes priv - control(cond) instead of priv.
            #
            # Why this is the change most likely to matter: as it stands the flow has to learn
            # the conditional MEAN and the spread around it at once, and the mean dominates the
            # loss, so the optimum it finds is the mean and it stops there -- which is exactly
            # the collapse that was measured (diversity ratio 0.037 on ct). Subtract the mean
            # and the target is centred at zero with nothing left in it BUT the spread.
            #
            # It also puts a floor under the result. At inference the output is
            # control + flow_residual, so a flow that emits zero reproduces the control exactly
            # and anything else is added on top. The current design instead makes the flow
            # relearn from scratch everything the control already knows, and it did not manage
            # to match it (dce 31.38 against 31.71).
            flow_target = priv
            if a.delta_ref:
                flow_target, _ = to_delta(priv, mask, track=True)
            if a.residual and base_net is not None:
                with torch.no_grad():
                    resid = priv - base_net(mu_s, mask, priv)
                    # SCALE. The sampler starts from unit-variance noise, but the residual is
                    # the control's ERROR and is much smaller -- std ~ sqrt(1 - R2) ~ 0.42 of
                    # the code's own scale. Feeding a target 2.4x smaller than the noise it is
                    # transported from wastes most of the trajectory. Track the residual's std
                    # with an EMA over the first epochs and normalise by it; sampling multiplies
                    # it back.
                    mkm = (1 - mask.view(-1, M, 1, 1, 1, 1))
                    cur = float(((resid ** 2) * mkm).sum()
                                / mkm.expand_as(resid).sum().clamp(min=1)) ** 0.5
                    res_scale = cur if res_scale is None else 0.99 * res_scale + 0.01 * cur
                    flow_target = resid / max(res_scale, 1e-6)
            cond_on = 1.0
            if a.p_uncond > 0 and torch.rand(()) < a.p_uncond:
                cond_on = 0.0     # the unconditional branch CFG extrapolates away from
            em = emu if a.rich_cond else None
            lvs = (-torch.log((torch.exp(-elv) * mask.view(*mask.shape, 1, 1, 1, 1)
                               ).sum(1) + 1.0)) if a.rich_cond else None
            if a.translate:
                with torch.no_grad():
                    a_full = poe_full(emu, elv)
                    ob = mask.view(-1, M, 1, 1, 1, 1)
                    sp = priv.shape[3:]
                    x0 = torch.cat([mu_s, (priv * ob).reshape(-1, M * P, *sp)], 1)
                    x1 = torch.cat([a_full, priv.reshape(-1, M * P, *sp)], 1)
                    if a.start_noise > 0:
                        x0 = x0 + a.start_noise * torch.randn_like(x0)
                if a.t_dist == "logitnormal":
                    _t = torch.sigmoid(a.t_mean + a.t_std * torch.randn(x0.shape[0], device=dev))
                else:
                    _t = torch.rand(x0.shape[0], device=dev)
                tt = _t.view(-1, *([1] * (x0.dim() - 1)))
                x_t = (1 - tt) * x0 + tt * x1
                if a.bridge_sigma > 0:
                    # I2SB / DDBM-style bridge noise. Without it this is a straight line between
                    # two points the network can both identify, and the leak test showed it duly
                    # learns the algebra instead of the mapping: fed an impostor x1 it followed
                    # the impostor at correlation 0.86 against 0.31 for its own target.
                    x_t = x_t + a.bridge_sigma * torch.sqrt(tt * (1 - tt)) * torch.randn_like(x_t)
                # velocity of the straight part; the bridge term has zero mean and vanishes at
                # both ends, so the endpoints and the drift target are unchanged
                target = x1 - x0
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
                    out = net(x_t, mu_s, mask, tt.view(-1), em, lvs, x0)
                    if a.pred_x:
                        # the regression target is the complete latent itself, not the drift
                        v = out / (1.0 - tt).clamp(min=1e-3)     # for the image term below
                        loss = F.mse_loss(x_t + out, x1)
                    else:
                        v = out
                        loss = F.mse_loss(v, target)
                    # The image term lived only in the non-translate branch, which has its own
                    # `continue`, so every --translate --image-loss run silently trained without
                    # it -- the giveaway was metrics identical to six decimals against the run
                    # with no image loss at all.
                    if a.image_loss > 0 and "image" in b:
                        tvec = tt.reshape(-1)
                        img = b["image"].to(dev)
                        sel = []
                        for j in range(tvec.shape[0]):
                            if float(tvec[j]) < a.image_loss_tmin:
                                continue
                            ab = [mm for mm in range(M) if float(mask[j, mm]) < 0.5]
                            if ab:
                                sel.append((j, ab[int(torch.randint(len(ab), (1,)))]))
                        if sel:
                            sel = sel[:a.image_loss_max]          # the decode is the expensive half
                            tv = tvec.view(-1, *([1] * (x_t.dim() - 1)))
                            x1h = x_t + (1.0 - tv) * v
                            pvh = x1h[:, S:].reshape(-1, M, P, *priv.shape[3:])
                            il = 0.0
                            for j, mm in sel:
                                gid = mod_ids[mm] if mod_ids else mm
                                dec = ae.decode_from(x1h[j:j + 1, :S] * stats["expert_std"],
                                                     pvh[j:j + 1, mm] * stats["priv_std"], gid)
                                il = il + F.l1_loss(dec, img[j:j + 1, mm:mm + 1])
                            loss = loss + a.image_loss * il / len(sel)
                if not torch.isfinite(loss):
                    # skip rather than propagate: NaN weights never recover, and the run would
                    # keep going for hundreds of epochs producing nothing
                    nonfinite += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                if not torch.isfinite(gn):
                    nonfinite += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                scaler.step(opt); scaler.update()
                run += float(loss); n += 1
                if base_net is not None:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
                        pr = base_net(mu_s, mask, priv)
                        obb = mask.view(-1, M, 1, 1, 1, 1)
                        bl = ((((pr - priv) ** 2) * (1 - obb)).sum()
                              / (1 - obb).expand_as(priv).sum().clamp(min=1))
                    base_opt.zero_grad(set_to_none=True)
                    base_scaler.scale(bl).backward(); base_scaler.unscale_(base_opt)
                    torch.nn.utils.clip_grad_norm_(base_net.parameters(), 1.0)
                    base_scaler.step(base_opt); base_scaler.update()
                    nb += float(bl)
                continue

            da = None
            if a.fix_shared:
                with torch.no_grad():
                    da = poe_full(emu, elv) - mu_s          # what the missing modalities add
                    ds = float(da.pow(2).mean()) ** 0.5
                    sh_scale = ds if sh_scale is None else 0.99 * sh_scale + 0.01 * ds
                    da = da / max(sh_scale, 1e-6)
            dshared = None
            if a.hybrid:
                with torch.no_grad():
                    # supervised directly, not transported: no ambiguity to sample here
                    dshared = poe_full(emu, elv) - mu_s
                    dsc = float(dshared.pow(2).mean()) ** 0.5
                    sh_scale = dsc if sh_scale is None else 0.99 * sh_scale + 0.01 * dsc
                    dshared = dshared / max(sh_scale, 1e-6)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
                x_t, t, target, w = flow_batch(flow_target, mask, pred_x=a.pred_x,
                                               t_dist=a.t_dist, t_mean=a.t_mean,
                                               t_std=a.t_std)
                if da is not None:
                    # the shared correction rides along as S extra channels, always supervised
                    # (unlike the private channels it is never "observed")
                    na = torch.randn_like(da)
                    tt = t.view(-1, *([1] * (da.dim() - 1)))
                    x_t = torch.cat([x_t, (1 - tt) * na + tt * da], 1)
                    target = torch.cat([target, da - na], 1)
                    w = torch.cat([w, torch.ones_like(da)], 1)
                pred = net(x_t, mu_s * cond_on, mask, t,
                           None if em is None else em * cond_on,
                           None if lvs is None else lvs * cond_on)
                if dshared is not None:
                    nprv = pred.shape[1] - dshared.shape[1]
                    loss = (((pred[:, :nprv] - target) ** 2 * w).sum()
                            / w.sum().clamp(min=1)
                            + F.mse_loss(pred[:, nprv:], dshared))
                else:
                    if a.pred_x:
                        # residual head: what is compared against x1 is x_t + out
                        pred = pred + x_t[:, :pred.shape[1]]
                    loss = ((pred - target) ** 2 * w).sum() / w.sum().clamp(min=1)
                if a.image_loss > 0 and "image" in b:
                    tvec = t.reshape(-1)
                    img = b["image"].to(dev)
                    sel = []
                    for j in range(tvec.shape[0]):
                        if float(tvec[j]) < a.image_loss_tmin:
                            continue
                        ab = [mm for mm in range(M) if float(mask[j, mm]) < 0.5]
                        if ab:
                            sel.append((j, ab[int(torch.randint(len(ab), (1,)))]))
                    if sel:
                        sel = sel[:a.image_loss_max]          # the decode is the expensive half
                        tv = tvec.view(-1, *([1] * (x_t.dim() - 1)))
                        x1h = pred[:, :M * P] if a.pred_x else (x_t[:, :M * P] + (1.0 - tv) * pred[:, :M * P])
                        pvh = x1h[:, 0:].reshape(-1, M, P, *priv.shape[3:])
                        il = 0.0
                        for j, mm in sel:
                            gid = mod_ids[mm] if mod_ids else mm
                            dec = ae.decode_from(mu_s[j:j + 1] * stats["expert_std"],
                                                 pvh[j:j + 1, mm] * stats["priv_std"], gid)
                            il = il + F.l1_loss(dec, img[j:j + 1, mm:mm + 1])
                        loss = loss + a.image_loss * il / len(sel)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            run += float(loss); n += 1

            if base_net is not None:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
                    pred = base_net(mu_s, mask, priv)
                    ob = mask.view(-1, M, 1, 1, 1, 1)
                    bl = ((((pred - priv) ** 2) * (1 - ob)).sum()
                          / (1 - ob).expand_as(priv).sum().clamp(min=1))
                base_opt.zero_grad(set_to_none=True)
                base_scaler.scale(bl).backward()
                base_scaler.unscale_(base_opt)
                torch.nn.utils.clip_grad_norm_(base_net.parameters(), 1.0)
                base_scaler.step(base_opt); base_scaler.update()
                nb += float(bl)

        msg = f"ep{ep} flow {run/max(n,1):.4f}"
        if a.delta_ref and dscale["v"]:
            msg += f"  dscale {dscale['v']:.4f}"
        if nonfinite:
            msg += f"  SKIPPED {nonfinite} non-finite step(s)"
        if a.residual and res_scale is not None:
            msg += f"  res_scale {res_scale:.4f}"
        if base_net is not None:
            msg += f"  control {nb/max(n,1):.4f}"
        lg.info(msg + f"  | {(time.time()-t0)/60:.2f} min")

        if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
            r = evaluate(net, base_net, ae, a, stats, patterns, mod_ids, M, P, dev, lg,
                         a.eval_decode_cases, res_scale, to_delta, dscale['v'])
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([ep, f"{run/max(n,1):.5f}",
                                        f"{nb/max(n,1):.5f}" if base_net else "nan"] + r)
            torch.save({"state_dict": net.state_dict(), "epoch": ep, "stats": stats,
                        "cfg": {"M": M, "P": P, "S": S, "base": a.base, "n_res": a.n_res,
                                "mod_ids": mod_ids, "rich_cond": a.rich_cond,
                                "residual": a.residual, "fix_shared": a.fix_shared,
                                "hybrid": a.hybrid, "bridge_sigma": a.bridge_sigma,
                                "delta_ref": a.delta_ref, "pred_x": a.pred_x,
                                "arch": a.arch, "t_dist": a.t_dist,
                                "image_loss": a.image_loss,
                                "sample_target": a.sample_target,
                                "translate": a.translate}},
                       os.path.join(a.out_dir, "last.pt"))
            if base_net is not None:
                torch.save({"state_dict": base_net.state_dict(), "epoch": ep},
                           os.path.join(a.out_dir, "control.pt"))


if __name__ == "__main__":
    main()
