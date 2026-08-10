"""Where, in frequency, does the flow fall short -- of the target, and of the AE oracle?

Two bands behave differently. Below 1/4 of the image Nyquist the latent grid (32^3 against a
128^3 image) can REPRESENT the content, so anything missing there is recoverable by a better
flow. Above it there is no latent representation at all and the decoder can only synthesise;
CT measurements already showed its synthesis above k~0.47 is worse than emitting zero.

So the question a high-frequency loss hinges on is: does the flow's spectrum already sit on the
AE oracle's? If yes, the recoverable band is exhausted and such a loss can only push into the
band where the decoder is counterproductive. If no, the gap is real and a latent-side
high-frequency emphasis targets it for free, with no decoder in the loop.
"""
import argparse, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contrastx_dataloader import COHORTS
from contrastx_dataloader.prep_dataset import PREP_ROOT, PrepSet
from autoencoder.maisi_slot import MaisiSlotAE
from flow.slot_fm import SlotFlow, sample, tailclip_fwd, tailclip_inv, flow_start

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", required=True)
ap.add_argument("--fm-ckpt", required=True)
ap.add_argument("--cases", type=int, default=6)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--sample-k", type=int, default=8)
ap.add_argument("--cube", type=int, default=64)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
keys = COHORTS[a.cohort][1]
M = len(keys)
fb = torch.load(a.fm_ckpt, map_location=dev, weights_only=False)
Z, scale, tclip = fb["Z"], fb["scale"], fb.get("tailclip", 0.0)
ns, bm = float(fb.get("noise_start", 0.0)), fb.get("base", "zero")
net = SlotFlow(M, Z, gate=fb.get("gate_head", False)).to(dev).eval()
net.load_state_dict(fb["ema"])
ae = MaisiSlotAE(latent_channels=Z, ckpt=None).to(dev).eval()
ae.load_state_dict(torch.load(fb["ae_ckpt"], map_location=dev, weights_only=False)["model"])

TASK = {"breast_dce": ("DCE1_to_DCE2", (1, 0, 0), "dce2"),
        "ct": ("CT_to_CTC", (1, 0), "ctc")}[a.cohort]
tname, pat, key = TASK
m = keys.index(key)
NB = 16


def fwd(z):
    z = z * scale
    return tailclip_fwd(z, tclip) if tclip > 0 else z


def inv(z):
    z = tailclip_inv(z, tclip) if tclip > 0 else z
    return z / scale


def radial(v):
    V = np.fft.fftn(v - v.mean())
    P = (V * np.conj(V)).real
    n = v.shape[0]
    f = np.fft.fftfreq(n) * 2.0
    kx, ky, kz = np.meshgrid(f, f, f, indexing="ij")
    k = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2) / np.sqrt(3.0)
    idx = np.clip((k * NB).astype(int), 0, NB - 1)
    return np.array([P[idx == b].mean() if (idx == b).any() else 0.0 for b in range(NB)])


def cube(v, s):
    for ax in range(3):
        n = v.shape[ax]
        st = max((n - s) // 2, 0)
        v = np.take(v, range(st, min(st + s, n)), axis=ax)
    return v


ds = PrepSet(PREP_ROOT, a.cohort, "test", z_window=128, train=False)
T = np.zeros(NB); O = np.zeros(NB); F = np.zeros(NB); n = 0
for i in range(min(a.cases, len(ds))):
    x = ds[i]["x"]
    x = x[..., :(x.shape[-1] // 32 * 32)]
    z0 = min(int(ds[i]["z0"]), x.shape[-1])
    xs = x.unsqueeze(0).to(dev).float()
    with torch.no_grad():
        mu, _ = ae.encode(xs)
        orc = ae.decode(mu)
        mask = torch.tensor([pat], dtype=torch.float32, device=dev)
        zf = fwd(mu)
        gs = []
        for _ in range(max(a.sample_k, 1)):
            st, cd, _, _ = flow_start(zf, mask, noise=ns, base=bm)
            gs.append(sample(net, st, mask, M, Z, steps=a.steps, cond=cd))
        g = torch.stack(gs).mean(0).reshape(1, M, Z, *mu.shape[3:])
        fm = ae.decode(inv(g))
    T += radial(cube(xs[0, m, ..., :z0].cpu().numpy(), a.cube))
    O += radial(cube(orc[0, m, ..., :z0].cpu().numpy(), a.cube))
    F += radial(cube(fm[0, m, ..., :z0].cpu().numpy(), a.cube))
    n += 1

T, O, F = T / n, O / n, F / n
print(f"\n{a.cohort}  {tname}  {n} cases   latent is /4 -> latent Nyquist at k = 0.25\n")
print(f"{'k/Nyq':>8} {'oracle/tgt':>11} {'fm/tgt':>9} {'fm/oracle':>10}   note")
for b in range(NB):
    k = (b + 0.5) / NB
    note = "<- latent Nyquist" if b == int(0.25 * NB) else ""
    print(f"{k:8.3f} {O[b]/max(T[b],1e-30):11.3f} {F[b]/max(T[b],1e-30):9.3f} "
          f"{F[b]/max(O[b],1e-30):10.3f}   {note}")
lo = slice(0, int(0.25 * NB)); hi = slice(int(0.25 * NB), NB)
print(f"\nbelow k=0.25 (representable, so recoverable):  fm/oracle "
      f"{F[lo].sum()/max(O[lo].sum(),1e-30):.3f}")
print(f"above k=0.25 (synthesised only):              fm/oracle "
      f"{F[hi].sum()/max(O[hi].sum(),1e-30):.3f}")
print("fm/oracle near 1 means the flow has taken what the autoencoder can give, and a")
print("high-frequency loss on the FLOW cannot help -- the autoencoder is the thing to change.")
print("SPEC_OK")
