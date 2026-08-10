# FlowMI — latent flow matching for contrast-phase completion

A slot autoencoder encodes each modality into its own latent field; a conditional flow-matching
bridge fills in the fields of whichever modalities are absent, and the decoder turns them back
into images. One model covers every observation pattern — there is no per-pattern head and no
mask branch in the decoder.

## Install

Python 3.10+, PyTorch, MONAI.

```bash
pip install torch monai nibabel numpy scipy pillow
```

## Configure

```bash
export FLOWMI_PREP_ROOT=data/prep128                        # preprocessed volumes
export FLOWMI_MAISI_CKPT=weights/autoencoder_MAISI_Brain.pt # optional encoder init
export FLOWMI_RESULTS=results                               # evaluation output
```

## Run

**1. Autoencoder** — reconstruction only; it never sees a mask.

```bash
python -m autoencoder.train_maisi_slot \
  --cohort ct --out-dir runs/ae \
  --epochs 80 --patch 64 --batch-size 4 --lr 1e-4 --amp \
  --recon-loss l1+l2 --perc-weight 0.1 --adv-weight 0.0 \
  --z-noise 0.05 --kl-weight 1e-7 --val-cap 0
```

**2. Latents** — encode the whole cohort once with the frozen autoencoder.

```bash
python -m autoencoder.slot_latents_cx \
  --ae-ckpt runs/ae/best.pt --cohort ct \
  --out cache/latents.pt --ae-kind maisi
```

**3. Flow** — trains on the cached latents; the autoencoder is not in the loop.

```bash
python -m flow.slot_fm \
  --latents cache/latents.pt --out-dir runs/fm \
  --epochs 600 --start-base copy --noise-start 1.0 \
  --ema 0.995 --val-k 4 --eval-every 5 --patience 24
```

**4. Evaluate**

```bash
python -m flow.slot_eval_image --cohort ct --fm-ckpt runs/fm/best.pt --sample-k 8 --steps 20
python tools/lesion_eval.py    --cohort ct --fm-ckpt runs/fm/best.pt --sample-k 8
python tools/ae_eval.py        runs/ae/best.pt --cohort ct --split val
```

`flow.slot_eval_image` reports PSNR and SSIM per modality. `tools/lesion_eval.py` scores a
region of interest defined as the top percentile of the enhancement map. `tools/ae_eval.py`
scores autoencoders on a full split and reports a paired difference with its standard error.

## Alternative: the PoE autoencoder

A second design is included. Instead of one latent per modality plus a flow that fills the
absent ones, the PoE autoencoder treats the latent as a **posterior over one shared tissue
state**, aggregated over whatever modalities happen to be present:

```
q(a | x_S)  ∝  p(a) · Π_{m∈S} q_m(a | x_m)
```

Gaussian experts make that a precision-weighted sum, so an absent modality simply does not
contribute a term — no mask branch, no per-pattern head. Each modality also carries a small
private code; observed modalities use their own, absent ones draw from the prior, and that
private code is what a flow can fill in.

The two designs put the completion burden in different places. The slot pipeline asks the flow
to predict complete latent fields; PoE hands the shared code over for free from whatever is
observed and asks a flow only for the private residues. They are worth comparing on the same
data before committing to either.

```bash
# PoE autoencoder on a Contrast-X cohort
python -m autoencoder.train_poe_cx \
  --cohort ct --out-dir runs/poe \
  --epochs 80 --patch 64 --batch-size 4 --lr 1e-4 --amp \
  --shared-ch 4 --private-ch 2 \
  --recon-loss l1+l2 --perc-weight 0.1 --val-cap 0 \
  --p-drop 0.5 --drop-warmup 5

# a flow over the private codes
python -m flow.train --ae-ckpt runs/poe/best.pt --out-dir runs/poe_fm
```

`train_poe_cx.py` scores reconstruction and completion **separately** at every eval, because
averaging them lets a good reconstruction hide a failed completion — which is the one question
this architecture exists to answer. It also trains on random non-empty observed subsets after a
short full-observation warmup; a PoE shown holes from step 0 has no reconstruction to fall back
on.

`flow.train` supports `--arch jit`, which requires an external model repository that is not
bundled here; the default `--arch unet` is self-contained.

## Baselines

Any comparison should carry all three:

```
copy      emit the observed phase unchanged
copy_ae   the same answer through encode+decode, so it pays the autoencoder round-trip
          cost that the flow also pays
ae_orc    the true target encoded and decoded -- the ceiling any flow can reach
```

`ae_orc` is a property of the autoencoder, not a constant. Re-measure it whenever the
autoencoder changes.

## Layout

```
autoencoder/  slot encoder-decoder and PoE autoencoder, training, latent extraction
flow/         conditional flow matching -- slot_fm over complete latent fields,
              model/train over PoE private codes -- and image-space evaluation
contrastx_dataloader/  cohort definitions and the preprocessed-volume dataset
tools/        autoencoder scoring, ROI scoring, latent diagnostics, spectral analysis
```

## Citation

If you use this code, please cite the accompanying paper.
