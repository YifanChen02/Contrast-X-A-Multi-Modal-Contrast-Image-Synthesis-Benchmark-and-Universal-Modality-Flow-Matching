"""Image-space eval for the slot-FM: decode its latent prediction back to images and score.

The decode chain must invert every transform the FM trained under, in order:

    FM output (asinh space)  -> sinh (undo asinh)  -> / scale (undo the global scale)
    -> MAISI decode          -> image

For each test case and each paper task, the observed modalities are encoded to slots, scaled and
asinh'd; the FM inpaints the missing slots; the chain above decodes them; PSNR/SSIM are taken per
(task, modality) against the prep128 target.
"""
import argparse, itertools, json, os, sys
import os as _os
RESULTS = _os.environ.get("FLOWMI_RESULTS", "results")
import numpy as np, torch, torch.nn.functional as F
import nibabel as nib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contrastx_dataloader import COHORTS
from contrastx_dataloader.prep_dataset import PREP_ROOT, PrepSet
from autoencoder.maisi_slot import MaisiSlotAE
from flow.slot_fm import SlotFlow, sample, tailclip_fwd, tailclip_inv, flow_start

TASKS = {"breast_dce": [("DCE1_to_DCE2",(1,0,0),["dce2"]),
                        ("DCE1and3_to_DCE2",(1,0,1),["dce2"]),
                        ("DCE1_to_DCE2and3",(1,0,0),["dce2","dce3"])],
         "ct": [("CT_to_CTC",(1,0),["ctc"])]}

def psnr_ssim(pred, tg):
    from monai.metrics import SSIMMetric
    lo=tg.min(); rng=(tg.max()-lo).clamp(min=1e-6)
    t=(tg-lo)/rng; p=((pred-lo)/rng).clamp(0,1)
    return (float(-10*torch.log10(F.mse_loss(p,t).clamp(min=1e-12))),
            float(SSIMMetric(spatial_dims=3,data_range=1.0)(p,t).mean()))


# ---------------------------------------------------------------- visualisation
ERR_GAIN = 4.0          # |pred-target| is small; without gain the error panel reads as black

def _win(sl, lo, hi):
    """One 2D slice -> uint8, windowed on [lo,hi] and rotated to a radiological view."""
    a = (np.asarray(sl, dtype=np.float32) - lo) / max(float(hi - lo), 1e-6)
    return (np.clip(np.rot90(a), 0, 1) * 255).astype(np.uint8)

def _grid(cells, col_labels, title):
    from PIL import Image, ImageDraw
    h, w = cells[0][0].shape
    pad, t_h, l_h = 2, 13, 12
    top = t_h + l_h
    canvas = np.zeros((top + len(cells)*(h+pad), len(cells[0])*(w+pad)), np.uint8)
    for r, row in enumerate(cells):
        for c, im in enumerate(row):
            canvas[top+r*(h+pad):top+r*(h+pad)+h, c*(w+pad):c*(w+pad)+w] = im
    img = Image.fromarray(canvas).convert("RGB")
    d = ImageDraw.Draw(img)
    d.text((2, 1), title, fill=(255, 210, 0))
    for c, lab in enumerate(col_labels):
        d.text((c*(w+pad)+2, t_h), lab, fill=(0, 230, 230))
    return img

def viz_case(vdir, cid, tname, pat, keys, k, m, x, dec, z0, ps, ss, save_nii,
             case=None, prep_root=None, cohort=None, orc=None):
    """One PNG per (case, task, predicted modality): conditioning | target | pred | |error|.

    Rows are three depth positions so a single lucky slice cannot flatter the result.
    Target and prediction share ONE window (taken from the target) -- windowing them
    independently would hide a global intensity shift, which is exactly the failure a
    latent-space model is prone to.
    """
    obs = [j for j, o in enumerate(pat) if o]
    tg = x[m, ..., :z0].float().cpu().numpy()
    pr = dec[0, m, ..., :z0].float().cpu().numpy()
    lo, hi = float(tg.min()), float(tg.max())
    rows, zs = [], [max(int(z0*f), 0) for f in (0.35, 0.50, 0.65)]
    for z in zs:
        row = [_win(x[j, ..., z].float().cpu().numpy(),
                    float(x[j, ..., :z0].min()), float(x[j, ..., :z0].max())) for j in obs]
        err = np.abs(pr[..., z] - tg[..., z]) * ERR_GAIN
        row += [_win(tg[..., z], lo, hi)]
        if orc is not None:
            row += [_win(orc[0, m, ..., :z0][..., z].float().cpu().numpy(), lo, hi)]
        row += [_win(pr[..., z], lo, hi), _win(err, 0.0, hi - lo)]
        rows.append(row)
    labels = ([f"in:{keys[j]}" for j in obs] + [f"target:{k}"]
              + ([f"AE-oracle:{k}"] if orc is not None else [])
              + [f"pred:{k}", f"|err|x{ERR_GAIN:g}"])
    img = _grid(rows, labels, f"case{cid:03d} {tname} -> {k}   PSNR {ps:.2f}  SSIM {ss*100:.1f}")
    os.makedirs(vdir, exist_ok=True)
    img.save(os.path.join(vdir, f"case{cid:03d}_{tname}_{k}.png"))
    if save_nii:
        # Take the affine from the prep source, never eye(4): these voxels are strongly
        # anisotropic (DCE 1.24x1.74x3.15 mm), so an identity affine renders at the wrong
        # aspect ratio in every viewer and makes any distance read off the volume wrong.
        aff, hdr = np.eye(4), None
        src = os.path.join(prep_root or "", cohort or "", "test", case or "",
                           f"{case}_{k}.nii.gz")
        if case and os.path.isfile(src):
            ref = nib.load(src); aff = ref.affine; hdr = ref.header
        else:
            print(f"  [viz] no prep source for {case} {k}; writing identity affine")
        stem = f"{case or ('case%03d' % cid)}_{tname}_{k}"
        nib.save(nib.Nifti1Image(pr.astype(np.float32), aff, hdr),
                 os.path.join(vdir, f"{stem}_pred.nii.gz"))
        nib.save(nib.Nifti1Image(tg.astype(np.float32), aff, hdr),
                 os.path.join(vdir, f"{stem}_target.nii.gz"))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cohort",required=True); ap.add_argument("--fm-ckpt",required=True)
    ap.add_argument("--prep-root",default=PREP_ROOT); ap.add_argument("--steps",type=int,default=20)
    ap.add_argument("--cases",type=int,default=0)
    ap.add_argument("--viz-n",type=int,default=4,help="render this many cases; 0 disables")
    ap.add_argument("--viz-dir",default="",help="default: results/viz_slotfm_<cohort>_<tag>")
    ap.add_argument("--viz-nifti",action="store_true",help="also dump pred/target as NIfTI")
    ap.add_argument("--nifti-n",type=int,default=-1,help="cases to dump as NIfTI; -1 = same as viz-n")
    ap.add_argument("--tag",default="",help="suffix for the result json and viz dir")
    ap.add_argument("--sample-k",type=int,default=1,
                    help="latent samples averaged before decoding; matches how validation scores")
    ap.add_argument("--dev-q",type=float,default=0.0,
                    help="keep the flow's departure from the base only above this quantile")
    ap.add_argument("--dev-soft",type=float,default=0.3)
    ap.add_argument("--cfg",type=float,default=1.0,
                    help="global classifier-free guidance; only the spatially adaptive "
                         "variant is dominated by the global one on both axes")
    ap.add_argument("--churn",type=float,default=0.0,
                    help="integrate the marginal-preserving SDE instead of the ODE; "
                         "0 is the old deterministic path, bit for bit")
    ap.add_argument("--noise-start",type=float,default=-1.0,
                    help="-1 = take it from the checkpoint (the correct default)")
    a=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    keys=COHORTS[a.cohort][1]; M=len(keys)
    fb=torch.load(a.fm_ckpt,map_location=dev,weights_only=False)
    Z=fb["Z"]; scale=fb["scale"]; asinh=fb.get("asinh",0.0); tclip=fb.get("tailclip",0.0)
    if a.noise_start<0: a.noise_start=float(fb.get("noise_start",0.0))
    basemode=fb.get("base","zero")
    print(f"  start base={basemode}  noise={a.noise_start}")
    net=SlotFlow(M,Z,gate=fb.get("gate_head",False)).to(dev).eval(); net.load_state_dict(fb["ema"])
    ae=MaisiSlotAE(latent_channels=Z,ckpt=None).to(dev).eval()
    ae.load_state_dict(torch.load(fb["ae_ckpt"],map_location=dev,weights_only=False)["model"])
    print(f"FM ep{fb['epoch']} scale {scale:.4f} asinh {asinh} tailclip {tclip} "
          f"valR2 {fb.get('val_r2')} k={a.sample_k} devq={a.dev_q} | AE {fb['ae_ckpt']}")

    # Must mirror SlotLatents exactly, and inv must undo them in the REVERSE order.
    def fwd(z):    # scale -> tailclip -> asinh, as training
        z=z*scale
        if tclip>0: z=tailclip_fwd(z,tclip)
        return asinh*torch.asinh(z/asinh) if asinh>0 else z
    def inv(z):    # undo asinh -> undo tailclip -> undo scale
        if asinh>0: z=asinh*torch.sinh(z/asinh)
        if tclip>0: z=tailclip_inv(z,tclip)
        return z/scale

    ds=PrepSet(a.prep_root,a.cohort,"test",z_window=128,train=False)
    idxs=range(len(ds)) if not a.cases else range(min(a.cases,len(ds)))
    sfx=(a.tag or os.path.basename(os.path.dirname(a.fm_ckpt)))
    vdir=a.viz_dir or f"{RESULTS}/viz_slotfm_{a.cohort}_{sfx}"
    cell={}
    for i in idxs:
        # The U-Net has 3 downsamples on a /4 latent, so the LATENT depth must be a multiple
        # of 8 -- i.e. the image depth a multiple of 32, not 16. At //16 a case with depth 36
        # gives a latent depth of 9 and the skip concat fails.
        it=ds[i]; x=it["x"]; x=x[...,:(x.shape[-1]//32*32)]; z0=min(int(it["z0"]),x.shape[-1])
        with torch.no_grad():
            mu,_=ae.encode(x.unsqueeze(0).to(dev))     # (1,M,Z,32,32,32) -- wait, spatial depends on z
            orc=ae.decode(mu)                          # AE round-trip = the ceiling
        for tname,pat,outs in TASKS[a.cohort]:
            mask=torch.tensor([pat],dtype=torch.float32,device=dev)
            zf=fwd(mu)                                  # scale + tailclip/asinh
            with torch.no_grad():
                gs=[]
                for _ in range(max(a.sample_k,1)):
                    st,cd,bfv,_=flow_start(zf,mask,noise=a.noise_start,base=basemode)
                    gs.append(sample(net,st,mask,M,Z,steps=a.steps,cond=cd,churn=a.churn,
                                 cfg=a.cfg,base=bfv))
                g=torch.stack(gs).mean(0)
                if a.dev_q>0:
                    delta=g-bfv
                    mag=delta.abs(); thr=mag.flatten()[::7].quantile(a.dev_q)
                    w=torch.sigmoid((mag-thr)/(a.dev_soft*thr.clamp(min=1e-6))) if a.dev_soft>0 \
                      else (mag>=thr).to(delta.dtype)
                    g=bfv+delta*w
                g=g.reshape(1,M,Z,*g.shape[2:])
                gz=inv(g)                                    # undo asinh+scale -> MAISI latent
                dec=ae.decode(gz)                            # (1,M,H,W,D)
            for k in outs:
                m=keys.index(k)
                pr=dec[:,m:m+1,...,:z0]; tg=x[m:m+1,...,:z0].unsqueeze(0).to(dev)
                ps,ss=psnr_ssim(pr,tg)
                cell.setdefault((tname,k),[]).append((ps,ss))
                # the bar any completion model must clear: emit the first observed modality
                o0=[j for j,o in enumerate(pat) if o][0]
                cp=x[o0:o0+1,...,:z0].unsqueeze(0).to(dev)
                cell.setdefault((tname,k+" [COPY]"),[]).append(psnr_ssim(cp,tg))
                # AE oracle: the true latent decoded. The ceiling for ANY flow on this AE --
                # if it sits below COPY on a metric, that metric is unreachable by construction
                # and the target has to move, not the model.
                cell.setdefault((tname,k+" [AE-ORC]"),[]).append(psnr_ssim(orc[:,m:m+1,...,:z0],tg))
                nn = a.viz_n if a.nifti_n < 0 else a.nifti_n
                if (a.viz_n and i<a.viz_n) or (a.viz_nifti and i<nn):
                    try:
                        viz_case(vdir,i,tname,pat,keys,k,m,x,dec,z0,ps,ss,
                                 a.viz_nifti and i<nn, case=it["case"],
                                 prep_root=a.prep_root, cohort=a.cohort, orc=orc)
                    except Exception as e:          # a plotting slip must never lose the metrics
                        print(f"  [viz] case{i} {tname} {k} failed: {e}")
    print(f"\n{'task':>18} {'modality':>9} {'PSNR':>8} {'SSIM(%)':>8} {'n':>4}")
    summ={}
    for kk in sorted(cell):
        v=cell[kk]; p=np.mean([q[0] for q in v]); s=np.mean([q[1] for q in v])*100
        print(f"{kk[0]:>18} {kk[1]:>9} {p:8.2f} {s:8.1f} {len(v):4d}")
        summ[f"{kk[0]}|{kk[1]}"]={"psnr":float(p),"ssim":float(s),"n":len(v)}
    jp=f"{RESULTS}/slotfm_{a.cohort}_{sfx}.json"
    json.dump({"fm_ckpt":a.fm_ckpt,"epoch":int(fb["epoch"]),
               "val_r2":fb.get("val_r2"),"asinh":asinh,"tailclip":tclip,
               "ae_ckpt":fb.get("ae_ckpt"),
               "metrics":summ},open(jp,"w"),indent=1)
    print(f"wrote {jp}")
    if a.viz_n: print(f"viz -> {vdir}")
    print("SLOTEVAL_OK")

if __name__=="__main__": main()
