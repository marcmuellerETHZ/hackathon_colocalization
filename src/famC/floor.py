"""Floor for Family C (DESIGN v4 §5 C "Floor"): a gradient-boosted classifier (LightGBM) on hand-built features.

Features per locus (all from the same records the network sees):
  * leads: |z| of each trait's lead, |z| of the other trait at each lead, r(lead1, lead2) (allele-invariant sign)
    and r², lead1 == lead2;
  * conditional z: max |z_c| per trait over the screen (lead-removed), |z_c| of each trait at the other's lead;
  * coloc.abf PP0-PP4 (W = 0.15, p1 = p2 = 1e-4, p12 = 1e-5) over the full window, and log-sum ABFs;
  * SuSiE summaries: famG GPU SuSiE-RSS (susieR 0.14.2 semantics, L = 10) on the screened variants with their
    inference LD, per trait: number of credible sets, max PIP; coloc.bf_bf over CS pairs: max PP4, PP3 at that
    pair (coloc.abf fallback when a trait has no CS);
  * context: rho-hat, its SE, log10 N1, N2, n_ref, p, trait types, counts of |z| > 5.45, LD effective rank and
    top-4 eigenvalues of the screen LD.
Training: multiclass (S, D, W, G, U) with the network's class weights; early stopping on a 10% split of the
TRAINING records; predictions are corrected to the natural mixture like the network's.

    python -m famC.floor --run floor_v1 --train-set train_v1 --valid-set valid_v1 --n-train 200000
    python -m famC.floor featurize --set train_v1 --worker k --n-workers K --device cpu   # parallel feature cache

Feature caches: data/processed/famC/floor_feats/<set>/<shard>.floor{_nosusie}.npy (outside the record set's
directory, so a registered record set is never modified). Features are deterministic per record; the device
(cpu / cuda) only changes floating-point rounding.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from famC.c0world import coloc_abf_torch
from famC.simadapter import collate_records
from famC.train_main import CLASS_W, MI, OUT, load_all, metrics_from_probs

FEATS = ["lead_z1", "lead_z2", "z2_at_lead1", "z1_at_lead2", "r12", "r12_sq", "lead_same", "max_zc1", "max_zc2",
         "zc2_at_lead1", "zc1_at_lead2", "PP0", "PP1", "PP2", "PP3", "PP4", "logPP4_PP3",
         "ncs1", "ncs2", "maxpip1", "maxpip2", "susie_PP4", "susie_PP3", "susie_fallback",
         "rho_hat", "rho_se", "log_n1", "log_n2", "log_nref", "log_p", "type1", "type2", "nsig1", "nsig2",
         "erank", "eig1", "eig2", "eig3", "eig4"]


@torch.no_grad()
def features_batch(recs: list, susie: bool = True, device="cuda") -> np.ndarray:
    device = torch.device(device)
    r = collate_records(recs, device)
    tr, gr, ar_ = r["tok_raw"].double(), r["glob_raw"].double(), r["all_raw"].double()
    B, M, _ = tr.shape
    isl1, isl2 = tr[..., 11] > 0.5, tr[..., 12] > 0.5
    pick = lambda col, m: (tr[..., col] * m).sum(-1)
    z2_l1 = pick(1, isl1).abs()
    z1_l2 = pick(0, isl2).abs()
    zc2_l1 = pick(3, isl1).abs()
    zc1_l2 = pick(2, isl2).abs()
    r12 = gr[:, 13] * gr[:, 14] * gr[:, 15]
    mzc1 = tr[..., 2].abs().amax(-1)
    mzc2 = tr[..., 3].abs().amax(-1)
    mask = r["mask"]
    se1, se2 = torch.exp(ar_[..., 4]), torch.exp(ar_[..., 5])
    se1 = torch.where(mask, se1, torch.ones_like(se1))
    se2 = torch.where(mask, se2, torch.ones_like(se2))
    pp, _ = coloc_abf_torch(ar_[..., 0], se1, ar_[..., 1], se2, mask)
    lr43 = torch.log(pp[:, 4].clamp_min(1e-300)) - torch.log(pp[:, 3].clamp_min(1e-300))
    sus = np.full((B, 7), np.nan)
    if susie:
        sus = susie_features(tr, r["Rmm"], r["valid"], gr, pp, device)
    F = torch.stack([gr[:, 8], gr[:, 9], z2_l1, z1_l2, r12, r12 ** 2, gr[:, 16], mzc1, mzc2, zc2_l1, zc1_l2,
                     pp[:, 0], pp[:, 1], pp[:, 2], pp[:, 3], pp[:, 4], lr43.clamp(-50, 50)], -1).cpu().numpy()
    G = torch.stack([gr[:, 0], gr[:, 1], gr[:, 2] * 5, gr[:, 3] * 5, gr[:, 5] * 5, gr[:, 4] * 4, gr[:, 6], gr[:, 7],
                     gr[:, 10], gr[:, 11], gr[:, 12], gr[:, 18], gr[:, 19], gr[:, 20], gr[:, 21]], -1).cpu().numpy()
    return np.concatenate([F, sus, G], 1)


def susie_features(tr, Rmm, valid, gr, pp_abf, device=None) -> np.ndarray:
    from famG.coloc import bf_bf_batch
    from famG.susie_rss import susie_rss_batch
    B = tr.shape[0]
    n1, n2 = 10 ** (gr[:, 2] * 5), 10 ** (gr[:, 3] * 5)
    zs, lds, ns = [], [], []
    for t, n in ((0, n1), (1, n2)):
        for b in range(B):
            k = int(valid[b].sum())
            zs.append(tr[b, :k, t].cpu().numpy())
            lds.append(Rmm[b, :k, :k].double().cpu().numpy())
            ns.append(float(n[b]))
    fits = susie_rss_batch(zs, lds, ns, "dense", device=device)
    f1, f2 = fits[:B], fits[B:]
    get = lambda f: f["lbf_variable"][f["sets"]["cs_index"]] if f["sets"]["cs_index"] else np.zeros((0, len(f["pip"])))
    pps = bf_bf_batch([get(f) for f in f1], [get(f) for f in f2], 1e-4, 1e-4, 5e-6, device=device)
    out = np.zeros((B, 7))
    pa = pp_abf.cpu().numpy()
    for b in range(B):
        a, c = f1[b], f2[b]
        out[b, 0], out[b, 1] = len(a["sets"]["cs"]), len(c["sets"]["cs"])
        out[b, 2], out[b, 3] = float(np.max(a["pip"])), float(np.max(c["pip"]))
        if pps[b].size:
            P = pps[b].reshape(-1, 5)
            i = int(np.argmax(P[:, 4]))
            out[b, 4], out[b, 5], out[b, 6] = P[i, 4], P[i, 3], 0
        else:
            out[b, 4], out[b, 5], out[b, 6] = pa[b, 4], pa[b, 3], 1
    return out


def featurize_set(recs, susie=True, bs=512, device="cuda"):
    out = []
    for i in range(0, len(recs), bs):
        out.append(features_batch(recs[i:i + bs], susie=susie, device=device))
    return np.concatenate(out)


def cache_path(shard: Path, susie: bool) -> Path:
    from famC.simadapter import DATA_DIR
    d = DATA_DIR / "floor_feats" / shard.parent.name
    d.mkdir(parents=True, exist_ok=True)
    return d / (shard.stem + (".floor.npy" if susie else ".floor_nosusie.npy"))


def featurize_shards(name: str, max_loci: int | None, susie=True, max_shards: int | None = None, device="cuda"):
    """Features + meta of the first shards (sorted, as train_main's --max-train-shards), cached per shard."""
    from famC.train_main import list_shards
    Xs, Ms, n = [], [], 0
    shards = list_shards(name)
    if max_shards:
        shards = shards[:max_shards]
    for f in shards:
        d = torch.load(f, weights_only=False)
        cache = cache_path(f, susie)
        if cache.exists():
            X = np.load(cache)
        else:
            X = featurize_set(d["recs"], susie, device=device)
            np.save(cache, X)
        assert len(X) == len(d["recs"]), (f, len(X))
        Xs.append(X)
        Ms.append(d["meta"])
        n += len(X)
        if max_loci and n >= max_loci:
            break
    X, M = np.concatenate(Xs), torch.cat(Ms)
    if max_loci:
        X, M = X[:max_loci], M[:max_loci]
    return X, M


def featurize_main(argv):
    """Fill the feature cache for shards worker::n_workers of a set (parallel CPU workers)."""
    from famC.train_main import list_shards
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--no-susie", action="store_true")
    ap.add_argument("--skip-newest", action="store_true",
                    help="skip the highest-numbered shard of each generator tag (may still be written)")
    a = ap.parse_args(argv)
    if a.device == "cpu":
        torch.set_num_threads(1)
    shards = list_shards(a.set)
    if a.max_shards:
        shards = shards[:a.max_shards]
    if a.skip_newest:
        newest = {}
        for f in shards:
            newest[f.stem.split("_")[0]] = f
        shards = [f for f in shards if f not in set(newest.values())]
    t0, n = time.time(), 0
    for f in shards[a.worker::a.n_workers]:
        cache = cache_path(f, not a.no_susie)
        if cache.exists():
            continue
        d = torch.load(f, weights_only=False)
        X = featurize_set(d["recs"], not a.no_susie, bs=a.bs, device=a.device)
        tmp = cache.with_suffix(".tmp.npy")
        np.save(tmp, X)
        tmp.rename(cache)
        n += len(X)
        print(f"[w{a.worker}] {f.name} {len(X)} -> {cache.name}; {n / (time.time() - t0):.2f} loci/s", flush=True)
    print("FEATURIZE DONE", n, flush=True)


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "featurize":
        return featurize_main(sys.argv[2:])
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--train-set", default="train_v1")
    ap.add_argument("--valid-set", default="valid_v1")
    ap.add_argument("--n-train", type=int, default=200000)
    ap.add_argument("--n-valid", type=int, default=30000)
    ap.add_argument("--no-susie", action="store_true")
    ap.add_argument("--max-train-shards", type=int, default=None, help="first N shards, as train_main")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    out_dir = OUT / a.run
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    Xt, tmeta = featurize_shards(a.train_set, None if a.max_train_shards else a.n_train, not a.no_susie,
                                 a.max_train_shards, a.device)
    t_feat = time.time() - t0
    Xv, vmeta = featurize_shards(a.valid_set, a.n_valid, not a.no_susie, device=a.device)
    yt = tmeta[:, MI["y"]].long().numpy()
    w = CLASS_W.numpy()[yt]
    g = np.random.default_rng(20261030)
    es = g.random(len(yt)) < 0.1
    ds = lgb.Dataset(Xt[~es], yt[~es], weight=w[~es], feature_name=FEATS)
    de = lgb.Dataset(Xt[es], yt[es], weight=w[es], reference=ds)
    params = dict(objective="multiclass", num_class=5, learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, seed=20261030, verbose=-1,
                  num_threads=a.threads)
    bst = lgb.train(params, ds, num_boost_round=3000, valid_sets=[de],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    t1 = time.time()
    Pv = bst.predict(Xv, num_iteration=bst.best_iteration)
    Pv = Pv / CLASS_W.numpy()[None]
    Pv = Pv / Pv.sum(1, keepdims=True)
    t_pred = time.time() - t1
    res = {"valid": metrics_from_probs(Pv, vmeta), "n_train": int(len(yt)), "best_iter": bst.best_iteration,
           "feat_s_per_locus_incl_cache": t_feat / len(yt), "predict_s_per_locus": t_pred / len(Pv),
           "importance": dict(zip(FEATS, bst.feature_importance("gain").round(1).tolist())), "susie": not a.no_susie}
    res.update({"train_set": a.train_set, "max_train_shards": a.max_train_shards, "valid_set": a.valid_set})
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1))
    (out_dir / "config.json").write_text(json.dumps(vars(a) | {"feats": FEATS, "params": params,
                                                               "class_w": CLASS_W.tolist()}, indent=1))
    np.save(out_dir / "valid_probs.npy", Pv)
    bst.save_model(str(out_dir / "model.txt"))
    v = res["valid"]
    print("FLOOR", json.dumps({k: {kk: v[k].get(kk) for kk in ("auprc_sd_pm", "auroc_dw", "logloss_nat")} for k in v}),
          flush=True)


if __name__ == "__main__":
    main()
