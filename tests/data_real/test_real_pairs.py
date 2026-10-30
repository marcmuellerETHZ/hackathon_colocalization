"""Toy tests for the pair builders (src/data/real/pairs.py): every selection rule has a positive and a
negative control with a known answer."""
import numpy as np
import pandas as pd
import pytest

from data.real import pairs, sources


def _genes():
    return pd.DataFrame({
        "gene_id": ["G1", "G2", "G3", "G4"], "gene_name": ["g1", "g2", "g3", "g4"], "chr": ["chr1"] * 4,
        "tss": [1_000_000, 3_000_000, 5_000_000, 7_000_000], "split": ["train", "valid", "test", "train"],
        "block_id": ["b1", "b2", "b3", "b4"],
        "win_lo": [500_001, 2_500_001, 4_500_001, 6_800_001], "win_hi": [1_500_000, 3_500_000, 5_500_000, 7_200_000]})


def _cs_rows(gene, cs, variants, pips, p, min_r2=0.9):
    return [{"gene_id": gene, "cs_id": f"{gene}_{cs}", "variant": v, "chr": "chr1", "pos": int(v.split("_")[1]),
             "pip": pp, "pvalue": p, "z": 6.0, "cs_size": len(variants), "cs_min_r2": min_r2}
            for v, pp in zip(variants, pips)]


def test_eqtl_gene_gate(monkeypatch):
    rows = []
    rows += _cs_rows("G1", "L1", ["chr1_1000000_A_G", "chr1_1000100_C_T"], [0.7, 0.3], 1e-9)       # gate pass
    rows += _cs_rows("G2", "L1", ["chr1_2000000_A_G"], [1.0], 1e-9)                               # lead outside window
    rows += _cs_rows("G3", "L1", ["chr1_5000000_A_G"], [1.0], 1e-4)                               # p >= 1e-5
    rows += _cs_rows("G4", "L1", ["chr1_7000000_A_G", "chr1_7100000_A_G"], [0.5, 0.5], 1e-9, 0.1)  # impure
    rows += _cs_rows("G4", "L2", ["chr1_7050000_A_G"], [1.0], 1e-12)                              # pure, pass
    monkeypatch.setattr(sources, "eqtl_cs_variants", lambda ds: pd.DataFrame(rows))
    g = pairs.eqtl_gene_cs("QTD000356", _genes()).set_index("gene_id")
    assert g.loc["G1", "gate"] and not g.loc["G2", "gate"] and not g.loc["G3", "gate"] and g.loc["G4", "gate"]
    assert g.loc["G4", "n_cs"] == 2 and g.loc["G4", "n_cs_pure"] == 1
    assert g.loc["G4", "top_lead_variant"] == "chr1_7050000_A_G"
    assert g.loc["G1", "top_lead_variant"] == "chr1_1000000_A_G"


def _gate(ds, genes_pass, leads=None):
    g = _genes().copy()
    g["gate"] = g["gene_id"].isin(genes_pass)
    g["n_cs"] = 1
    g["top_lead_variant"] = [(leads or {}).get(x, f"chr1_{t}_A_G") for x, t in zip(g["gene_id"], g["tss"])]
    g["top_lead_snv"] = g["top_lead_variant"]
    g["top_snv_by_pip"] = g["top_lead_variant"]
    g["top_lead_p"] = 1e-10
    g["top_lead_z"] = 7.0
    g["top_members"] = g["top_lead_variant"]
    g["top_n_members"] = 1
    g["dataset"] = ds
    return g


def test_build_r2_gene_level_and_no_same_donor():
    gates = {d: _gate(d, []) for d in ["QTD000356", "QTD000266", "QTD000373", "QTD000549", "QTD000021", "QTD000409",
                                        "QTD000609", "QTD000110", "QTD000539"]}
    gates["QTD000356"] = _gate("QTD000356", ["G1", "G2"])
    gates["QTD000373"] = _gate("QTD000373", ["G1", "G2"], leads={"G2": "chr1_3100000_C_T"})  # different lead
    gates["QTD000549"] = _gate("QTD000549", ["G1"])
    gates["QTD000539"] = _gate("QTD000539", ["G1"])   # TwinsUK LCL: other R2 group, same donors as 549
    r2 = pairs.build_r2(gates)
    ids = set(r2["locus_id"])
    assert ids == {"R2__QTD000356__QTD000373__G1", "R2__QTD000356__QTD000373__G2",
                   "R2__QTD000356__QTD000549__G1", "R2__QTD000373__QTD000549__G1"}
    # lead disagreement does NOT remove a pair (selection is gene-level)
    row = r2.set_index("locus_id").loc["R2__QTD000356__QTD000373__G2"]
    assert not row["same_top_lead"] and row["split"] == "valid"
    assert not any("QTD000539" in i for i in ids)  # LCL group has only one passing cohort here


def test_r3b_prefilter_rules(monkeypatch):
    genes = _genes()
    # eQTL: G1 one CS (members 1.00M, 1.0001M), G2 one CS, G4 two CSs (excluded)
    gate = _gate("QTD000266", ["G1", "G2", "G4"])
    gate.loc[gate.gene_id == "G1", ["top_members", "top_lead_variant", "top_lead_snv", "top_snv_by_pip"]] = \
        ["chr1_1000000_A_G;chr1_1000100_C_T", "chr1_1000000_A_G", "chr1_1000000_A_G", "chr1_1000000_A_G;chr1_1000100_C_T"]
    gate.loc[gate.gene_id == "G4", "n_cs"] = 2
    gates = {"QTD000266": gate}
    fg = pd.DataFrame([
        # G1 window: one pure FG CS, disjoint, p < 5e-8 -> R3b candidate
        {"region": "r1", "cs": 1, "lead_variant": "chr1_1200000_G_T", "lead_snv": "chr1_1200000_G_T", "lead_pos": 1_200_000,
         "lead_p": 1e-20, "lead_z": 9.3, "members": "chr1_1200000_G_T", "n_members": 1, "pure": True, "chr": "chr1"},
        # G2 window: FG CS sharing the eQTL lead variant (allele-swapped id) -> not disjoint -> excluded
        {"region": "r2", "cs": 1, "lead_variant": "chr1_3000000_G_A", "lead_snv": "chr1_3000000_G_A", "lead_pos": 3_000_000,
         "lead_p": 1e-20, "lead_z": 9.3, "members": "chr1_3000000_G_A", "n_members": 1, "pure": True, "chr": "chr1"},
        # G4 window: would pass on the GWAS side, but G4 has 2 eQTL CSs
        {"region": "r4", "cs": 1, "lead_variant": "chr1_6900000_G_T", "lead_snv": "chr1_6900000_G_T", "lead_pos": 6_900_000,
         "lead_p": 1e-20, "lead_z": 9.3, "members": "chr1_6900000_G_T", "n_members": 1, "pure": True, "chr": "chr1"},
    ])
    fg["snv_by_pip"] = fg["members"]
    mem = pd.DataFrame([{"region": r.region, "cs": r.cs, "variant": r.members, "chr": "chr1", "pos": r.lead_pos,
                         "pip": 1.0, "pvalue": r.lead_p, "z": r.lead_z} for r in fg.itertuples()])
    monkeypatch.setattr(pairs, "finngen_traits", lambda: {"TOY": object()})
    monkeypatch.setattr(pairs, "finngen_cs_table", lambda p: fg.copy())
    monkeypatch.setattr(sources, "finngen_cs_variants", lambda t: mem.copy())
    pre = pairs.r3b_prefilter(gates, genes)
    assert pre["locus_id"].tolist() == ["R3b__FG_TOY__QTD000266__G1"]

    # second FG CS in the G1 window -> "exactly one" violated -> excluded
    fg2 = pd.concat([fg, fg.iloc[[0]].assign(region="r1b", lead_pos=1_300_000, lead_variant="chr1_1300000_A_C",
                                             lead_snv="chr1_1300000_A_C", members="chr1_1300000_A_C")])
    mem2 = pd.concat([mem, mem.iloc[[0]].assign(region="r1b", pos=1_300_000, variant="chr1_1300000_A_C")])
    monkeypatch.setattr(pairs, "finngen_cs_table", lambda p: fg2.copy())
    monkeypatch.setattr(sources, "finngen_cs_variants", lambda t: mem2.copy())
    assert pairs.r3b_prefilter(gates, genes).empty

    # low-purity or weak (p >= 5e-8) FinnGen CS -> excluded
    for mod in ({"pure": False}, {"lead_p": 1e-6}):
        fg3 = fg.copy()
        for k, v in mod.items():
            fg3.loc[0, k] = v
        monkeypatch.setattr(pairs, "finngen_cs_table", lambda p, f=fg3: f.copy())
        monkeypatch.setattr(sources, "finngen_cs_variants", lambda t: mem.copy())
        assert pairs.r3b_prefilter(gates, genes).empty


def test_r3b_ld_filter_thresholds():
    pre = pd.DataFrame({"chr": ["chr1"] * 4, "cs1_snv_by_pip": ["a", "b", "c", "d"], "cs2_snv_by_pip": ["w", "x", "y", "z"],
                        "cs1_members": ["a"] * 4, "cs2_members": ["w"] * 4})
    poolb = {"a": 0.05, "b": 0.2, "c": np.nan, "d": 0.01}
    fgld = {"a": (0.0, "below_0.01"), "d": (0.3, "found")}
    out = pairs.r3b_ld_filter(pre, workers=0, poolb_fn=lambda x: (poolb[x[1]], x[1], x[2]), csmax_fn=lambda x: 0.0,
                              finngen_fn=lambda prs: [fgld[a] for a, _ in prs])
    # a: both low -> pass; b: pool-B 0.2 -> fail; c: not computable -> fail; d: FinnGen 0.3 -> fail
    assert out["pass_ld"].tolist() == [True, False, False, False]
    assert out["finngen_ld_status"].tolist() == ["below_0.01", "not_queried", "not_queried", "found"]


def test_write_arm_test_ids_only(tmp_path, monkeypatch):
    monkeypatch.setattr(pairs, "PAIRS", tmp_path)
    df = pd.DataFrame({"locus_id": ["x", "y", "z"], "arm": "R2", "dataset1": "d1", "dataset2": "d2",
                       "gene_id": ["G1", "G2", "G3"], "split": ["train", "valid", "test"], "chr": "chr1",
                       "win_lo": 1, "win_hi": 2, "lead1_p": [1e-9, 1e-9, 1e-9]})
    counts = pairs.write_arm(df, "R2")
    assert counts == {"train": 1, "valid": 1, "test": 1}
    tv = pd.read_csv(tmp_path / "R2_candidates.tsv", sep="\t")
    te = pd.read_csv(tmp_path / "R2_test_ids.tsv", sep="\t")
    assert set(tv["split"]) == {"train", "valid"} and te["locus_id"].tolist() == ["z"]
    assert "lead1_p" not in te.columns  # identifiers only


def test_z_from_p_roundtrip():
    from scipy import stats
    z = np.array([0.5, 3.0, 8.0, 30.0])
    assert np.allclose(pairs.z_from_p(2 * stats.norm.sf(z)), z, rtol=1e-6)
