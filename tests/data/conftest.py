"""Shared toy fixture: a tiny phased VCF -> pgen (plink2) with known QC outcome and allele frequencies."""
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PLINK2 = shutil.which("plink2") or str(ROOT / "env/tools/bin/plink2")

SAMPLES = ["S1", "S2", "S3", "S4", "S5", "S6"]
POOLS = {"A": ["S1", "S2"], "B": ["S3", "S4"], "FIN": ["S5"], "ASW_ACB": ["S6"], "L6c": ["S1"]}
# (pos, ref, alt, GTs for S1..S6, expected_keep)
TOY = [
    (1000, "A", "G", ["0|1", "1|1", "0|0", "1|0", "0|0", "1|1"], True),
    (2000, "C", "CT", ["0|1", "0|0", "0|0", "0|0", "0|0", "0|0"], False),   # indel
    (3000, "G", "A", ["0|1", "0|0", "0|0", "0|0", "0|0", "0|0"], False),    # 2 records at 3000
    (3000, "G", "T", ["0|0", "0|0", "1|0", "0|0", "0|0", "0|0"], False),
    (4000, "T", "C", ["0|1", "1|0", "0|0", "0|0", "0|0", "0|0"], False),    # multi-record in raw VCF
    (5000, "AC", "GT", ["0|1", "0|0", "0|0", "0|0", "0|0", "0|0"], False),  # MNP
    (6000, "C", "T", ["1|0", "0|1", "1|1", "0|0", "1|0", "0|1"], True),
    (7000, "G", "C", ["0|0", "0|1", "1|0", "1|1", "0|1", "1|0"], True),
    (28_499_999, "A", "C", ["0|1", "0|0", "0|0", "0|0", "0|0", "0|0"], True),   # just before MHC
    (28_500_000, "A", "C", ["0|1", "0|0", "0|0", "0|0", "0|0", "0|0"], False),  # MHC start (inclusive)
    (31_000_000, "A", "C", ["0|1", "0|0", "0|0", "0|0", "0|0", "0|0"], False),  # MHC
    (33_400_000, "A", "C", ["0|1", "0|0", "0|0", "0|0", "0|0", "0|0"], False),  # MHC end (inclusive)
    (33_400_001, "A", "C", ["1|1", "0|1", "0|0", "0|0", "0|0", "0|0"], True),
]


@pytest.fixture(scope="session")
def toy_pgen(tmp_path_factory):
    d = tmp_path_factory.mktemp("toy")
    vcf = d / "toy.vcf"
    with open(vcf, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=chr6,length=170805979>\n")
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(SAMPLES) + "\n")
        for pos, ref, alt, gts, _ in TOY:
            fh.write(f"chr6\t{pos}\tchr6:{pos}:{ref}:{alt}\t{ref}\t{alt}\t.\tPASS\t.\tGT\t" + "\t".join(gts) + "\n")
    pg = d / "pgen"
    pg.mkdir()
    subprocess.run([PLINK2, "--vcf", str(vcf), "--make-pgen", "--output-chr", "chrM",
                    "--out", str(pg / "chr6"), "--threads", "1", "--memory", "1000"],
                   check=True, capture_output=True)
    mr = d / "multirec"
    mr.mkdir()
    (mr / "chr6.pos").write_text("3000\t2\n4000\t2\n")
    pools = d / "pools.tsv"
    pd.DataFrame([(s, "X", "X", p) for p, ss in POOLS.items() for s in ss],
                 columns=["sample_id", "pop", "superpop", "pool"]).to_csv(pools, sep="\t", index=False)
    out = d / "variants"
    return {"dir": d, "pgen_dir": pg, "multirec_dir": mr, "pools": pools, "out_dir": out}


def pytest_configure(config):
    """Keep pytest temp dirs under /tmp/claude-<uid>/ (project rule: no temp files elsewhere)."""
    import os
    if not config.option.basetemp:
        config.option.basetemp = f"/tmp/claude-{os.getuid()}/pytest-data-{os.getpid()}"
