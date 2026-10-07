"""Download the trained AEV-PLIG models used by DockingScoring.ipynb.

The ensemble (model_GATv2Net_ligsim90_fep_benchmark, 10 models, ~303 MB plus
its scaler) comes from the official AEV-PLIG repository
(https://github.com/oxpig/AEV-PLIG, BSD 3-Clause licence), pinned to a
specific commit. Files are saved to aev_plig/output/trained_models/.

    python aev_plig/download_models.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "3ccac4b84d6fc1fb372cb623e705837fbf280f15"
BASE = f"https://raw.githubusercontent.com/oxpig/AEV-PLIG/{COMMIT}/output/trained_models/"
MODEL = "model_GATv2Net_ligsim90_fep_benchmark"
FILES = [f"{MODEL}.pickle"] + [f"{MODEL}_{i}.model" for i in range(10)]
GIT_BLOB_SHA = {
    "model_GATv2Net_ligsim90_fep_benchmark.pickle": "67e5ff03718d3a4c77c48fd1afd5fe4f8ad50b1a",
    "model_GATv2Net_ligsim90_fep_benchmark_0.model": "7ee1da5a0511c07e66d41fb70196445fb9b6a2b5",
    "model_GATv2Net_ligsim90_fep_benchmark_1.model": "a7b16bdf969f20b4596c6a725b88e02e984a1461",
    "model_GATv2Net_ligsim90_fep_benchmark_2.model": "e804719e82b50f36e9e2caea8059147a75dec3ef",
    "model_GATv2Net_ligsim90_fep_benchmark_3.model": "ff0382af7eb348519112aa461dea2e085735e70f",
    "model_GATv2Net_ligsim90_fep_benchmark_4.model": "12e74542a748db8eeeeb085787061bea6aefdf47",
    "model_GATv2Net_ligsim90_fep_benchmark_5.model": "f4b1a118fcc2dd839fcfbd3c804de3f46dbc4439",
    "model_GATv2Net_ligsim90_fep_benchmark_6.model": "38e7bc1a50416ad12ca44a19815a0e15732e6059",
    "model_GATv2Net_ligsim90_fep_benchmark_7.model": "fd631a6800e207c7a867253d34610d82faa17aa5",
    "model_GATv2Net_ligsim90_fep_benchmark_8.model": "827e2c0ecce09d3e5ea77d7975580dc6cdd01cc2",
    "model_GATv2Net_ligsim90_fep_benchmark_9.model": "e1ab61c455c64e702d5ce8178fffaffe225627f7",
}
DEST = Path(__file__).resolve().parent / "output" / "trained_models"


def git_blob_sha(path):
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DEST / name
        if target.exists() and git_blob_sha(target) == GIT_BLOB_SHA[name]:
            print(f"already present: {name}")
            continue
        print(f"downloading {name} ...", flush=True)
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(BASE + name, tmp)
        if git_blob_sha(tmp) != GIT_BLOB_SHA[name]:
            tmp.unlink()
            sys.exit(f"checksum mismatch for {name} - download failed or file changed upstream")
        tmp.rename(target)
    print(f"Models in {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
