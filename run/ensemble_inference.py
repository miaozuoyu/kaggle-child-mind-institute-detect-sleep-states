import pdb
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from pytorch_lightning import seed_everything

import sys
sys.path.append("/mnt/task_runtime")
from src.conf import EnsembleConfig

from src.utils.common import nearest_valid_size, trace
from src.utils.post_process import post_process_for_seg

import joblib

def make_submission(
    keys: list[str], preds: np.ndarray, score_th, distance
) -> pl.DataFrame:
    sub_df = post_process_for_seg(
        keys,
        preds,  # type: ignore
        score_th=score_th,
        distance=distance,  # type: ignore
    )

    return sub_df

def load_preds(path_dir):
    key_names = path_dir.rglob("keys_*.joblib")
    pred_names = path_dir.rglob("preds_*.joblib")
    # pdb.set_trace()

    keys = [joblib.load(key_name) for key_name in key_names]
    all_preds = [joblib.load(pred_name) for pred_name in pred_names]
    stacked_arrays = np.stack(all_preds, axis=0)
    preds = np.mean(stacked_arrays, axis=0)

    return keys[0], preds

@hydra.main(config_path="conf", config_name="ensemble", version_base="1.2")
def main(cfg: EnsembleConfig):
    seed_everything(cfg.seed)

    keys, preds = load_preds(Path(cfg.dir.output_dir))

    with trace("make submission"):
        sub_df = make_submission(
            keys,
            preds,
            score_th=cfg.pp.score_th,
            distance=cfg.pp.distance,
        )
    sub_df.write_csv(Path(cfg.dir.sub_dir) / "submission.csv")


if __name__ == "__main__":
    main()
