import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.conf.train_schema import AppCfg as TrainCfg
from src.dataloader.datasets import BagDataset

try:
    from colorama import Fore, init  # type: ignore

    init(autoreset=True)
    reset = Fore.RESET
    green = Fore.GREEN
except:
    reset = ""
    green = ""


def plot_and_save_losses(
    pred_losses: List[int],
    score_losses: List[int],
    emb_losses: List[int],
    file_name: str,
):
    plt.figure()
    plt.plot(pred_losses, label="Prediction Loss")
    plt.plot(score_losses, label="Score similarity Loss")
    plt.plot(emb_losses, label="Embedding loss")

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(file_name)
    plt.close()


def plot_and_save_num_tile_distribution(
    train_dl: DataLoader, file_path: str | Path
) -> None:
    dataset: BagDataset = train_dl.dataset  # type: ignore

    size = []
    for slide in dataset.slides:
        for scan in slide.scans.values():
            size.append(scan.num_tiles)

    plt.figure()
    plt.hist(size)
    plt.title("Tile distribution training")
    plt.xlabel("Number of tiles")
    plt.savefig(Path(file_path) / "num_tile_distribution_training.png")
    plt.close()


def print_value_counts(value_counts: pd.Series, modifier: str) -> None:
    for label, count in value_counts.items():
        logger.info(f"{modifier}{label}: {count}")


def log_df_info(train_df: pd.DataFrame, config: TrainCfg) -> None:
    logger.info(
        f"Label distribution: {[f'{label}: {count}' for label, count in train_df[config.data.target_label].value_counts().items()]}"
    )

    if "pT" in train_df.columns:
        logger.info(
            f"T-stage distribution: {[f'{label}: {count}' for label, count in train_df['pT'].value_counts().items()]}"
        )

    if "pN" in train_df.columns:
        logger.info(
            f"N-stage distribution: {[f'{label}: {count}' for label, count in train_df['pN'].value_counts().items()]}"
        )

    if "stage" in train_df.columns:
        logger.info(
            f"Stage distribution: {[f'{label}: {count}' for label, count in train_df['stage'].value_counts().items()]}"
        )

    logger.info(
        f"Scanner distribution: {[f'{label}: {count}' for label, count in train_df['scanner'].value_counts().items()]}"
    )

    if "project" in train_df.columns:
        logger.info(
            f"Project distribution: {[f'{label}: {count}' for label, count in train_df['project'].value_counts().items()]}"
        )

    logger.info("Number of included .h5:", train_df.shape[0])


def get_category_to_index(series: pd.Series) -> Dict:
    return {category: i for i, category in enumerate(sorted(series.dropna().unique()))}


def get_cohort_df(slide_table: str, target_label: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Assume there is one 'unit' (thing to be classified) per row.
    If there are multiple scans per patient etc. this is handled by the script
    creating the input files.
    """
    df = pd.read_csv(slide_table, dtype=str)
    if df.empty:
        raise ValueError("Dataframe is empty")
    msg = ""
    if "slide" not in df.columns:
        msg += "No 'slide' column found in slide table.\n"
    if not "scanner" in df.columns:
        msg += "No 'scanner' column found in slide table.\n"
    if not target_label in df.columns:
        msg += f"No '{target_label}' column found in slide table.\n"
    if not "path" in df.columns:
        msg += "No 'path' column found in slide table.\n"

    if msg != "":
        msg += f"Found columns: {df.columns.tolist()}"
        raise ValueError(msg)

    if "case_id" not in df.columns:
        logger.warning(
            "No case_id in input list, adding one based on slide and scanner."
        )
        df = df.reset_index().rename(columns={"index": "case_id"})
        df["case_id"] = df.apply(lambda row: f"{row['slide']}_{row['scanner']}", axis=1)

    # logger.info(str(df[target_label].unique()))
    # logger.info(f"{target_label}")
    # logger.info(f"{df.columns.tolist()}")
    category_to_index = get_category_to_index(df[target_label])

    df[target_label + "_index"] = [
        category_to_index[x] if (not isinstance(x, float) or not np.isnan(x)) else pd.NA
        for x in df[target_label].to_numpy()
    ]
    if len(df) == 1:
        raise ValueError(f"Something went wrong, the dataframe is empty")
    return df, category_to_index


def verify_data_exists(df: pd.DataFrame, disable_TQDM: bool) -> None:
    logger.info("Verifying existence of files")
    num_doesnt_exist = 0
    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Verifying",
        leave=False,
        dynamic_ncols=True,
        disable=disable_TQDM,
    ):
        src_path = Path(row["path"])

        if not src_path.exists():
            logger.info(f"File {src_path} does not exist")
            num_doesnt_exist += 1

    if num_doesnt_exist > 0:
        raise ValueError(f"Didn't find {num_doesnt_exist} datapoints...")
    else:
        logger.info("All files exist, continuing training.")


def move_data_to_temp(
    df: pd.DataFrame,
    temp_dir: str | Path = "temp/",
    disable_TQDM: bool = False,
    verify: bool = False,
) -> pd.DataFrame:
    """
    Copy files from source paths to temp directory using a single rsync call.

    Uses rsync's --files-from option to batch all file copies into one subprocess,
    avoiding the overhead of spawning thousands of individual rsync processes.

    When ``verify`` is True, existing temp files are compared against their source
    by size and mtime; mismatches are re-copied (overwriting the stale temp copy).
    """
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Moving files to temp storage")

    # Build mapping of src -> dest and track which files need copying
    path_mapping: Dict[str, Path] = {}  # src_path -> dest_path
    files_to_copy: List[Path] = []  # list of src paths that don't exist at dest yet
    num_stale = 0

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Checking files",
        leave=False,
        dynamic_ncols=True,
        disable=disable_TQDM,
    ):
        src_path = Path(row["path"])
        # Put it as the entire source path since .h5 files can have same names but be different files
        # Only differentiated by their filepaths. Since it's only temp storage, we don't care about
        # readability in the path system anyway
        dest_path = (
            temp_dir / src_path.parent.relative_to(src_path.anchor) / src_path.name
        )
        path_mapping[str(src_path)] = dest_path

        if not dest_path.exists():
            files_to_copy.append(src_path)
            dest_path.parent.mkdir(exist_ok=True, parents=True)
        elif verify:
            src_stat = src_path.stat()
            dest_stat = dest_path.stat()
            if (
                src_stat.st_size != dest_stat.st_size
                or src_stat.st_mtime != dest_stat.st_mtime
            ):
                files_to_copy.append(src_path)
                num_stale += 1

    if verify:
        if num_stale:
            logger.info(
                f"{num_stale}/{len(df)} temp file(s) are stale (size/mtime mismatch). Will overwrite"
            )
        else:
            logger.info("All temp files verified as up-to-date")

    if files_to_copy:
        logger.info(f"Copying {len(files_to_copy)} files to temp storage via rsync")

        # Write file list to temp file for rsync --files-from
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            for src in files_to_copy:
                # Write absolute paths; rsync will read them relative to "/"
                f.write(str(src) + "\n")
            file_list_path = f.name

        try:
            # rsync with --files-from reads absolute paths from the file,
            # copies to dest preserving directory structure
            result = subprocess.run(
                [
                    "rsync",
                    "-av",
                    "--files-from",
                    file_list_path,
                    "/",  # root as source base for absolute paths
                    str(temp_dir),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.debug(f"rsync output: {result.stdout.decode()}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"rsync failed: {e.stderr.decode().strip()}") from e
        finally:
            Path(file_list_path).unlink()  # Clean up temp file
    else:
        logger.info("All files already exist in temp storage, skipping copy")

    # Update DataFrame paths
    for idx, row in df.iterrows():
        src_path = str(row["path"])
        df.at[idx, "path"] = str(path_mapping[src_path])

    return df
