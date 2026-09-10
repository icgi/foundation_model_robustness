import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from itertools import combinations

from loguru import logger
import numpy as np


def get_scanner_pairs(scanners):
    scanner_pairs = list(combinations(scanners, 2))
    return scanner_pairs


def print_best_model_info(
    score_dict: Dict,
    metrics: List,
    best_idx: int,
    scanners: List[str],
    scores: List[float],
):
    logger.info(f"Best model metrics:")
    logger.info(f"\tScore: {scores[best_idx]}")
    logger.info(f"\tBest model checkpoint: {score_dict['checkpoint'][best_idx]}")
    for metric in metrics:
        if metric.is_paired_metric:
            for pair in get_scanner_pairs(scanners):
                result = score_dict[f"{metric.name}_{'_'.join(pair)}"][best_idx]
                logger.info(f"\t{metric.name} ({pair}): {result}")
        else:
            for scanner in scanners:
                if metric.supports_per_scanner:
                    result = score_dict[f"{metric.name}_{scanner}"][best_idx]
                    logger.info(f"\t{metric.name} ({scanner}): {result}")
            logger.info(f"\t{metric.name}: {score_dict[f'{metric.name}'][best_idx]}")
    logger.info("")


def get_auc_dict_from_scanners(df, scanners, idx):
    name_to_auc_dict = {}
    for scanner in scanners:
        name_to_auc_dict[f"auc_{scanner}"] = df.loc[idx][f"auc_{scanner}"]
    return name_to_auc_dict


def get_dict_from_partial_name(df, partial_name, idx):
    name_to_val_dict = {}
    for col in df:
        if partial_name in col:
            name_to_val_dict[col] = df.loc[idx][col]
    return name_to_val_dict


def save_best_model_info(
    score_df,
    paths,
    scanners,
    inference_path,
    main_metric,
    main_metric_weight,
    secondary_metric,
    secondary_metric_weight,
    best_idx,
    choose_last,
):
    best_auc = score_df.loc[best_idx]["auc"]
    auc_scanner_dict = get_auc_dict_from_scanners(score_df, scanners, best_idx)
    try:
        best_acc = score_df.loc[best_idx]["balanced_accuracy"]
    except KeyError:
        best_acc = None
    best_rmse_dict = get_dict_from_partial_name(score_df, "rmse", best_idx)
    best_score = score_df.loc[best_idx]["score"]
    best_mean_difference_over_std_dict = get_dict_from_partial_name(
        score_df, "mean_difference_over_std", best_idx
    )
    best_percentage_agreement_dict = get_dict_from_partial_name(
        score_df, "percentage_agreement", best_idx
    )
    bac_scanner_dict = get_dict_from_partial_name(
        score_df, "balanced_accuracy", best_idx
    )
    best_spread = score_df.loc[best_idx]["spread"]
    spread_scanner_dict = get_dict_from_partial_name(score_df, "spread", best_idx)

    best_model_path_csv = paths[best_idx]
    model_path = score_df.loc[best_idx]["model_used"]

    info = {
        "best_checkpoint": str(best_model_path_csv),
        "model_path": model_path,
        "auc": best_auc,
        **auc_scanner_dict,
        **best_rmse_dict,
        "score": best_score,
        "balanced_accuracy": best_acc,
        "main_metric": main_metric,
        "metric_weight": main_metric_weight,
        "secondary_metric": secondary_metric,
        "secondary_metric_weight": secondary_metric_weight,
        **best_mean_difference_over_std_dict,
        **best_percentage_agreement_dict,
        "chose_last": choose_last,
        "spread": best_spread,
        **bac_scanner_dict,
        **spread_scanner_dict,
    }

    if "CSS_event" in score_df.columns:
        best_c_index_dict = get_dict_from_partial_name(score_df, "c_index", best_idx)
        info.update(best_c_index_dict)

    with open(inference_path / "best_checkpoint.json", "w") as f:
        json.dump(info, f)


def get_prob_cols(df: pd.DataFrame, label, labels):
    try:
        prob_cols = [
            col
            for col in df.columns
            if label in col
            and col != label
            and (col.split("_")[-1]).lower() in [str(l).lower() for l in labels]
        ]
    except Exception as e:
        prob_cols = []
        raise e
    if not prob_cols:
        print("Labels:\n", labels)
        print("Dataframe columns:\n", df.columns)
        print("Label:\n\t", label)
        raise ValueError(f"No probability columns found for label '{label}'")
    return prob_cols


def fix_df(df: pd.DataFrame) -> pd.DataFrame:
    if not "slide" in df.columns:
        df["slide"] = df["case_id"].apply(lambda x: "_".join(x.split("_")[:-1]))
    if not "scanner" in df.columns:
        df["scanner"] = df["case_id"].apply(lambda x: x.split("_")[-1])

    return df


def get_score(
    main_metric_value,
    main_metric_weight,
    secondary_metric_value,
    secondary_metric_weight,
    subtract_or_add,
):
    if subtract_or_add == "subtract":
        score = (
            main_metric_value * main_metric_weight
            - secondary_metric_value * secondary_metric_weight
        )
    elif subtract_or_add == "add":
        score = (
            secondary_metric_value * secondary_metric_weight
            + main_metric_value * main_metric_weight
        )
    elif subtract_or_add == "old":
        score = main_metric_value * main_metric_weight + (
            (1 - secondary_metric_value) * secondary_metric_weight
        )
    else:
        raise ValueError(f"{subtract_or_add} is an invalid subtract_or_add")
    return score


def error_nans(df: pd.DataFrame, csv_name: str, label: str, labels: List[str]):
    err = False
    for col in get_prob_cols(df, label, labels) + [
        "predicted_label",
        "predicted_idx",
        label,
        "slide",
        "scanner",
    ]:
        nans = df[col].isna().to_numpy()
        if nans.sum() > 0:
            logger.warning(f"Found {nans.sum()} NaNs in column {col} for {csv_name}")
            err = True
    if err:
        raise ValueError(
            f"Found NaNs. Please review your csvs or set the allow_nans flag."
        )


def handle_nans(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    all_nans = np.zeros(df.shape[0], dtype=np.bool_)
    for col in cols:
        nans = df[col].isna().to_numpy()
        if nans.sum() > 0:
            nan_str = f"Found {nans.sum()} NaNs in column {col}"
            logger.warning(nan_str)

            all_nans = all_nans | nans

    if all_nans.sum() > 0:
        logger.warning(f"Dropping {all_nans.sum()} rows due to NaNs.")
        df = df.loc[~all_nans]
    return df
