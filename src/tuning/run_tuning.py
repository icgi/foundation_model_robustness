from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

from src.tuning.metrics import (
    Metric,
    CIndexMetric,
    RMSEMetric,
    BalancedAccuracyMetric,
    SpreadMetric,
    AUCMetric,
    PercentageAgreementMetric,
    RobustnessMetric,
    MeanAbsDiffOverStdMetric,
    CoVMetric,
    ConcordanceCorrelationCoefficient,
    RankBasedHazardRatio,
    CategoricalHazardRatio,
    get_metric_from_name,
)
from src.tuning.utils import (
    get_scanner_pairs,
    print_best_model_info,
    save_best_model_info,
    fix_df,
    get_score,
    error_nans,
)


def tune_inference(
    inference_path: Path,
    label,
    labels,
    main_metric,
    main_metric_weight,
    secondary_metric,
    secondary_metric_weight,
    subtract_or_add,
    choose_last,
    allow_nans,
    verbose=True,
):

    # Get all the checkpoint predictions. The weird code here is a workaround for a bug...
    inference_csvs = list((inference_path).rglob("*model-step*_predictions.csv"))
    if len(inference_csvs) > 1:
        # We get the last one by date
        inference_csvs = sorted(
            inference_csvs, key=lambda x: x.stat().st_mtime, reverse=True
        )[:1]
    assert len(inference_csvs) > 0, f"No inference csvs found in {inference_path}"
    # Sort inference csvs by number. Format is
    get_step = lambda x: int(x.stem.split("_")[-2].split("-")[-1])
    inference_csvs = sorted(inference_csvs, key=get_step)

    kwargs = {"label": label, "labels": labels}

    metrics: List[Metric] = [
        CIndexMetric(**kwargs),
        RMSEMetric(**kwargs),
        BalancedAccuracyMetric(**kwargs),
        SpreadMetric(**kwargs),
        AUCMetric(label, labels, labels),
        PercentageAgreementMetric(**kwargs),
        RobustnessMetric(**kwargs),
        MeanAbsDiffOverStdMetric(**kwargs),
        # CoVMetric(**kwargs),  # NOTE: This is the metric that uses logits...
        ConcordanceCorrelationCoefficient(**kwargs),
        RankBasedHazardRatio(**kwargs),
        # CategoricalHazardRatio(**kwargs),
    ]

    scanners = pd.read_csv(inference_csvs[0])["scanner"].unique().tolist()

    score_df_dict = {"checkpoint": [], "paths": [], "model_used": []}
    for metric in metrics:
        if metric.is_paired_metric:
            for pair in get_scanner_pairs(scanners):
                score_df_dict[f"{metric.name}_{'_'.join(pair)}"] = []
        else:
            for scanner in scanners:
                if metric.supports_per_scanner:
                    score_df_dict[f"{metric.name}_{scanner}"] = []
            score_df_dict[f"{metric.name}"] = []

    i = 0
    for csv_path in inference_csvs:
        score_df_dict["checkpoint"].append(get_step(csv_path))
        score_df_dict["paths"].append(csv_path)
        df = pd.read_csv(csv_path)
        df = fix_df(df)
        if not allow_nans:
            error_nans(df, csv_path.name, label, labels)
        try:
            models_used = list(df["model_used"].unique())
        except:
            models_used = ["unknown"]
        assert (
            len(df["model_used"].unique()) == 1
        ), f"Should be only one model but: {df['model_used'].unique()}"
        score_df_dict["model_used"].append(models_used[0])

        for metric in metrics:
            if metric.is_paired_metric:
                try:
                    metric_results = metric.calculate(df, scanners)
                except Exception as e:
                    logger.warning(f"Scanners: {scanners}")
                    logger.warning(f"DF: {df.head()}")
                    raise e
                for pair, result in metric_results.items():
                    score_df_dict[f"{metric.name}_{'_'.join(pair)}"].append(result)

            else:
                if metric.supports_per_scanner:
                    metric_results_per_scanner = metric.calculate(
                        df, scanners, per_scanner=True
                    )
                    for scanner, result in metric_results_per_scanner.items():
                        score_df_dict[f"{metric.name}_{scanner}"].append(result)

                metric_results_total = metric.calculate(df, scanners, per_scanner=False)
                score_df_dict[f"{metric.name}"].append(metric_results_total)

        if verbose:
            logger.info("Metrics:")
            for metric in metrics:
                if metric.is_paired_metric:
                    for pair in get_scanner_pairs(scanners):
                        result = score_df_dict[f"{metric.name}_{'_'.join(pair)}"][-1]
                        logger.info(f"\t{metric.name} ({pair}): {result}")
                else:
                    for scanner in scanners:
                        if metric.supports_per_scanner:
                            result = score_df_dict[f"{metric.name}_{scanner}"][-1]
                            logger.info(f"\t{metric.name} ({scanner}): {result}")
                    logger.info(
                        f"\t{metric.name}: {score_df_dict[f'{metric.name}'][-1]}"
                    )
            logger.info("")
        i += 1
    # Also save csv with all information, except for score which we calculate later
    score_df = pd.DataFrame(score_df_dict)

    # TODO: This part is still ugly
    try:
        main_metric_value = score_df[main_metric]
    except KeyError:
        raise ValueError(
            f"{main_metric} is an invalid main_metric. Valid metrics are: {', '.join(score_df_dict.keys())}"
        )
    try:
        if secondary_metric == "none":
            secondary_metric_value = 0
        else:
            metric = get_metric_from_name(secondary_metric, metrics)
            if metric.is_paired_metric:
                values = []
                for pair in get_scanner_pairs(scanners):
                    values.append(score_df[f"{secondary_metric}_{'_'.join(pair)}"])
                secondary_metric_value = np.mean(values)
            else:
                secondary_metric_value = score_df[secondary_metric]
    except KeyError:
        raise ValueError(
            f"{secondary_metric} is an invalid secondary_metric. Valid metrics are: {', '.join(score_df_dict.keys())}"
        )

    scores = get_score(
        main_metric_value,
        main_metric_weight,
        secondary_metric_value,
        secondary_metric_weight,
        subtract_or_add,
    )

    if verbose:
        logger.info(
            "TODO: Implement a scaling so that the min/max between second metric is the same as the range of the first metric"
        )
    score_df["score"] = scores

    if choose_last:
        best_idx = score_df["checkpoint"].idxmax()
        logger.info("Choosing last")
    else:
        best_idx = score_df["score"].idxmax()
        logger.info("Choosing best")

    if verbose:
        print_best_model_info(score_df_dict, metrics, best_idx, scanners, scores)

    score_df.to_csv(inference_path / "scores.csv", index=False)
    logger.info(f"Saved scores to {inference_path / 'scores.csv'}")

    save_best_model_info(
        score_df,
        score_df["paths"],
        scanners,
        inference_path,
        main_metric,
        main_metric_weight,
        secondary_metric,
        secondary_metric_weight,
        best_idx,
        choose_last,
    )
    logger.info(f"Saved best model info to {inference_path / 'best_checkpoint.json'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # TODO: Rename
    parser.add_argument(
        "-ip",
        "--inference_path",
        type=Path,
        help="Path to folder of the inference_files",
    )

    parser.add_argument(
        "-l",
        "--label",
        default="outcome_lnm",
        help="The name of the label in the .csv",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="The possible labels in the .csv. If not provided, will use all unique labels in the .csv. This can cause issue if the test set is very imbalanced",
    )

    parser.add_argument(
        "--scanners",
        nargs="+",
        default=["aperio", "xr"],
        help="The scanners to compare",
    )

    parser.add_argument(
        "--main_metric",
        choices=["auc", "balanced_accuracy", "cindex"],
        default="auc",
        help="The metric to use for the best model",
    )
    parser.add_argument(
        "--main_metric_weight",
        type=float,
        default=1,
        help="The weight of the main metric",
    )

    parser.add_argument(
        "--secondary_metric",
        choices=["rmse", "mean_abs_difference_over_std", "robustness", "none"],
        default="rmse",
        help="The secondary metric to use for the best model",
    )

    parser.add_argument(
        "--secondary_metric_weight",
        type=float,
        default=1,
        help="The weight of the secondary metric",
    )

    parser.add_argument(
        "--add_or_subtract",
        choices=["add", "subtract", "old"],
        default="subtract",
        help="Whether to add or subtract the secondary metric. Old means 1-secondary_metric * secondary_metric_weight",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Set this flag in order to overwrite previous results",
    )
    parser.add_argument(
        "--choose_last",
        action="store_true",
        help="Set this flag to choose the last checkpoint",
    )

    parser.add_argument(
        "--allow_nans",
        action="store_true",
        help="If this flag is set, NaNs will be dropped. Otherwise, an error will be raised",
    )

    args = parser.parse_args()

    assert (
        len((list(args.inference_path.rglob("*_predictions.csv")))) > 0
    ), "No predictions found"

    if not args.force:
        assert not (
            args.inference_path.parent / "best_checkpoint.json"
        ).exists(), f"Best checkpoint already found in {args.inference_path.parent.name}, skipping"

    if args.secondary_metric_weight is None:
        args.secondary_metric_weight = 1 - args.main_metric_weight

    tune_inference(
        args.inference_path,
        args.label,
        args.labels,
        args.main_metric,
        args.main_metric_weight,
        args.secondary_metric,
        args.secondary_metric_weight,
        args.add_or_subtract,
        args.choose_last,
        args.allow_nans,
        verbose=True,
    )
