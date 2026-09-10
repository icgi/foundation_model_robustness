from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from sklearn.metrics import root_mean_squared_error

from src.tuning.metric_utils import calculate_auc, calculate_bac
from src.tuning.utils import get_prob_cols, get_scanner_pairs, handle_nans


# Custom exception when a metric does not exist (e.g. RMSE for a single scanner)
class MetricCannotBeCalculatedError(Exception):
    def __init__(self, metric_name):
        super().__init__(f"Metric '{metric_name}' does not exist for the given data.")
        self.metric_name = metric_name


class Metric:
    def __init__(self, name, description=None):
        self.name = name
        self.description = description
        self._is_paired_metric = None
        self._supports_per_scanner = None

    @property
    def is_paired_metric(self):
        if self._is_paired_metric is None:
            raise ValueError(
                f"is_between_pairs has not been set for {self}. This should be set in the __init__ method of the subclass."
            )
        return self._is_paired_metric

    @property
    def supports_per_scanner(self):
        return self._supports_per_scanner

    def get_scanner_dfs(self, df, scanners):
        scanner_dfs = {}
        df = df.copy()

        for scanner in scanners:
            scanner_df = df[df["scanner"].str.lower() == scanner.lower()].copy()
            scanner_dfs[scanner] = scanner_df

        return scanner_dfs

    def calculate(self, df, scanners, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses")

    def __repr__(self):
        return f"Metric(name={self.name}, description={self.description})"


class CIndexMetric(Metric):
    def __init__(
        self,
        label: str,
        labels: List[str],
        event_col: str = "CSS_event",
        time_col: str = "CSS_time",
        flip_pred_sign: bool = True,
    ):
        # Needed kwargs are: Time-column and Event-column
        super().__init__(
            name="cindex",
            description="Concordance index for survival analysis.",
        )
        self.event_col = event_col
        self.time_col = time_col
        self.flip_pred_sign_multiplier = 1 if not flip_pred_sign else -1
        self.label = label
        self.labels = labels
        self._is_paired_metric = False
        self._supports_per_scanner = True

    def calculate(self, df, scanners, per_scanner=False):
        if not (self.event_col in df.columns and self.time_col in df.columns):
            return 0.0 if not per_scanner else {scanner: 0.0 for scanner in scanners}

        df = df.copy()
        pred_cols = get_prob_cols(df, self.label, self.labels)
        pred_col = pred_cols[1]
        df = handle_nans(df, ["slide", self.event_col, self.time_col, pred_col])

        if not per_scanner:
            # Calculate the concordance index for the entire dataframe
            # TODO: Average over scanners per slide in the prediciton.
            df = df.copy()[["slide", self.event_col, self.time_col, pred_col]]
            df_average_pred_col_per_slide = df.groupby("slide", as_index=False).agg(
                {
                    pred_col: "mean",
                    self.event_col: "first",
                    self.time_col: "first",
                }
            )

            check = df.groupby("slide")[[self.event_col, self.time_col]].nunique()
            if (check[self.event_col] > 1).any() or (check[self.time_col] > 1).any():
                raise ValueError("event/time are not consistent within slide.")

            c_index_patient_level = concordance_index(
                df_average_pred_col_per_slide[self.time_col],
                self.flip_pred_sign_multiplier
                * df_average_pred_col_per_slide[pred_col],
                df_average_pred_col_per_slide[self.event_col],
            )

            return c_index_patient_level

        else:
            scanner_dfs = self.get_scanner_dfs(df, scanners)

            scanner_to_c_index = {}
            for scanner, scanner_df in scanner_dfs.items():

                # Assume columns "CSS_event" and "CSS_TIME" are the event and time columns
                # NOTE: This is hardcoded for CRC, fix at some point
                try:
                    c_index = concordance_index(
                        scanner_df[self.time_col],
                        self.flip_pred_sign_multiplier * scanner_df[pred_col],
                        scanner_df[self.event_col],
                    )
                    scanner_to_c_index[scanner] = c_index
                except Exception as e:
                    logger.info(f"Error calculating C-index for {scanner}: {e}")
                    scanner_to_c_index[scanner] = None

            return scanner_to_c_index


class RMSEMetric(Metric):
    def __init__(
        self,
        label: str,
        labels: List[str],
    ):
        super().__init__(
            name="rmse",
            description="Root Mean Squared Error between the probability predictions of different scanners.",
        )
        self.label = label
        self.labels = labels
        self._is_paired_metric = True

    def calculate(self, df, scanners, **kwargs):
        prob_cols = get_prob_cols(df, self.label, self.labels)

        scanner_dfs = self.get_scanner_dfs(df, scanners)
        pairs = get_scanner_pairs(scanners)
        df = df.copy()
        df = handle_nans(df, prob_cols)

        if len(scanners) < 2:
            raise MetricCannotBeCalculatedError(
                "RMSE metric can only be calculated for at least two scanners."
            )

        pair_to_rmse = {}

        for scanner_1_name, scanner_2_name in pairs:

            scanner_1 = scanner_dfs[scanner_1_name]
            scanner_2 = scanner_dfs[scanner_2_name]

            merged_df = pd.merge(
                scanner_1,
                scanner_2,
                on="slide",
                suffixes=(f"_{scanner_1_name}", f"_{scanner_2_name}"),
                validate="one_to_one",
            )

            # Extract the probability columns
            scanner_1_probs = []
            scanner_2_probs = []
            for col in prob_cols:
                scanner_1_probs.extend(merged_df[f"{col}_{scanner_1_name}"].values)
                scanner_2_probs.extend(merged_df[f"{col}_{scanner_2_name}"].values)

            # Compute RMSE between the probability vectors
            RMSE = root_mean_squared_error(scanner_1_probs, scanner_2_probs)

            pair_to_rmse[(scanner_1_name, scanner_2_name)] = RMSE
        return pair_to_rmse


class BalancedAccuracyMetric(Metric):
    def __init__(
        self,
        label: str,
        labels: List[str],
    ):
        super().__init__(
            name="balanced_accuracy",
            description="Balanced accuracy of the predictions.",
        )
        self.label = label
        self.labels = labels
        self._is_paired_metric = False
        self._supports_per_scanner = True

    def calculate(self, df, scanners, per_scanner: bool = False, **kwargs):
        """
        Calculates the accuracy of predictions.

        Parameters:
        df (pd.DataFrame): The input dataframe.
        label_col (str): The name of the true label column.

        Returns:
        float: The accuracy of the predictions.
        """
        df = df.copy()
        assert (
            "predicted_idx" in df.columns
        ), "predicted_idx column is missing from the dataframe"
        assert (
            self.label + "_index" in df.columns
        ), f"{self.label}_index column is missing from the dataframe"
        df = handle_nans(
            df, ["predicted_idx", "predicted_label", self.label + "_index"]
        )
        if per_scanner:
            bac_per_scanner = {}
            for scanner, scanner_df in self.get_scanner_dfs(df, scanners).items():
                labels = scanner_df[self.label + "_index"]
                preds = scanner_df["predicted_idx"]

                bac = calculate_bac(
                    preds,
                    labels,
                )
                bac_per_scanner[scanner] = bac
            return bac_per_scanner

        else:
            labels = df[self.label + "_index"]
            preds = df["predicted_idx"]

            return calculate_bac(preds, labels)


class SpreadMetric(Metric):
    def __init__(
        self,
        label: str,
        labels: List[str],
        label_column_number: int = 0,
    ):
        super().__init__(
            name="spread",
            description="Spread of the predictions across scanners.",
        )
        self.label_column_number = label_column_number
        self.label = label
        self.labels = labels
        self._is_paired_metric = False
        self._supports_per_scanner = True

    def calculate(self, df, scanners, per_scanner: bool = False, **kwargs):
        """
        Calculates the spread of the predictions.

        Parameters:
        df (pd.DataFrame): The input dataframe.
        scanners (list): List of scanner names.

        Returns:
        tuple: total_spread, scanner_spreads
        """

        # Calculate the spread of the predictions, to make sure they are not all the same
        prob_cols = get_prob_cols(df, self.label, self.labels)
        col = prob_cols[self.label_column_number]
        df = df.copy()
        df = handle_nans(df, [col])
        if per_scanner:
            spread_dict = {}
            for scanner, scanner_df in self.get_scanner_dfs(df, scanners).items():
                diff = scanner_df[col].max() - scanner_df[col].min()
                spread_dict[scanner] = diff
            return spread_dict
        else:
            # Calculate spread for all scanners together
            diff = df[col].max() - df[col].min()

            return diff


class AUCMetric(Metric):
    def __init__(
        self,
        label: str,
        labels: List[str],
        unique_labels,
    ):
        super().__init__(
            name="auc",
            description="Area Under the Receiver Operating Characteristic Curve (AUC) for the predictions.",
        )
        self.label = label
        self.labels = labels
        self.unique_labels = unique_labels
        self._is_paired_metric = False
        self._supports_per_scanner = True

    def calculate(self, df, scanners, per_scanner: bool = False, **kwargs):
        """
        Calculates the AUC for the overall predictions, and separately for aperio and xr scanners.

        Parameters:
        df (pd.DataFrame): The input dataframe.
        label_col (str): The name of the true label column.
        prob_cols (list of str): The list of probability column names.

        Returns:
        tuple: total_auc, aperio_auc, xr_auc
        """
        df = df.copy()
        prob_cols = get_prob_cols(df, self.label, self.labels)
        df = handle_nans(df, prob_cols + [self.label + "_index"])
        labels = df[self.label + "_index"].values
        probs = df[prob_cols].values

        binary = probs.shape[1] == 2
        scanner_aucs = {}

        scanner_dfs = self.get_scanner_dfs(df, scanners)

        if not per_scanner:
            if binary:
                total_auc = calculate_auc(
                    probs[:, 1], labels, labels=self.unique_labels
                )
            else:
                total_auc = calculate_auc(
                    probs, labels, multi_class="ovr", labels=self.unique_labels
                )
            return total_auc
        else:
            # TODO: Could move the if statement into the loop here...
            if binary:
                for scanner in scanners:
                    scanner_labels = scanner_dfs[scanner][self.label + "_index"].values
                    scanner_probs = scanner_dfs[scanner][prob_cols].values
                    scanner_auc = calculate_auc(
                        scanner_probs[:, 1], scanner_labels, labels=self.unique_labels
                    )
                    scanner_aucs[scanner] = scanner_auc
            else:
                for scanner in scanners:
                    scanner_labels = scanner_dfs[scanner][self.label + "_index"].values
                    scanner_probs = scanner_dfs[scanner][prob_cols].values
                    scanner_auc = calculate_auc(
                        scanner_probs,
                        scanner_labels,
                        multi_class="ovr",
                        labels=self.unique_labels,
                    )
                    scanner_aucs[scanner] = scanner_auc
            return scanner_aucs


class PercentageAgreementMetric(Metric):
    def __init__(
        self,
        label: str,
        labels: List[str],
    ):
        super().__init__(
            name="percentage_agreement",
            description="Percentage agreement between aperio and xr predictions.",
        )

        self._is_paired_metric = True

    def calculate(self, df, scanners, **kwargs):
        """
        Calculates the percentage agreement between aperio and xr predictions.

        Parameters:
        df (pd.DataFrame): The input dataframe.

        Returns:
        float: The percentage of agreement between aperio and xr predictions.
        """
        df = df.copy()
        scanner_dfs = self.get_scanner_dfs(df, scanners)
        pairs = get_scanner_pairs(scanners)
        if len(scanners) == 1:
            return {pair: 0 for pair in pairs}

        df = handle_nans(df, ["predicted_label"])

        pair_to_percentage_agreement = {}
        for pair in pairs:
            scanner_1_name = pair[0]
            scanner_2_name = pair[1]
            scanner_1_df = scanner_dfs[scanner_1_name]
            scanner_2_df = scanner_dfs[scanner_2_name]
            merged_df = pd.merge(
                scanner_1_df,
                scanner_2_df,
                on="slide",
                suffixes=(f"_{scanner_1_name}", f"_{scanner_2_name}"),
                validate="one_to_one",
            )

            scanner_1_pred = merged_df[f"predicted_label_{scanner_1_name}"]
            scanner_2_pred = merged_df[f"predicted_label_{scanner_2_name}"]

            num_agree = (scanner_1_pred == scanner_2_pred).sum()
            percentage_agreement = num_agree / len(merged_df)
            pair_to_percentage_agreement[pair] = percentage_agreement
        return pair_to_percentage_agreement


class RobustnessMetric(Metric):
    def __init__(self, label: str, labels: List[str], label_column_number: int = 0):
        super().__init__(
            name="robustness",
            description="Robustness of the predictions across scanners.",
        )
        self.label = label
        self.labels = labels
        self._is_paired_metric = False
        self._supports_per_scanner = False
        self.label_column_number = label_column_number

    def calculate(self, df, scanners, per_scanner: bool = False, **kwargs):
        if per_scanner:
            raise MetricCannotBeCalculatedError(
                "Robustness metric does not support per-scanner calculation"
            )
        df = df.copy()
        prob_col = get_prob_cols(df, self.label, self.labels)[self.label_column_number]
        df = handle_nans(df, [prob_col])

        # 1. Get the standard deviation of the prediction score across all scanners for each slide
        slide_std = []
        for slide in df["slide"].unique():
            slide_df = df[df["slide"] == slide]

            if len(slide_df) != len(scanners):
                continue

            std = np.std(slide_df[prob_col].to_numpy().flatten())
            slide_std.append(std)

        slide_std = np.array(slide_std)
        # 2. Get the standard deviation of the prediction score across all whole slide images
        all_std = np.std([df[prob_col].to_numpy()])
        # 3. Calculate the mean of the slide standard deviations and divide by the overall standard deviation
        mean_slide_std = np.mean(slide_std)
        robustness = mean_slide_std / (all_std + 1e-8)
        return robustness


class MeanAbsDiffOverStdMetric(Metric):
    def __init__(
        self,
        label: str,
        labels: List[str],
    ):
        super().__init__(
            name="mean_abs_difference_over_std",
            description="Mean absolute difference of probabilities over the standard deviation of all probabilities.",
        )
        self.label = label
        self.labels = labels
        self._is_paired_metric = True

    def calculate(self, df, scanners, **kwargs):
        df = df.copy()
        prob_cols = get_prob_cols(df, self.label, self.labels)
        df = handle_nans(df, prob_cols)
        scanner_dfs = self.get_scanner_dfs(df, scanners)
        pairs = get_scanner_pairs(scanners)

        pair_to_mean_abs_diff_over_std = {}
        for scanner_1_name, scanner_2_name in pairs:
            scanner_1_df = scanner_dfs[scanner_1_name]
            scanner_2_df = scanner_dfs[scanner_2_name]
            merged_df = pd.merge(
                scanner_1_df,
                scanner_2_df,
                on="slide",
                suffixes=(f"_{scanner_1_name}", f"_{scanner_2_name}"),
                validate="one_to_one",
            )

            # Concatenate all probabilities to compute standard deviation
            all_probs = np.concatenate(
                [
                    merged_df[
                        [f"{col}_{scanner_1_name}" for col in prob_cols]
                    ].values.flatten(),
                    merged_df[
                        [f"{col}_{scanner_2_name}" for col in prob_cols]
                    ].values.flatten(),
                ]
            )
            std_probs = np.std(all_probs)

            # Compute absolute differences
            scanner_1_probs = merged_df[
                [f"{col}_{scanner_1_name}" for col in prob_cols]
            ].values
            scanner_2_probs = merged_df[
                [f"{col}_{scanner_2_name}" for col in prob_cols]
            ].values
            abs_diff = np.abs(scanner_1_probs - scanner_2_probs)

            mean_abs_diff_over_std = np.mean(abs_diff) / std_probs
            pair_to_mean_abs_diff_over_std[(scanner_1_name, scanner_2_name)] = (
                mean_abs_diff_over_std
            )

        return pair_to_mean_abs_diff_over_std


class CoVMetric(Metric):
    """
    CoV across scanners computed on a *scalar* logit score per (slide, scanner).

    With your setup:
      prob_cols = get_prob_cols(...)
      logit_cols = [col + "_logits" for col in prob_cols]

    For binary heads (2 logits), we use the shift-invariant logit difference:
      score = poor_logit - good_logit
    assuming prob_cols[1] is the "poor" column (same convention you use elsewhere).
    """

    def __init__(
        self,
        label: str,
        labels: list[str],
        require_all_scanners: bool = True,
        eps: float = 1e-8,
        positive_index: int = 1,
        negative_index: int = 0,
    ):
        super().__init__(
            name="cov",
            description="Average coefficient of variation across scanners on logit score (binary: poor-good).",
        )
        self.label = label
        self.labels = labels
        self.require_all_scanners = require_all_scanners
        self.eps = eps
        self.positive_index = positive_index
        self.negative_index = negative_index

        self._is_paired_metric = False
        self._supports_per_scanner = False

    def calculate(
        self, df: pd.DataFrame, scanners: List, per_scanner: bool = False, **kwargs
    ):
        if per_scanner:
            raise MetricCannotBeCalculatedError(
                "CoV metric does not support per-scanner calculation."
            )
        scanners = [s for s in scanners if str(s).lower() != "external"]

        if len(scanners) != len(scanners):
            logger.warning(
                "Removing external for CoVMetric, as it's only meant across scanners"
            )
        if len(scanners) < 2:
            raise MetricCannotBeCalculatedError("CoV requires at least two scanners.")

        df = df.copy()

        # Resolve prob/logit columns from your convention
        prob_cols = get_prob_cols(df, self.label, self.labels)
        logit_cols = [f"{c}_logits" for c in prob_cols]

        for col in logit_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in dataframe.")

        df = handle_nans(df, logit_cols)

        # Filter to requested scanners (case-insensitive)
        scanners_lc = {s.lower() for s in scanners}
        df = df[df["scanner"].astype(str).str.lower().isin(scanners_lc)]

        # Build scalar score per row:
        # - binary (2 logits): poor - good (shift-invariant)
        # - otherwise: use the "positive_index" logit directly
        if len(logit_cols) == 2:
            pos_logit = logit_cols[self.positive_index]
            neg_logit = logit_cols[self.negative_index]
            df["_cov_score"] = df[pos_logit].astype(float) - df[neg_logit].astype(float)
        elif len(logit_cols) > 2:
            return 0.0
        else:
            df["_cov_score"] = df[logit_cols[self.positive_index]].astype(float)

        # Ensure one score per (slide, scanner): average duplicates if present
        per_slide_scanner = df.groupby(["slide", "scanner"], as_index=False)[
            "_cov_score"
        ].mean()

        covs = []
        skipped_missing = 0
        skipped_mean0 = 0

        # CoV per slide across scanners, then average over slides
        for slide, g in per_slide_scanner.groupby("slide"):
            n_scanners = g["scanner"].nunique()

            if self.require_all_scanners:
                if n_scanners != len(scanners):
                    skipped_missing += 1
                    continue
            else:
                if n_scanners < 2:
                    skipped_missing += 1
                    continue

            vals = g["_cov_score"].to_numpy(dtype=float)
            mu = float(np.mean(vals))
            denom = abs(mu)

            # Avoid unstable division when mean ≈ 0
            if denom < self.eps:
                skipped_mean0 += 1
                continue

            sigma = float(np.std(vals))
            covs.append(sigma / denom)

        if not covs:
            raise MetricCannotBeCalculatedError(
                "CoV could not be calculated (no valid slides after filtering; "
                f"skipped_missing={skipped_missing}, skipped_mean≈0={skipped_mean0})."
            )

        if skipped_missing or skipped_mean0:
            logger.info(
                f"CoVMetric: used {len(covs)} slides; "
                f"skipped_missing={skipped_missing}, skipped_mean≈0={skipped_mean0}"
            )

        return float(np.mean(covs))


class ConcordanceCorrelationCoefficient(Metric):
    # See https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    def __init__(
        self,
        label: str,
        labels: List[str],
    ):
        super().__init__(
            name="concordance_correlation_coefficient",
            description='Concordance correlation coefficient." '
            '" To quote Wikipedia: "In statistics, the concordance correlation coefficient measures the agreement between two variables," '
            '" e.g., to evaluate reproducibility or for inter-rater reliability. "',
        )
        self.label = label
        self.labels = labels
        self._is_paired_metric = True

    def calculate(self, df, scanners, **kwargs):
        df = df.copy()
        prob_col = get_prob_cols(df, self.label, self.labels)[0]
        df = handle_nans(df, [prob_col])
        scanner_dfs = self.get_scanner_dfs(df, scanners)
        pairs = get_scanner_pairs(scanners)

        pair_to_CCC = {}
        for scanner_1_name, scanner_2_name in pairs:
            scanner_1_df = scanner_dfs[scanner_1_name]
            scanner_2_df = scanner_dfs[scanner_2_name]
            merged_df = pd.merge(
                scanner_1_df,
                scanner_2_df,
                on="slide",
                suffixes=(f"_x", f"_y"),
                validate="one_to_one",
            )
            x = merged_df[prob_col + "_x"]
            y = merged_df[prob_col + "_y"]

            cov = x.cov(y)

            CCC = (2 * cov) / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
            pair_to_CCC[(scanner_1_name, scanner_2_name)] = CCC

        return pair_to_CCC


class RankBasedHazardRatio(Metric):
    def __init__(
        self,
        label: str,
        labels: List[str],
        event_col: str = "CSS_event",
        time_col: str = "CSS_time",
    ):
        super().__init__(
            name="rank_based_hazard_ratio",
            description="The rank-based hazard ratio, using a rank-transformed variable.",
        )
        self.label = label
        self.labels = labels
        self.event_col = event_col
        self.time_col = time_col

        self._is_paired_metric = False
        self._supports_per_scanner = True

    def _prepare_df(self, df, prob_col):
        df = df.copy()[["slide", self.time_col, self.event_col, prob_col]]
        df = df.dropna(subset=[self.time_col, self.event_col, prob_col])

        df = df.groupby("slide", as_index=False).agg(
            {
                prob_col: "mean",
                self.event_col: "first",
                self.time_col: "first",
            }
        )
        return df

    def _calculate_HR(self, df, prob_col):
        df = self._prepare_df(df, prob_col)

        if len(df) < 2:
            raise ValueError()

        if df[self.event_col].sum() == 0:
            raise ValueError()

        if df[self.time_col].nunique() < 2:
            raise ValueError()

        std = df[prob_col].std()
        if pd.isna(std) or std == 0:
            return np.nan
            raise ValueError(f"Std is too low or nan: {std}")
        df = df.copy()
        # Rank-transform to be more robust, and to get HR per quantile
        df["rank"] = df[prob_col].rank(method="first") / len(df)
        cph = CoxPHFitter(penalizer=1e-5)
        try:
            cph.fit(
                df[[self.time_col, self.event_col, "rank"]],
                duration_col=self.time_col,
                event_col=self.event_col,
            )
        except ConvergenceError as e:
            print(e)
            return np.nan

        beta = float(cph.params_["rank"])
        hr = float(np.exp(beta))
        return hr

    def calculate(self, df, scanners, per_scanner=False, **kwargs):
        if not (self.event_col in df.columns and self.time_col in df.columns):
            return (
                np.nan if not per_scanner else {scanner: np.nan for scanner in scanners}
            )
        df = df.copy()

        prob_cols = get_prob_cols(df, self.label, self.labels)
        prob_col = prob_cols[-1]
        df = handle_nans(df, [prob_col])

        if per_scanner:
            scanner_dfs = self.get_scanner_dfs(df, scanners)
            CHR_per_scanner = {}
            for scanner, scanner_df in scanner_dfs.items():
                CHR_per_scanner[scanner] = self._calculate_HR(scanner_df, prob_col)
            return CHR_per_scanner

        return self._calculate_HR(df, prob_col)


class CategoricalHazardRatio(Metric):
    def __init__(
        self,
        label: str,
        labels: List[str],
        event_col: str = "CSS_event",
        time_col: str = "CSS_time",
        threshold: float | None = 0.5,
    ):
        super().__init__(
            name="categorical_hazard_ratio",
            description="The categorical hazard ratio, using a binary high-risk vs low-risk group.",
        )
        self.label = label
        self.labels = labels
        self.event_col = event_col
        self.time_col = time_col
        self.threshold = threshold

        self._is_paired_metric = False
        self._supports_per_scanner = True

    def _prepare_df(self, df, prob_col):
        df = df.copy()[["slide", self.time_col, self.event_col, prob_col]]
        df = df.dropna(subset=[self.time_col, self.event_col, prob_col])

        df = df.groupby("slide", as_index=False).agg(
            {
                prob_col: "mean",
                self.event_col: "first",
                self.time_col: "first",
            }
        )
        return df

    def _calculate_HR(self, df, prob_col):
        df = self._prepare_df(df, prob_col)

        if len(df) < 2:
            raise ValueError()

        if df[self.event_col].sum() == 0:
            raise ValueError()

        if df[self.time_col].nunique() < 2:
            raise ValueError()

        threshold = self.threshold
        if threshold is None:
            threshold = df[prob_col].median()

        df = df.copy()
        df["high_risk"] = (df[prob_col] >= threshold).astype(int)

        if df["high_risk"].nunique() < 2:
            return np.nan

        try:
            cph = CoxPHFitter(penalizer=1e-4)
            cph.fit(
                df[[self.time_col, self.event_col, "high_risk"]],
                duration_col=self.time_col,
                event_col=self.event_col,
            )
        except ConvergenceError as e:
            print(e)
            return np.nan

        beta = float(cph.params_["high_risk"])
        hr = float(np.exp(beta))
        return hr

    def calculate(self, df, scanners, per_scanner=False, **kwargs):
        if not (self.event_col in df.columns and self.time_col in df.columns):
            return (
                np.nan if not per_scanner else {scanner: np.nan for scanner in scanners}
            )
        df = df.copy()

        prob_cols = get_prob_cols(df, self.label, self.labels)
        prob_col = prob_cols[-1]
        df = handle_nans(df, [prob_col])

        if per_scanner:
            scanner_dfs = self.get_scanner_dfs(df, scanners)
            CHR_per_scanner = {}
            for scanner, scanner_df in scanner_dfs.items():
                CHR_per_scanner[scanner] = self._calculate_HR(scanner_df, prob_col)

            return CHR_per_scanner

        return self._calculate_HR(df, prob_col)


def get_metric_from_name(metric_name: str, metrics: List) -> Metric:
    for metric in metrics:
        if metric.name == metric_name:
            return metric
    raise ValueError(f"Metric '{metric_name}' not found")
