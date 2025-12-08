from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score



def load_datasets(
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if data_dir is None:
        # locate the data directory two levels up (src/models -> src -> project)
        data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"

    train_path = data_dir / "matches_train.csv"
    val_path = data_dir / "matches_val.csv"
    test_path = data_dir / "matches_test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    # Work on a copy to avoid mutating the original DataFrame
    X = df.copy()
    # Ensure numeric types where possible
    for col in [
        "home_rank",
        "away_rank",
        "home_points",
        "away_points",
        "rank_diff",
        "points_diff",
        "same_confed",
        "neutral",
    ]:
        # Convert to numeric and leave NaN for non-convertible values
        X[col] = pd.to_numeric(X[col], errors="coerce")
    # Select features
    feature_cols = [
        "home_rank",
        "away_rank",
        "home_points",
        "away_points",
        "rank_diff",
        "points_diff",
        "same_confed",
        "neutral",
    ]
    X_feat = X[feature_cols]
    # Impute missing values with column means
    X_feat = X_feat.fillna(X_feat.mean())
    y_home = pd.to_numeric(df["home_score"], errors="coerce").fillna(0).astype(int)
    y_away = pd.to_numeric(df["away_score"], errors="coerce").fillna(0).astype(int)
    return X_feat, y_home, y_away

def train_poisson_models(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    alpha: float = 1e-3,
    max_iter: int = 300,
) -> Tuple[PoissonRegressor, PoissonRegressor, Dict[str, Dict[str, float]]]:
    # Prepare training features and targets
    X_train, y_home_train, y_away_train = _prepare_features(train_df)

    # Fit models
    home_model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
    away_model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
    home_model.fit(X_train, y_home_train)
    away_model.fit(X_train, y_away_train)

    # Determine evaluation data
    if val_df is None:
        X_val, y_home_val, y_away_val = X_train, y_home_train, y_away_train
    else:
        X_val, y_home_val, y_away_val = _prepare_features(val_df)

    # Evaluate regression metrics
    metrics: Dict[str, Dict[str, float]] = {}

    def _eval(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(root_mean_squared_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
        }

    home_pred = home_model.predict(X_val)
    away_pred = away_model.predict(X_val)

    metrics["home"] = _eval(y_home_val, home_pred)
    metrics["away"] = _eval(y_away_val, away_pred)

    true_results = []
    pred_results = []

    for h_true, a_true, h_lam, a_lam in zip(
        y_home_val, y_away_val, home_pred, away_pred
    ):
        # True label from actual goals
        if h_true > a_true:
            true_results.append("home_win")
        elif h_true < a_true:
            true_results.append("away_win")
        else:
            true_results.append("draw")

        # Predicted label from Poisson outcome probabilities
        outcome_probs, _ = scoreline_probabilities(
            home_lambda=h_lam,
            away_lambda=a_lam,
            max_goals=10,
        )
        pred_label = max(outcome_probs, key=outcome_probs.get)
        pred_results.append(pred_label)

    true_results = np.array(true_results)
    pred_results = np.array(pred_results)
    outcome_accuracy = float((true_results == pred_results).mean())

    metrics["outcome"] = {
        "accuracy": outcome_accuracy,
    }

    print("Poisson validation metrics:")
    print(
        f"  Home goals -> MAE={metrics['home']['MAE']:.4f}, "
        f"RMSE={metrics['home']['RMSE']:.4f}, R2={metrics['home']['R2']:.4f}"
    )
    print(
        f"  Away goals -> MAE={metrics['away']['MAE']:.4f}, "
        f"RMSE={metrics['away']['RMSE']:.4f}, R2={metrics['away']['R2']:.4f}"
    )
    print(
        f"  Match outcome accuracy (home/draw/away): "
        f"{metrics['outcome']['accuracy'] * 100:.2f}%"
    )

    return home_model, away_model, metrics

def tune_poisson_alpha(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    alphas=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    max_iter: int = 300,
):
    best_alpha = None
    best_acc = -1.0
    best_models = None
    best_metrics = None

    for a in alphas:
        print(f"\n=== Training Poisson models with alpha={a} ===")
        home_model, away_model, metrics = train_poisson_models(
            train_df=train_df,
            val_df=val_df,
            alpha=a,
            max_iter=max_iter,
        )
        acc = metrics["outcome"]["accuracy"]
        print(f"  Outcome accuracy: {acc * 100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_alpha = a
            best_models = (home_model, away_model)
            best_metrics = metrics

    print(f"\n>>> Best alpha: {best_alpha} with outcome accuracy {best_acc * 100:.2f}%")
    return best_models, best_alpha, best_metrics



def _aggregate_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = {}
    for team in pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True)):
        # Filter matches involving the team either as home or away
        team_home = df[df["home_team"] == team]
        team_away = df[df["away_team"] == team]
        # Concatenate numeric columns for averaging
        ranks = pd.concat([
            team_home["home_rank"],
            team_away["away_rank"],
        ], ignore_index=True).dropna()
        points = pd.concat([
            team_home["home_points"],
            team_away["away_points"],
        ], ignore_index=True).dropna()
        confeds = pd.concat([
            team_home["home_confed"],
            team_away["away_confed"],
        ], ignore_index=True).dropna()
        # Compute averages or defaults
        avg_rank = float(ranks.mean()) if not ranks.empty else np.nan
        avg_points = float(points.mean()) if not points.empty else np.nan
        # Choose the most common confederation; if none exists assign empty string
        confed = confeds.mode().iloc[0] if not confeds.empty else ""
        stats[team] = {
            "rank": avg_rank,
            "points": avg_points,
            "confed": confed,
        }
    return pd.DataFrame.from_dict(stats, orient="index")


@dataclass
class MatchPrediction:
    home_expected_goals: float
    away_expected_goals: float
    outcome_probabilities: Dict[str, float]
    scoreline_matrix: np.ndarray


def scoreline_probabilities(
    home_lambda: float,
    away_lambda: float,
    max_goals: int = 10,
) -> Tuple[Dict[str, float], np.ndarray]:
    # Probability mass for goals 0..max_goals
    home_dist = poisson.pmf(np.arange(max_goals + 1), home_lambda)
    away_dist = poisson.pmf(np.arange(max_goals + 1), away_lambda)
    # Outer product yields matrix of joint probabilities
    matrix = np.outer(home_dist, away_dist)
    # Sum matrix entries to compute outcome probabilities
    prob_home_win = float(np.tril(matrix, -1).sum())
    prob_draw = float(np.trace(matrix))
    prob_away_win = float(np.triu(matrix, 1).sum())
    return {
        "home_win": prob_home_win,
        "draw": prob_draw,
        "away_win": prob_away_win,
    }, matrix


def predict_match(
    home_team: str,
    away_team: str,
    df_reference: pd.DataFrame,
    home_model: PoissonRegressor,
    away_model: PoissonRegressor,
    neutral: int = 0,
    max_goals: int = 10,
) -> MatchPrediction:

    # Compute aggregated stats for teams
    stats_table = _aggregate_team_stats(df_reference)
    if home_team not in stats_table.index:
        raise ValueError(f"Home team '{home_team}' not found in reference data.")
    if away_team not in stats_table.index:
        raise ValueError(f"Away team '{away_team}' not found in reference data.")
    home_stats = stats_table.loc[home_team]
    away_stats = stats_table.loc[away_team]
    # Build feature vector matching _prepare_features
    features = pd.DataFrame([
        {
            "home_rank": home_stats["rank"],
            "away_rank": away_stats["rank"],
            "home_points": home_stats["points"],
            "away_points": away_stats["points"],
            "rank_diff": away_stats["rank"] - home_stats["rank"],
            "points_diff": home_stats["points"] - away_stats["points"],
            "same_confed": int(home_stats["confed"] == away_stats["confed"]),
            "neutral": neutral,
        }
    ])
    # Handle missing values (e.g., NaN ranks) by imputing with column means
    features = features.fillna(features.mean())
    # Predict expected goals
    home_lambda = float(home_model.predict(features)[0])
    away_lambda = float(away_model.predict(features)[0])
    # Convert expected goals into outcome probabilities and scoreline matrix
    outcome_probs, matrix = scoreline_probabilities(home_lambda, away_lambda, max_goals)
    return MatchPrediction(
        home_expected_goals=home_lambda,
        away_expected_goals=away_lambda,
        outcome_probabilities=outcome_probs,
        scoreline_matrix=matrix,
    )