"""
Statistical comparison and modeling for the WC passing network project.

Three analytical layers:
  1. Statistical tests  – compare quarterfinalists vs. eliminated teams
  2. Classification     – logistic regression + random forest to predict
                          reached_quarterfinal at match and team level
  3. Advancement profiles – cluster teams by network signature
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        classification_report,
        roc_auc_score,
        ConfusionMatrixDisplay,
    )
    from sklearn.model_selection import (
        cross_val_score,
        StratifiedKFold,
        LeaveOneOut,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# 1. Statistical comparison
# ---------------------------------------------------------------------------

def compare_groups(
    df: pd.DataFrame,
    features: List[str],
    group_col: str = "reached_quarterfinal",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Mann-Whitney U test for each feature between group=1 and group=0.

    Returns a DataFrame with effect size (rank-biserial correlation),
    p-value, and significance flag sorted by p-value.
    """
    group1 = df[df[group_col] == 1]
    group0 = df[df[group_col] == 0]

    rows = []
    for feat in features:
        s1 = group1[feat].dropna()
        s0 = group0[feat].dropna()
        if len(s1) < 2 or len(s0) < 2:
            continue
        stat, p = stats.mannwhitneyu(s1, s0, alternative="two-sided")
        n1, n0 = len(s1), len(s0)
        # Rank-biserial correlation as effect size
        rbc = 1 - (2 * stat) / (n1 * n0)
        rows.append({
            "feature": feat,
            "mean_qf": s1.mean(),
            "mean_elim": s0.mean(),
            "difference": s1.mean() - s0.mean(),
            "effect_size_rbc": rbc,
            "p_value": p,
            "significant": p < alpha,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("p_value")
        .reset_index(drop=True)
    )


def descriptive_stats(
    df: pd.DataFrame,
    features: List[str],
    group_col: str = "reached_quarterfinal",
) -> pd.DataFrame:
    """Return mean ± std for each feature by group."""
    rows = []
    for feat in features:
        for group_val, label in [(1, "QF+"), (0, "Eliminated")]:
            vals = df[df[group_col] == group_val][feat].dropna()
            rows.append({
                "feature": feat,
                "group": label,
                "n": len(vals),
                "mean": vals.mean(),
                "std": vals.std(),
                "median": vals.median(),
                "q25": vals.quantile(0.25),
                "q75": vals.quantile(0.75),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Classification
# ---------------------------------------------------------------------------

FEATURE_SETS = {
    "network_only": [
        "network_density_mean",
        "degree_centralization_mean",
        "top_player_pass_share_mean",
        "clustering_coefficient_mean",
        "mean_betweenness_centrality_mean",
    ],
    "pass_quality": [
        "progressive_pass_share_mean",
        "final_third_entry_share_mean",
        "centre_channel_share_mean",
        "lateral_imbalance_mean",
    ],
    "combined": [
        "network_density_mean",
        "degree_centralization_mean",
        "top_player_pass_share_mean",
        "clustering_coefficient_mean",
        "mean_betweenness_centrality_mean",
        "progressive_pass_share_mean",
        "final_third_entry_share_mean",
        "centre_channel_share_mean",
        "lateral_imbalance_mean",
    ],
}


def _build_pipelines() -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            )),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )),
        ]),
    }


def evaluate_classifiers(
    df: pd.DataFrame,
    feature_set: str = "combined",
    target: str = "reached_quarterfinal",
    cv_strategy: str = "loo",
) -> pd.DataFrame:
    """
    Evaluate classifiers using cross-validation on the team-level feature table.

    Parameters
    ----------
    df           : team-level feature table
    feature_set  : key in FEATURE_SETS dict or list of column names
    target       : column to predict
    cv_strategy  : 'loo' (Leave-One-Out, good for n=32) or 'stratified5'

    Returns a DataFrame of model × metric results.
    """
    if isinstance(feature_set, str):
        features = [f for f in FEATURE_SETS.get(feature_set, []) if f in df.columns]
    else:
        features = [f for f in feature_set if f in df.columns]

    clean = df[features + [target]].dropna()
    X = clean[features].values
    y = clean[target].values.astype(int)

    if cv_strategy == "loo":
        cv = LeaveOneOut()
        scoring = "roc_auc"
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = "roc_auc"

    rows = []
    for model_name, pipeline in _build_pipelines().items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
        rows.append({
            "model": model_name,
            "feature_set": feature_set if isinstance(feature_set, str) else "custom",
            "cv_strategy": cv_strategy,
            "roc_auc_mean": scores.mean(),
            "roc_auc_std": scores.std(),
            "n_features": len(features),
            "n_samples": len(y),
        })

    return pd.DataFrame(rows)


def fit_and_report(
    df: pd.DataFrame,
    feature_set: str = "combined",
    target: str = "reached_quarterfinal",
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """
    Fit the best model (Random Forest) on all data and return:
      - fitted pipeline
      - feature importance DataFrame
      - per-team prediction DataFrame
    """
    if isinstance(feature_set, str):
        features = [f for f in FEATURE_SETS.get(feature_set, []) if f in df.columns]
    else:
        features = list(feature_set)

    clean = df[features + [target, "team"]].dropna()
    X = clean[features].values
    y = clean[target].values.astype(int)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    pipeline.fit(X, y)

    importances = pd.DataFrame({
        "feature": features,
        "importance": pipeline.named_steps["clf"].feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    proba = pipeline.predict_proba(X)[:, 1]
    predictions = clean[["team", target]].copy()
    predictions["predicted_proba"] = proba
    predictions["predicted_label"] = (proba >= 0.5).astype(int)

    return pipeline, importances, predictions


# ---------------------------------------------------------------------------
# 3. Advancement profiles (clustering)
# ---------------------------------------------------------------------------

def cluster_teams(
    df: pd.DataFrame,
    feature_set: str = "combined",
    n_clusters: int = 3,
) -> pd.DataFrame:
    """
    K-means cluster teams by passing network signature.

    Returns the input DataFrame with 'cluster' and PCA coordinates added.
    """
    features = [f for f in FEATURE_SETS.get(feature_set, []) if f in df.columns]
    clean = df[features + ["team"]].dropna()
    X = StandardScaler().fit_transform(clean[features].values)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clean = clean.copy()
    clean["cluster"] = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    clean["pca1"] = coords[:, 0]
    clean["pca2"] = coords[:, 1]

    result = df.merge(
        clean[["team", "cluster", "pca1", "pca2"]], on="team", how="left"
    )

    explained = pca.explained_variance_ratio_
    print(f"PCA variance explained: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}")

    return result


def profile_clusters(
    df: pd.DataFrame,
    feature_set: str = "combined",
) -> pd.DataFrame:
    """Return mean feature values per cluster alongside QF advancement rate."""
    features = [f for f in FEATURE_SETS.get(feature_set, []) if f in df.columns]
    if "cluster" not in df.columns:
        raise ValueError("Run cluster_teams() first.")
    grouped = df.groupby("cluster")[features + ["reached_quarterfinal"]].mean()
    grouped["n_teams"] = df.groupby("cluster")["team"].count()
    return grouped.reset_index()
