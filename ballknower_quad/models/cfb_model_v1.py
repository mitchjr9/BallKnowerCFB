"""
ballknower_quad.models.cfb_model_v1
===================================

V1 win-probability model — the college-football analog of Gridiron's production
**V3.1** classifier. An XGBoost binary classifier on the leakage-free feature
matrix, probability-calibrated, then **blended with the Elo baseline** (default
weight ``settings.default_blend_elo``) exactly the way Gridiron blended Elo and
the NFL project blended Vegas: the tree model captures interactions, Elo provides
a stable, well-calibrated anchor, and the blend tames overconfident leaves.

We calibrate with isotonic regression on a held-out tail of the (time-ordered)
training data rather than random CV folds, so calibration respects time order and
doesn't leak future games backward.

DISCLAIMER: entertainment/education only — not betting advice.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from ballknower_quad.models.cfb_elo import CFBElo
from ballknower_quad.models.feature_builder import FeatureBuilder, columns_for
from ballknower_quad.settings import settings


def _elo_prob_from_diff(elo_diff_with_hca: float) -> float:
    return 1.0 / (1.0 + 10 ** (-elo_diff_with_hca / 400.0))


@dataclass
class CFBModelV1:
    clf: XGBClassifier
    calibrator: Optional[IsotonicRegression]
    feature_columns: List[str]
    blend_elo: float
    trained_at: str
    metrics: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def _raw_prob(self, X: pd.DataFrame) -> np.ndarray:
        p = self.clf.predict_proba(X[self.feature_columns])[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 1e-4, 1 - 1e-4)

    def predict_proba(self, X: pd.DataFrame, blend_elo: Optional[float] = None
                      ) -> pd.DataFrame:
        """Return a frame with model, Elo-baseline, and blended home-win probs."""
        w = self.blend_elo if blend_elo is None else blend_elo
        p_model = self._raw_prob(X)
        p_elo = X["elo_diff_with_hca"].map(_elo_prob_from_diff).to_numpy()
        p_blend = (1 - w) * p_model + w * p_elo
        return pd.DataFrame({
            "p_model": p_model, "p_elo": p_elo, "p_blended": p_blend,
        }, index=X.index)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.clf.save_model(model_dir / "xgb_clf.json")
        meta = {
            "feature_columns": self.feature_columns,
            "blend_elo": self.blend_elo,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
            "has_calibrator": self.calibrator is not None,
        }
        (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if self.calibrator is not None:
            np.savez(model_dir / "calibrator.npz",
                     x=self.calibrator.X_thresholds_,
                     y=self.calibrator.y_thresholds_)

    @classmethod
    def load(cls, model_dir: Path) -> "CFBModelV1":
        model_dir = Path(model_dir)
        clf = XGBClassifier()
        clf.load_model(model_dir / "xgb_clf.json")
        meta = json.loads((model_dir / "meta.json").read_text())
        calibrator = None
        if meta.get("has_calibrator") and (model_dir / "calibrator.npz").exists():
            d = np.load(model_dir / "calibrator.npz")
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(d["x"], d["y"])
        return cls(clf=clf, calibrator=calibrator,
                   feature_columns=meta["feature_columns"],
                   blend_elo=meta["blend_elo"], trained_at=meta["trained_at"],
                   metrics=meta.get("metrics", {}))


# ---------------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------------- #
def train_v1(games: pd.DataFrame, *, blend_elo: Optional[float] = None,
             calibrate: bool = True, version: str = "v1",
             game_ppa: Optional[pd.DataFrame] = None,
             season_ppa: Optional[pd.DataFrame] = None,
             preseason: Optional[pd.DataFrame] = None
             ) -> tuple[CFBModelV1, FeatureBuilder]:
    """
    Fit a win-probability model. Returns (model, warm FeatureBuilder) — the
    builder carries the final Elo (+ PPA, for v1.1) state for inference.

    `version` selects the feature set: "v1" is Elo + rolling margin + schedule;
    "v1.1" adds the opponent-adjusted PPA block, which needs `game_ppa` and
    (ideally) `season_ppa` for the prior-season anchor.
    """
    cols = columns_for(version)
    fb = FeatureBuilder(version=version, game_ppa=game_ppa,
                        season_ppa=season_ppa, preseason=preseason)
    X, y_win, _ = fb.build(games, fit_elo=True)

    mask = y_win.notna()
    Xc, yc = X[mask].reset_index(drop=True), y_win[mask].astype(int).reset_index(
        drop=True)

    # time-ordered split: last `test_size` for calibration + quick metrics
    n = len(Xc)
    cut = int(n * (1 - settings.test_size))
    X_tr, y_tr = Xc.iloc[:cut], yc.iloc[:cut]
    X_te, y_te = Xc.iloc[cut:], yc.iloc[cut:]

    clf = XGBClassifier(**settings.xgb_clf_params,
                        random_state=settings.random_state)
    clf.fit(X_tr[cols], y_tr)

    calibrator = None
    if calibrate and len(X_te) >= 50:
        p_raw = clf.predict_proba(X_te[cols])[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(p_raw, y_te.to_numpy())

    w = settings.default_blend_elo if blend_elo is None else blend_elo
    model = CFBModelV1(clf=clf, calibrator=calibrator,
                       feature_columns=cols, blend_elo=w,
                       trained_at=datetime.now(timezone.utc).isoformat(
                           timespec="seconds"))

    # quick holdout metrics (honest; full walk-forward lives in backtest/)
    if len(X_te):
        probs = model.predict_proba(X_te)["p_blended"].to_numpy()
        yt = y_te.to_numpy()
        eps = 1e-9
        model.metrics = {
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "accuracy": float(((probs >= 0.5).astype(int) == yt).mean()),
            "brier": float(np.mean((probs - yt) ** 2)),
            "log_loss": float(-np.mean(yt * np.log(probs + eps)
                                       + (1 - yt) * np.log(1 - probs + eps))),
            "home_win_rate": float(yt.mean()),
        }
    return model, fb
