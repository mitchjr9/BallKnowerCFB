"""
ballknower_quad.models.cfb_margin_v1
====================================

V1 margin model — the analog of Gridiron's **V5** spread model. An XGBoost
regressor on the same leakage-free feature matrix, targeting point margin
(home − away). Outputs a predicted margin and, when a Vegas spread is supplied,
an against-the-spread (ATS) read.

CFBD's ``/lines`` endpoint reports spreads as the **home team's expected margin**
(positive = home favored) — the *same convention as nflverse* ``spread_line``,
not the traditional bettor convention. So the cover signal is
``predicted_margin − spread_line`` (we carry forward the sign fix the Gridiron
leakage audit landed on). A positive gap = the model likes the home side to beat
the number.

As in Gridiron, the margin model is **optional** everywhere downstream — the win-
probability model stands on its own and the pipeline degrades gracefully if this
isn't trained.

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
from xgboost import XGBRegressor

from ballknower_quad.models.feature_builder import FeatureBuilder, columns_for
from ballknower_quad.settings import settings


@dataclass
class CFBMarginV1:
    reg: XGBRegressor
    feature_columns: List[str]
    trained_at: str
    metrics: Dict[str, float] = field(default_factory=dict)

    def predict_margin(self, X: pd.DataFrame) -> np.ndarray:
        return self.reg.predict(X[self.feature_columns])

    def ats(self, predicted_margin: float, spread_line: Optional[float],
            home: str, away: str) -> dict:
        """ATS read. spread_line uses CFBD/nflverse convention (+ = home favored)."""
        if spread_line is None:
            return {"ats_gap": None, "ats_pick": None, "ats_confidence": None}
        gap = predicted_margin - spread_line          # sign fix from Gridiron audit
        pick = home if gap > 0 else away
        return {"ats_gap": float(gap), "ats_pick": pick,
                "ats_confidence": float(abs(gap))}

    # ------------------------------------------------------------------ #
    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.reg.save_model(model_dir / "xgb_reg.json")
        (model_dir / "margin_meta.json").write_text(json.dumps({
            "feature_columns": self.feature_columns,
            "trained_at": self.trained_at, "metrics": self.metrics,
        }, indent=2))

    @classmethod
    def load(cls, model_dir: Path) -> "CFBMarginV1":
        model_dir = Path(model_dir)
        reg = XGBRegressor()
        reg.load_model(model_dir / "xgb_reg.json")
        meta = json.loads((model_dir / "margin_meta.json").read_text())
        return cls(reg=reg, feature_columns=meta["feature_columns"],
                   trained_at=meta["trained_at"], metrics=meta.get("metrics", {}))


def train_margin_v1(games: pd.DataFrame,
                    fb: Optional[FeatureBuilder] = None,
                    *, version: str = "v1",
                    game_ppa: Optional[pd.DataFrame] = None,
                    season_ppa: Optional[pd.DataFrame] = None,
                    preseason: Optional[pd.DataFrame] = None) -> CFBMarginV1:
    """Fit the margin regressor on the feature set for `version`.

    Always builds on a fresh FeatureBuilder so the returned y_margin is aligned
    and the caller's warm state is left undisturbed.
    """
    cols = columns_for(version)
    X, _, y_margin = FeatureBuilder(
        version=version, game_ppa=game_ppa, season_ppa=season_ppa,
        preseason=preseason
    ).build(games, fit_elo=True)

    mask = y_margin.notna()
    Xc = X[mask].reset_index(drop=True)
    yc = y_margin[mask].reset_index(drop=True)

    n = len(Xc)
    cut = int(n * (1 - settings.test_size))
    X_tr, y_tr = Xc.iloc[:cut], yc.iloc[:cut]
    X_te, y_te = Xc.iloc[cut:], yc.iloc[cut:]

    reg = XGBRegressor(**settings.xgb_reg_params, random_state=settings.random_state)
    reg.fit(X_tr[cols], y_tr)

    model = CFBMarginV1(reg=reg, feature_columns=cols,
                        trained_at=datetime.now(timezone.utc).isoformat(
                            timespec="seconds"))
    if len(X_te):
        pred = reg.predict(X_te[cols])
        err = pred - y_te.to_numpy()
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((y_te.to_numpy() - y_te.mean()) ** 2)) or 1.0
        model.metrics = {
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1 - ss_res / ss_tot),
        }
    return model
