"""
ballknower_quad.content_utils
=============================

Newsletter content engine — the CFB analog of Gridiron's ``content_utils.py``.
A ``GamePrediction`` dataclass carries the V1 win probability (and optional margin),
a tiering helper turns confidence into Lock/Strong/Lean/Pass, and a set of small
section renderers build a markdown newsletter that mirrors the basketball/football
layout. Each section is its own ``render_*`` function so it's easy to tweak length
and tone.

DISCLAIMER: entertainment/education only — not betting advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ballknower_quad.settings import settings


def confidence_tier(confidence: float) -> str:
    """confidence = 2*|p - 0.5| (0 = coin flip, 1 = certain). Mapped via favorite
    win prob thresholds in settings.confidence_tiers."""
    fav_prob = 0.5 + confidence / 2.0
    for threshold, label in settings.confidence_tiers:
        if fav_prob >= threshold:
            return label
    return "🪙 Pass"


@dataclass
class GamePrediction:
    game_date: str
    home_team: str
    away_team: str

    p_model: float
    p_elo: float
    p_blended: float
    elo_home: float
    elo_away: float

    favorite: str = ""
    fav_prob: float = 0.0
    confidence: float = 0.0
    tier: str = ""

    season: Optional[int] = None
    week: Optional[int] = None
    neutral_site: bool = False
    conference_game: bool = False

    # optional margin (V5-analog) outputs
    predicted_margin: Optional[float] = None
    spread_line: Optional[float] = None
    ats_gap: Optional[float] = None
    ats_pick: Optional[str] = None
    ats_confidence: Optional[float] = None

    extra_notes: List[str] = field(default_factory=list)
    # Feature values behind the pick, for narrative copy. Populated by the
    # pipeline; used to say *why* rather than just *what*.
    drivers: dict = field(default_factory=dict)
    games_played_min: Optional[int] = None

    @classmethod
    def from_probs(cls, *, game_date, home_team, away_team,
                   p_blended, p_model, p_elo, elo_home, elo_away,
                   **kwargs) -> "GamePrediction":
        confidence = abs(p_blended - 0.5) * 2.0
        favorite = home_team if p_blended >= 0.5 else away_team
        fav_prob = max(p_blended, 1 - p_blended)
        return cls(game_date=game_date, home_team=home_team, away_team=away_team,
                   p_model=p_model, p_elo=p_elo, p_blended=p_blended,
                   elo_home=elo_home, elo_away=elo_away,
                   favorite=favorite, fav_prob=fav_prob, confidence=confidence,
                   tier=confidence_tier(confidence), **kwargs)

    def attach_margin(self, predicted_margin: float,
                      spread_line: Optional[float]) -> "GamePrediction":
        self.predicted_margin = predicted_margin
        self.spread_line = spread_line
        if spread_line is not None:
            gap = predicted_margin - spread_line
            self.ats_gap = float(gap)
            self.ats_pick = self.home_team if gap > 0 else self.away_team
            self.ats_confidence = float(abs(gap))
        return self


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def _matchup(p: GamePrediction) -> str:
    sep = "vs" if p.neutral_site else "@"
    return f"{p.away_team} {sep} {p.home_team}"


def render_header(date_label: str) -> str:
    return (f"# 🏟️ BallKnower Quad — College Football, {date_label}\n\n"
            f"_{settings.disclaimer}_\n")


def render_confidence_summary(preds: List[GamePrediction]) -> str:
    if not preds:
        return ""
    locks = [p for p in preds if "Lock" in p.tier]
    strong = [p for p in preds if "Strong" in p.tier]
    return (f"**This week:** {len(preds)} games · {len(locks)} Locks · "
            f"{len(strong)} Strong leans.\n")


def render_top_picks(preds: List[GamePrediction], k: int = 5) -> str:
    ranked = sorted(preds, key=lambda p: p.confidence, reverse=True)[:k]
    lines = ["## 🎯 Top Picks\n"]
    for p in ranked:
        lines.append(f"- **{p.favorite}** to beat "
                     f"{p.away_team if p.favorite == p.home_team else p.home_team} "
                     f"— {p.fav_prob * 100:.0f}% ({p.tier}) · {_matchup(p)}")
    return "\n".join(lines) + "\n"


def margin_label(p: GamePrediction) -> str:
    """Render margin as '<team> by X.X' — a bare '+38.4' doesn't tell the reader
    whose margin it is, and a bare '-1.5' reads like a typo in a newsletter."""
    if p.predicted_margin is None:
        return "—"
    m = p.predicted_margin
    if abs(m) < 0.05:
        return "pick'em"
    team = p.home_team if m > 0 else p.away_team
    return f"{team} by {abs(m):.1f}"


def models_disagree(p: GamePrediction) -> bool:
    """True when the win-prob model and the margin model point at different teams.

    They're separate models, so this is a legitimate (if awkward) disagreement
    rather than a bug — but it should never be published silently.
    """
    if p.predicted_margin is None:
        return False
    margin_pick = p.home_team if p.predicted_margin > 0 else p.away_team
    return margin_pick != p.favorite


def render_full_slate(preds: List[GamePrediction]) -> str:
    lines = ["## 📋 Full Slate\n",
             "| Matchup | Pick | Win % | Tier | Projected margin |",
             "|---|---|---:|---|---|"]
    flagged = 0
    for p in sorted(preds, key=lambda x: x.confidence, reverse=True):
        label = margin_label(p)
        if models_disagree(p):
            label += " ⚠️"
            flagged += 1
        lines.append(f"| {_matchup(p)} | {p.favorite} | "
                     f"{p.fav_prob * 100:.0f}% | {p.tier} | {label} |")
    out = "\n".join(lines) + "\n"
    if flagged:
        out += (f"\n_⚠️ = the win-probability model and the margin model lean "
                f"different ways ({flagged} game{'s' if flagged > 1 else ''}). "
                f"Treat those as true coin flips._\n")
    return out


def render_vegas_disagreement(preds: List[GamePrediction], min_gap: float = 4.0
                              ) -> str:
    flagged = [p for p in preds if p.ats_gap is not None
               and abs(p.ats_gap) >= min_gap]
    if not flagged:
        return ""
    flagged.sort(key=lambda p: abs(p.ats_gap), reverse=True)
    lines = ["## 📈 Where We Disagree With the Number\n",
             "_Interesting reads, not bets._\n"]
    for p in flagged:
        lines.append(f"- **{_matchup(p)}** — model {p.predicted_margin:+.1f} vs "
                     f"line {p.spread_line:+.1f} → leans **{p.ats_pick}** by "
                     f"{abs(p.ats_gap):.1f}")
    return "\n".join(lines) + "\n"


def build_newsletter(preds: List[GamePrediction], date_label: str) -> str:
    parts = [
        render_header(date_label),
        render_confidence_summary(preds),
        render_top_picks(preds),
        render_vegas_disagreement(preds),
        render_full_slate(preds),
        f"\n---\n_{settings.disclaimer}_\n",
    ]
    return "\n".join(p for p in parts if p)


def markdown_to_html(md: str) -> str:
    """Tiny converter (no deps) — good enough for a Substack paste/preview."""
    html, in_table, in_list = [], False, False
    for line in md.splitlines():
        s = line.rstrip()
        if s.startswith("| "):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue  # separator row
            if not in_table:
                html.append("<table>"); in_table = True
            tag = "th" if all(c and not c[0].isdigit() for c in cells) else "td"
            html.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells)
                        + "</tr>")
            continue
        if in_table:
            html.append("</table>"); in_table = False
        if s.startswith("# "):
            html.append(f"<h1>{s[2:]}</h1>")
        elif s.startswith("## "):
            html.append(f"<h2>{s[3:]}</h2>")
        elif s.startswith("- "):
            if not in_list:
                html.append("<ul>"); in_list = True
            html.append(f"<li>{s[2:]}</li>")
        else:
            if in_list:
                html.append("</ul>"); in_list = False
            html.append(f"<p>{s}</p>" if s else "")
    if in_table:
        html.append("</table>")
    if in_list:
        html.append("</ul>")
    out = "\n".join(html)
    # bold
    while "**" in out:
        out = out.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
    return out


# --------------------------------------------------------------------------- #
# Narrative blog post
# --------------------------------------------------------------------------- #
# The newsletter above is a scannable briefing: tiers, tables, a top-picks list.
# This is the other thing — prose a reader can follow end to end, with the
# reasoning attached to each call. Same numbers, different job. Publish one or
# both.

def _driver_phrase(p: "GamePrediction") -> str:
    """One clause explaining what's actually driving the pick."""
    bits = []
    elo_gap = abs(p.elo_home - p.elo_away)
    stronger = p.home_team if p.elo_home >= p.elo_away else p.away_team
    if elo_gap >= 40:
        bits.append(f"a {elo_gap:.0f}-point Elo edge to {stronger}")

    d = p.drivers or {}
    net_ppa = d.get("net_ppa_diff")
    if net_ppa is not None and abs(net_ppa) >= 0.05:
        side = p.home_team if net_ppa > 0 else p.away_team
        bits.append(f"a {abs(net_ppa):.2f} per-play efficiency edge to {side}")

    prior = d.get("preseason_prior_diff")
    if prior is not None and abs(prior) >= 40:
        side = p.home_team if prior > 0 else p.away_team
        bits.append(f"a clear preseason roster edge to {side}")

    if p.p_model is not None and p.p_elo is not None:
        gap = abs(p.p_model - p.p_elo)
        if gap >= 0.10:
            bits.append("the model reading this differently than raw Elo does")

    if not bits:
        return "little separating these two on any measure we track"
    if len(bits) == 1:
        return bits[0]
    return ", ".join(bits[:-1]) + f", and {bits[-1]}"


def _headline_game(preds: List["GamePrediction"]) -> Optional["GamePrediction"]:
    """Best matchup: both teams strong, outcome genuinely in doubt."""
    ranked = [p for p in preds if p.confidence <= 0.45]
    if not ranked:
        ranked = list(preds)
    return max(ranked, key=lambda p: min(p.elo_home, p.elo_away), default=None)


def render_blog(preds: List["GamePrediction"], date_label: str,
                week: Optional[int] = None) -> str:
    if not preds:
        return ""
    week_label = f"Week {week}" if week else date_label
    locks = [p for p in preds if "Lock" in p.tier]
    # A "coin flip" has to actually be close. Taking the three least-confident
    # games regardless of their probability produced a section labelling 70%
    # favourites as toss-ups on a lopsided slate.
    COIN_MAX_FAV = 0.62
    coin = [p for p in sorted(preds, key=lambda p: p.confidence)
            if p.fav_prob <= COIN_MAX_FAV][:3]
    ranked = sorted(preds, key=lambda p: p.confidence, reverse=True)

    out = [f"# {week_label}: What the Model Sees\n",
           f"_{len(preds)} games on the board. "
           f"{len(locks)} the model feels strongly about, "
           f"{len([p for p in preds if p.fav_prob <= COIN_MAX_FAV])} it "
           f"genuinely can't separate._\n"]

    head = _headline_game(preds)
    if head is not None:
        out.append("## The one to watch\n")
        loser = head.away_team if head.favorite == head.home_team else head.home_team
        lean = ("barely a lean" if head.fav_prob < 0.60
                else "a real edge but not a certainty" if head.fav_prob < 0.75
                else "a clear favourite")
        out.append(
            f"**{_matchup(head)}** is the best game on the slate — "
            f"{head.favorite} at {head.fav_prob * 100:.0f}% is {lean}. "
            f"What's driving it: {_driver_phrase(head)}. "
            f"{'A neutral field strips out home advantage entirely. ' if head.neutral_site else ''}"
            f"If you only watch one thing, watch whether {loser} can hang early; "
            f"the model's margin read is "
            f"{margin_label(head) if head.predicted_margin is not None else 'close to even'}.\n")

    strong = [p for p in ranked if p.confidence >= 0.36][:3]
    if strong:
        out.append("## Where the model is confident\n")
        for p in strong:
            other = p.away_team if p.favorite == p.home_team else p.home_team
            out.append(
                f"- **{p.favorite} over {other}** ({p.fav_prob * 100:.0f}%, "
                f"{p.tier}) — {_driver_phrase(p)}.")
        out.append("")

    flagged = [p for p in preds if p.ats_gap is not None
               and abs(p.ats_gap) >= 4.0]
    if flagged:
        out.append("## Where we disagree with the number\n")
        for p in sorted(flagged, key=lambda x: abs(x.ats_gap), reverse=True)[:3]:
            out.append(
                f"- **{_matchup(p)}** — the market says {p.spread_line:+.1f}, "
                f"the model says {p.predicted_margin:+.1f}. That's a "
                f"{abs(p.ats_gap):.1f}-point gap leaning {p.ats_pick}.")
        out.append("\n_These are the interesting ones, not recommendations._\n")

    if coin:
        out.append("## Genuine coin flips\n")
        out.append("Games where the honest answer is that we don't know:\n")
        for p in coin:
            phrase = _driver_phrase(p)
            phrase = phrase[:1].upper() + phrase[1:]   # NOT .capitalize(),
            # which lowercases everything after the first character and turns
            # "a 197-point Elo edge to Alabama" into "...to alabama".
            out.append(f"- **{_matchup(p)}** — {p.favorite} "
                       f"{p.fav_prob * 100:.0f}%. {phrase}.")
        out.append("")

    thin = [p for p in preds if p.games_played_min is not None
            and p.games_played_min < 2]
    if thin:
        out.append("## A note on early-season confidence\n")
        out.append(
            f"{len(thin)} of these teams have almost no {date_label[-4:]} football "
            f"on the books, so their ratings still lean on preseason roster "
            f"signals — recruiting, returning production, transfers — rather than "
            f"results. The model knows this and caps how confident it's allowed "
            f"to be. Ratings firm up considerably by early October.\n")

    out.append("## How to read this\n")
    out.append(
        "Every number here is a probability, not a prediction. A 70% pick is "
        "*supposed* to lose three times in ten — if it never did, the number "
        "would be wrong. What matters over a season isn't the hit rate, it's "
        "whether the 70s land near 70. Every forecast is timestamped and locked "
        "before kickoff, and graded afterward whether it worked or not.\n")
    out.append(f"\n---\n_{settings.disclaimer}_\n")
    return "\n".join(out)
