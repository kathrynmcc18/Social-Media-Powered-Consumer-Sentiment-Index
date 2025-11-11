# -*- coding: utf-8 -*-
"""
Social Media Powered Consumer Sentiment Index (MCSI) Model
By Kathryn McCarthy
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Union
import warnings, os, re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tempfile


try:
    from transformers import pipeline
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

try:
    from huggingface_hub import login as hf_login
    _HF_HUB_AVAILABLE = True
except Exception:
    _HF_HUB_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except Exception:
    _VADER_AVAILABLE = False

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# === CONFIG ===
HF_MODEL_PRIMARY   = "cardiffnlp/twitter-roberta-base-sentiment-latest"
HF_MODEL_FALLBACK  = "distilroberta-base-finetuned-sst-2-english"

IMPRESSIONS_CAP = 1_000_000.0
USE_LOG_WEIGHTS = True
DAILY_MIN_POSTS = 5
ROLLING_DAYS = 5
OUTPUT_DIR = "./nowcast_outputs"

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 50)

# === Helper Functions ===
def _ensure_datetime_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC")
    return ts

def _safe_log1p(x: pd.Series, cap: Optional[float] = None) -> pd.Series:
    if cap is not None:
        x = x.clip(upper=cap)
    return np.log1p(x.fillna(0))

def _zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


# === UMich CSI Windows and Data ===
UMICH_WINDOWS_2025 = {
    "September 2024": ("2024-08-27", "2024-09-09", "2024-09-23"),
    "October 2024":   ("2024-09-24", "2024-10-07", "2024-10-21"),
    "November 2024":  ("2024-10-22", "2024-11-05", "2024-11-18"),
    "December 2024":  ("2024-11-19", "2024-12-02", "2024-12-16"),
    "January 2025": ("2024-12-17", "2025-01-06", "2025-01-20"),
    "February 2025": ("2025-01-21", "2025-02-04", "2025-02-17"),
    "March 2025": ("2025-02-18", "2025-03-10", "2025-03-24"),
    "April 2025": ("2025-03-25", "2025-04-08", "2025-04-21"),
    "May 2025": ("2025-04-22", "2025-05-13", "2025-05-26"),
    "June 2025": ("2025-05-27", "2025-06-09", "2025-06-23"),
    "July 2025": ("2025-06-24", "2025-07-14", "2025-07-28"),
    "August 2025": ("2025-07-29", "2025-08-11", "2025-08-25"),
    "September 2025": ("2025-08-26", "2025-09-08", "2025-09-22"),
    "October 2025": ("2025-09-22", "2025-10-06", "2025-10-20"),
    "November 2025": ("2025-10-21", "2025-11-03", "2025-11-17"),
    "December 2025": ("2025-11-18", "2025-12-01", "2025-12-15"),
}

def get_umich_windows_2025() -> pd.DataFrame:
    rows = []
    for month, (start, prelim_end, final_end) in UMICH_WINDOWS_2025.items():
        rows.append({
            "month": month,
            "window_start": pd.to_datetime(start, utc=True),
            "prelim_end": pd.to_datetime(prelim_end, utc=True),
            "final_end": pd.to_datetime(final_end, utc=True),
        })
    return pd.DataFrame(rows).sort_values("window_start").reset_index(drop=True)

def load_michigan_from_pdf_table_2024_2025() -> pd.DataFrame:
    data = [
        ("September 2024", "2024-09-30", 70.1),
        ("October 2024",   "2024-10-31", 70.5),
        ("November 2024",  "2024-11-30", 71.8),
        ("December 2024",  "2024-12-31", 74.0),
        ("January 2025",   "2025-01-31", 71.7),
        ("February 2025",  "2025-02-28", 64.7),
        ("March 2025",     "2025-03-31", 57.0),
        ("April 2025",     "2025-04-30", 52.2),
        ("May 2025",       "2025-05-31", 52.2),
        ("June 2025",      "2025-06-30", 60.7),
        ("July 2025",      "2025-07-31", 61.7),
        ("August 2025",    "2025-08-31", 58.2),
        ("September 2025", "2025-09-30", 55.1),
        ("October 2025", "2025-10-31", 53.6),
        ("November 2025", "2025-11-28", 50.3),

    ]
    df = pd.DataFrame(data, columns=["month","period_end","csi"])
    df["period_end"] = pd.to_datetime(df["period_end"], utc=True)
    return df


# === Sentiment Scorer ===
@dataclass
class HFConfig:
    model_name: str = HF_MODEL_PRIMARY
    device: int = -1

def _norm_label(lbl: str) -> str:
    u = (lbl or "").upper()
    if u in ("NEGATIVE", "LABEL_0"): return "NEG"
    if u in ("NEUTRAL", "LABEL_1"): return "NEU"
    if u in ("POSITIVE", "LABEL_2"): return "POS"
    return u

class HFSentimentScorer:
    def __init__(self, hf_config: HFConfig = HFConfig(),
                 model_path_or_id: Optional[str] = None,
                 token: Optional[str] = None,
                 local_files_only: bool = False,
                 max_chars: int = 1000,
                 batch_size: int = 16):
        self.hf_config = hf_config
        self.max_chars = max_chars
        self.batch_size = batch_size
        self._pipe = None
        self._vader = None

        # Try VADER
        if _VADER_AVAILABLE:
            try:
                try:
                    nltk.data.find('sentiment/vader_lexicon.zip')
                except LookupError:
                    nltk.download('vader_lexicon')
                self._vader = SentimentIntensityAnalyzer()
            except Exception as e:
                warnings.warn(f"VADER unavailable: {e}")

        # HuggingFace login
        if token and _HF_HUB_AVAILABLE:
            try:
                hf_login(token=token)
            except Exception:
                pass

        # Try HuggingFace pipeline
        if _TRANSFORMERS_AVAILABLE:
            try:
                self._pipe = pipeline(
                    "sentiment-analysis",
                    model=model_path_or_id or self.hf_config.model_name,
                    device=self.hf_config.device,
                    top_k=None,
                    function_to_apply="softmax",
                    truncation=True,
                    max_length=512,
                )
                _ = self._pipe(["ok"])
            except Exception as e:
                warnings.warn(f"HF pipeline unavailable ({e}); using VADER only.")
                self._pipe = None

        if self._pipe is None and self._vader is None:
            raise RuntimeError("No sentiment scorer available.")

    def _clean(self, t: str) -> str:
        if t is None: return ""
        t = str(t).replace("\x00", " ")
        t = re.sub(r"\s+", " ", t).strip()
        return t[: self.max_chars] if len(t) > self.max_chars else t

    def score_texts(self, texts: List[str]) -> np.ndarray:
        texts = [self._clean(t) for t in texts]
        out = np.zeros(len(texts), dtype=float)
        if self._pipe is not None:
            i = 0
            while i < len(texts):
                chunk = texts[i:i+self.batch_size]
                try:
                    results = self._pipe(chunk, truncation=True)
                    for j, r in enumerate(results):
                        probs = {"NEG": 0.0, "NEU": 0.0, "POS": 0.0}
                        for d in r:
                            probs[_norm_label(d.get("label",""))] = float(d.get("score", 0.0))
                        out[i+j] = probs["POS"] - probs["NEG"]
                except Exception:
                    out[i:i+len(chunk)] = 0.0
                i += self.batch_size
            return out
        if self._vader is not None:
            return np.array([self._vader.polarity_scores(t)["compound"] for t in texts])
        return out


# === Model Pipeline ===
@dataclass
class PipelineConfig:
    impressions_cap: Optional[float] = 1_000_000.0
    weight_log: bool = True
    daily_min_posts: int = 10
    min_days_per_window: int = 5
    rolling_days: int = 7
    topic_filter: Optional[List[str]] = None
    platform_filter: Optional[List[str]] = None
    outdir: str = OUTPUT_DIR

class CSINowcaster:
    def __init__(self, scorer: HFSentimentScorer, cfg: PipelineConfig):
        self.scorer = scorer
        self.cfg = cfg
        self.reg = LinearRegression()
        self.fitted = False

    def load_social(self, path: str, date_col="upload_date", text_col="snippet_text",
                    impressions_col="impression_count_comb", platform_col=None, topic_col="company") -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        df = pd.read_excel(path) if ext in [".xlsx", ".xls"] else pd.read_csv(path)
        df = df.rename(columns={date_col:"timestamp", text_col:"text", impressions_col:"impressions"})
        if platform_col and platform_col in df.columns:
            df = df.rename(columns={platform_col:"platform"})
        else:
            df["platform"] = None
        if topic_col and topic_col in df.columns:
            df = df.rename(columns={topic_col:"topic"})
        else:
            df["topic"] = None
        df["timestamp"] = _ensure_datetime_utc(df["timestamp"])
        df = df.dropna(subset=["timestamp","text"])
        if self.cfg.topic_filter:
            df = df[df["topic"].isin(self.cfg.topic_filter)]
        return df

    def score_social(self, social_df: pd.DataFrame) -> pd.DataFrame:
        df = social_df.copy()
        df["sentiment"] = self.scorer.score_texts(df["text"].astype(str).tolist())
        df["weight"] = _safe_log1p(df["impressions"], cap=self.cfg.impressions_cap)
        return df

    def daily_index(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        df = scored_df.copy()
        df["date"] = df["timestamp"].dt.date
        grp = df.groupby("date").apply(
            lambda g: pd.Series({
                "n_posts": len(g),
                "weighted_mean": np.average(g["sentiment"], weights=(g["weight"] + 1e-9))
            })
        ).reset_index()
        grp = grp[grp["n_posts"] >= self.cfg.daily_min_posts]
        grp["daily_index"] = grp["weighted_mean"].rolling(self.cfg.rolling_days, min_periods=1).mean()
        grp["daily_index_z"] = _zscore(grp["daily_index"])
        return grp

    def aggregate_by_umich_window(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        windows = get_umich_windows_2025()
        d = daily_df.copy()
        d["date_ts"] = pd.to_datetime(d["date"])
        def assign_label(dt):
            d0 = pd.to_datetime(dt).date()
            for _, row in windows.iterrows():
                ws, we = pd.to_datetime(row["window_start"]).date(), pd.to_datetime(row["final_end"]).date()
                if ws <= d0 <= we:
                    return row["month"]
            return None
        d["umich_month"] = d["date_ts"].apply(assign_label)
        agg = d.dropna(subset=["umich_month"]).groupby("umich_month").agg(
            social_window_index=("daily_index_z","mean"),
            days_in_window=("umich_month","count")
        ).reset_index().rename(columns={"umich_month":"month"})
        return agg.merge(windows, on="month", how="left").sort_values("window_start")

    def fit_mapping(self, social_path, michigan_path_or_df, umich_year=2025,
                    save_outputs=True, return_artifacts=True):
        social_df = self.load_social(social_path)
        scored = self.score_social(social_df)
        daily = self.daily_index(scored)
        windows_agg = self.aggregate_by_umich_window(daily,)

        if isinstance(michigan_path_or_df, pd.DataFrame):
            mich = michigan_path_or_df.copy()
        else:
            mich = pd.read_csv(michigan_path_or_df)

        joint = windows_agg.merge(mich[["month","csi"]], on="month", how="inner")
        X, y = joint[["social_window_index"]].values, joint["csi"].values
        self.reg.fit(X, y)
        yhat = self.reg.predict(X)

        html_path = os.path.join(self.cfg.outdir, "nowcast_plot.html")
        if save_outputs:
            os.makedirs(self.cfg.outdir, exist_ok=True)
            daily.to_csv(os.path.join(self.cfg.outdir, "daily_social_index.csv"), index=False)
            joint.assign(predicted_csi=yhat).to_csv(
                os.path.join(self.cfg.outdir, "window_nowcast.csv"), index=False
            )
            plot_nowcast_plotly(joint.assign(predicted_csi=yhat), html_path)

        out = {
            "metrics": {"mae": float(mean_absolute_error(y, yhat)),
                        "r2": float(r2_score(y, yhat))},
            "daily": daily,
            "window_join": joint.assign(predicted_csi=yhat),
            "plot_path": html_path
        }
        return out


# === Plotly Visualization ===
def plot_nowcast_plotly(joint_df, output_path=None):
    """
    Creates and saves the interactive Plotly chart comparing
    the University of Michigan CSI vs the Social Media Powered CSI (fitted).
    Uses a temporary directory when deployed to Streamlit Cloud.
    """
    # --- Ensure output path is safe and writable ---
    if output_path is None:
        tmpdir = tempfile.gettempdir()        # ✅ portable, writable directory
        output_path = os.path.join(tmpdir, "nowcast_plot.html")

    # --- Create Plotly Figure ---
    fig = go.Figure()

    # --- Official Michigan CSI ---
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(joint_df["final_end"]),
        y=joint_df["csi"],
        mode="lines+markers",
        name="UMich CSI (official)",
        line=dict(color="#1f77b4", width=2)
    ))

    # --- Social Media CSI (fitted) ---
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(joint_df["final_end"]),
        y=joint_df["predicted_csi"],
        mode="lines+markers",
        name="Social Media CSI (fitted)",
        line=dict(color="#ff7f0e", width=2)
    ))

    # --- Layout & styling ---
    fig.update_layout(
        title=dict(
            text="University of Michigan CSI vs. Social Media Powered CSI",
            font=dict(color="#FAFAFA", size=20),
            x=0.5
        ),
        xaxis_title="Date",
        yaxis_title="CSI",
        template="plotly_dark",  # 🌑 dark mode
        hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all")
                ]
            ),
            rangeslider=dict(
                visible=True,
                bgcolor="rgba(40,40,40,0.7)",
                thickness=0.08
            ),
            type="date"
        ),
        autosize=True,
        height=550,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#FAFAFA")
        ),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117"
    )

    # --- Ensure directory exists ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Save interactive HTML plot ---
    fig.write_html(output_path, include_plotlyjs="cdn")

    print(f"✅ Plot successfully saved at: {output_path}")
    return output_path


# === High-Level Run Function for Streamlit ===
def run_mcsi_model(filtered_path, selected_topics):
    print("=== Starting MCSI Model Run ===")
    print(f"Input file: {filtered_path}")
    print(f"Topics selected: {selected_topics}")

    scorer = HFSentimentScorer(HFConfig())
    cfg = PipelineConfig(topic_filter=selected_topics)
    nowcaster = CSINowcaster(scorer, cfg)

    artifacts = nowcaster.fit_mapping(
        social_path=filtered_path,
        michigan_path_or_df=load_michigan_from_pdf_table_2024_2025(),
        umich_year=2025
    )
    print("✅ Model run complete.")
    return artifacts






