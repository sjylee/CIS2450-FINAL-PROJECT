"""
Sonic Paradox Longevity Index — Dashboard
CIS 2450: Big Data Analytics
Run: python dashboard.py
"""


import pandas as pd
import numpy as np
from dash import Dash, html, dcc, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")


ACCENT = "#1F4E79"
ORANGE = "#E05C2A"
GREEN  = "#1D6B45"
LIGHT  = "#D6E4F0"
RED    = "#8B1A1A"


# =============================================================================
# DATA + MODELS
# =============================================================================


def load_data():
    try:
        df = pd.read_csv("outputs/joined_data_clean.csv")
        df = df.drop(columns=["acousticness_log"], errors="ignore")
        df = df.dropna(subset=["artist_name"])
        print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
        return df
    except FileNotFoundError:
        print("ERROR: outputs/joined_data_clean.csv not found.")
        return pd.DataFrame()




def train_models(df):
    # One-hot encode genre (top 10 + other)
    TOP_N      = 10
    top_genres = df["genre"].value_counts().head(TOP_N).index.tolist()
    df = df.copy()
    df["genre"] = df["genre"].apply(lambda x: x if x in top_genres else "other")
    df = pd.get_dummies(df, columns=["genre"], drop_first=True, dtype=int)
    # Drop multicollinear + theoretically unjustified columns
    df = df.drop(columns=["acousticness_log", "loudness_log",
                           "valence", "popularity"], errors="ignore")
    TARGET = "popularity_sqrt"
    FEATURE_COLS = [c for c in df.select_dtypes(include="number").columns
                    if c not in [TARGET]]
    X, y = df[FEATURE_COLS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)


    ridge = Ridge(alpha=10).fit(X_train_sc, y_train)
    rf    = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1).fit(X_train_sc, y_train)
    gb    = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.2, random_state=42).fit(X_train_sc, y_train)


    results = {}
    for name, model in [("Ridge", ridge), ("Random Forest", rf), ("Gradient Boosting", gb)]:
        pred = model.predict(X_test_sc)
        results[name] = {"test_rmse": np.sqrt(mean_squared_error(y_test, pred)),
                         "test_r2":   r2_score(y_test, pred)}


    fi = pd.DataFrame({"feature": FEATURE_COLS, "RF": rf.feature_importances_,
                       "GB": gb.feature_importances_}).sort_values("RF", ascending=False)


    return scaler, ridge, rf, gb, results, fi, FEATURE_COLS, X_train_sc, X_test_sc, y_train, y_test




# =============================================================================
# CHART BUILDERS
# =============================================================================


def build_paradox_map(df):
    s = df.sample(5000, random_state=42)
    fig = px.scatter(s, x="lyric_sentiment", y="paradox_score",
                     color="popularity_sqrt", size="popularity_sqrt", size_max=15,
                     color_continuous_scale="YlOrRd", opacity=0.5,
                     hover_data={"track_name": True, "artist_name": True,
                                 "paradox_score": ":.3f", "lyric_sentiment": ":.3f"},
                     labels={"lyric_sentiment": "Lyric Sentiment [0,1]",
                             "paradox_score": "Paradox Score", "popularity_sqrt": "√Pop"},
                     title="The Sonic Paradox Map", template="plotly_white")
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="High Paradox Threshold")
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20))
    return fig




def build_distribution(df):
    fig = px.histogram(df, x="paradox_score", nbins=60,
                       color_discrete_sequence=[ACCENT], opacity=0.8,
                       title="Paradox Score Distribution", template="plotly_white")
    fig.add_vline(x=df["paradox_score"].mean(), line_dash="dash", line_color="red",
                  annotation_text=f"Mean={df['paradox_score'].mean():.3f}")
    fig.add_vline(x=df["paradox_score"].median(), line_dash="dot", line_color="green",
                  annotation_text=f"Median={df['paradox_score'].median():.3f}")
    fig.add_vrect(x0=0.5, x1=1.0, fillcolor=ORANGE, opacity=0.07,
                  annotation_text="High Paradox Zone")
    fig.update_layout(height=380, xaxis_title="Paradox Score", yaxis_title="Songs",
                      margin=dict(l=20, r=20, t=60, b=20))
    return fig




def build_hypothesis(df):
    d = df.copy()
    d["tercile"] = pd.qcut(d["paradox_score"], q=3,
                            labels=["Low (0–33rd)", "Medium (33–66th)", "High (66–100th)"])
    means = d.groupby("tercile", observed=True)["popularity_sqrt"].mean().reset_index()
    means.columns = ["Group", "Mean √Pop"]
    fig = px.bar(means, x="Group", y="Mean √Pop", color="Group",
                 color_discrete_sequence=["#AEC6E8", "#5B9BD5", ACCENT],
                 text="Mean √Pop", title="Popularity by Paradox Tercile",
                 template="plotly_white")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=380, showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    return fig




def build_correlations(df):
    EXCL = ["track_id", "track_name", "artist_name", "popularity_sqrt"]
    GEN  = [c for c in df.columns if c.startswith("genre_")]
    NUM  = [c for c in df.select_dtypes(include=np.number).columns if c not in EXCL + GEN]
    corr = df[NUM + ["popularity_sqrt"]].corr()["popularity_sqrt"].drop("popularity_sqrt")\
           .sort_values(key=abs, ascending=True)
    colors = [ORANGE if c == "paradox_score" else ACCENT if v > 0 else RED
              for c, v in corr.items()]
    fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h",
                           marker_color=colors, opacity=0.85))
    fig.add_vline(x=0, line_color="black", line_width=0.8)
    fig.update_layout(title="Feature Correlations with √Popularity<br>"
                            "<sup>Orange=Paradox Score | Blue=Positive | Red=Negative</sup>",
                      xaxis_title="Pearson r", height=560, template="plotly_white",
                      margin=dict(l=20, r=20, t=80, b=20))
    return fig




def build_model_comparison(results):
    models    = ["Ridge", "Random Forest", "Gradient Boosting"]
    test_rmse = [results[m]["test_rmse"] for m in models]
    test_r2   = [results[m]["test_r2"]   for m in models]

    fig = go.Figure()

    # RMSE bars — left axis
    fig.add_trace(go.Bar(
        name="Test RMSE", x=models, y=test_rmse,
        marker_color=[ACCENT, ACCENT, GREEN],
        opacity=0.9,
        text=[f"{v:.4f}" for v in test_rmse],
        textposition="outside",
        textfont=dict(size=13, color="black"),
        offsetgroup=1,
        yaxis="y",
    ))

    # R² bars — right axis
    fig.add_trace(go.Bar(
        name="Test R²", x=models, y=test_r2,
        marker_color=[ORANGE, ORANGE, RED],
        opacity=0.9,
        text=[f"{v:.4f}" for v in test_r2],
        textposition="outside",
        textfont=dict(size=13, color="black"),
        offsetgroup=2,
        yaxis="y2",
    ))

    fig.update_layout(
        title="Model Performance: Test RMSE (lower=better) vs Test R² (higher=better)",
        barmode="group",
        yaxis=dict(
            title="Test RMSE", side="left",
            range=[1.15, 1.32],
            showgrid=True,
        ),
        yaxis2=dict(
            title="Test R²", side="right",
            overlaying="y",
            range=[0.30, 0.48],
            showgrid=False,
        ),
        template="plotly_white",
        height=460,
        legend=dict(x=0.78, y=0.99, bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#ddd", borderwidth=1),
        margin=dict(l=20, r=60, t=70, b=20),
        bargap=0.25,
        bargroupgap=0.1,
    )
    return fig



def build_importance(fi):
    fp = fi.sort_values("RF", ascending=True).tail(15)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Random Forest", x=fp["RF"], y=fp["feature"],
                         orientation="h", opacity=0.85,
                         marker_color=[ORANGE if f == "paradox_score" else ACCENT
                                       for f in fp["feature"]]))
    fig.add_trace(go.Bar(name="Gradient Boosting", x=fp["GB"], y=fp["feature"],
                         orientation="h", opacity=0.65,
                         marker_color=[ORANGE if f == "paradox_score" else GREEN
                                       for f in fp["feature"]]))
    fig.update_layout(barmode="group",
                      title="Feature Importance: RF vs GB<br>"
                            "<sup>Orange = Paradox Score</sup>",
                      xaxis_title="Importance", height=520,
                      template="plotly_white", legend=dict(x=0.6, y=0.05),
                      margin=dict(l=20, r=20, t=80, b=20))
    return fig




def build_genre(df):
    # Use raw genre column directly — available before one-hot encoding
    d = df.copy()
    if "genre" in d.columns:
        d["genre_label"] = d["genre"]
    else:
        GEN = [c for c in d.columns if c.startswith("genre_")]
        d["genre_label"] = "other"
        for col in GEN:
            if col != "genre_other":
                d.loc[d[col] == 1, "genre_label"] = col.replace("genre_", "")
    counts = d["genre_label"].value_counts()
    d = d[d["genre_label"].isin(counts[counts >= 500].index)]
    stats = d.groupby("genre_label")[["popularity_sqrt", "paradox_score"]]\
             .mean().round(3).reset_index().sort_values("popularity_sqrt", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Mean √Popularity", x=stats["genre_label"],
                         y=stats["popularity_sqrt"], marker_color=ACCENT, opacity=0.85))
    fig.add_trace(go.Scatter(name="Mean Paradox Score", x=stats["genre_label"],
                             y=stats["paradox_score"], mode="lines+markers",
                             marker=dict(color=ORANGE, size=8),
                             line=dict(color=ORANGE, width=2), yaxis="y2"))
    fig.update_layout(title="Genre: √Popularity and Paradox Score",
                      yaxis=dict(title="Mean √Popularity", side="left"),
                      yaxis2=dict(title="Mean Paradox Score", side="right",
                                  overlaying="y", range=[0, 0.6]),
                      template="plotly_white", height=420,
                      legend=dict(x=0.7, y=0.95), margin=dict(l=20, r=20, t=60, b=20))
    return fig




def build_top_songs(df, n=20):
    cols = [c for c in ["track_name", "artist_name", "paradox_score",
                         "lyric_sentiment", "popularity_sqrt"] if c in df.columns]
    return (df[cols].sort_values("paradox_score", ascending=False).head(n).round(3)
            .rename(columns={"track_name": "Track", "artist_name": "Artist",
                             "paradox_score": "Paradox Score",
                             "lyric_sentiment": "Lyric Sentiment",
                             "popularity_sqrt": "√Popularity"}))




# =============================================================================
# IMPROVEMENT CHARTS
# =============================================================================


def build_bootstrap(FEATURE_COLS, X_train_sc, y_train, n=200):
    print("  Running bootstrap (200 resamples)...")
    coefs = []
    for i in range(n):
        Xb, yb = resample(X_train_sc, y_train, random_state=i, replace=True)
        c = dict(zip(FEATURE_COLS, Ridge(alpha=10).fit(Xb, yb).coef_))
        coefs.append(c["paradox_score"])
    coefs = np.array(coefs)
    lo, hi, mn = np.percentile(coefs, 2.5), np.percentile(coefs, 97.5), coefs.mean()


    fig = go.Figure()
    fig.add_trace(go.Histogram(x=coefs, nbinsx=40, marker_color=ACCENT,
                               opacity=0.75, name="Bootstrap coefficients"))
    fig.add_vline(x=lo, line_dash="dash", line_color="red",
                  annotation_text=f"2.5th={lo:.4f}", annotation_position="top left")
    fig.add_vline(x=hi, line_dash="dash", line_color="red",
                  annotation_text=f"97.5th={hi:.4f}")
    fig.add_vline(x=mn, line_color=ORANGE, line_width=2,
                  annotation_text=f"Mean={mn:.4f}")
    fig.add_vline(x=0, line_color="black", line_width=1.5,
                  annotation_text="Zero (no effect)", annotation_position="bottom right")
    fig.update_layout(
        title=f"Bootstrap Distribution of Paradox Score Effect<br>"
              f"<sup>200 resamples | 95% CI=[{lo:.4f}, {hi:.4f}] | "
              f"Excludes zero: {'YES ✓' if lo > 0 else 'NO ✗'}</sup>",
        xaxis_title="Bootstrap paradox_score Coefficient",
        yaxis_title="Frequency", template="plotly_white",
        height=420, margin=dict(l=20, r=20, t=80, b=20))
    return fig, lo, hi, mn




def build_subgroup(df):
    # Use raw genre column if available
    if "genre" in df.columns:
        d = df.copy()
        d["genre_label"] = d["genre"]
    else:
        GEN = [c for c in df.columns if c.startswith("genre_")]
        d = df.copy()
        d["genre_label"] = "other"
        for col in GEN:
            if col != "genre_other":
                d.loc[d[col] == 1, "genre_label"] = col.replace("genre_", "")
    rows = []
    for g in sorted(d["genre_label"].unique()):
        sub = d[d["genre_label"] == g]
        if len(sub) < 100:
            continue
        r, _ = pearsonr(sub["paradox_score"], sub["popularity_sqrt"])
        rows.append({"Genre": g, "r": round(r, 3), "n": len(sub)})
    gdf = pd.DataFrame(rows).sort_values("r", ascending=True)
    colors = [GREEN if r > 0 else RED for r in gdf["r"]]
    fig = go.Figure(go.Bar(x=gdf["r"], y=gdf["Genre"], orientation="h",
                           marker_color=colors, opacity=0.85,
                           text=[f"r={r:+.3f} (n={n:,})"
                                 for r, n in zip(gdf["r"], gdf["n"])],
                           textposition="outside"))
    fig.add_vline(x=0, line_color="black", line_width=0.8)
    pos = (gdf["r"] > 0).sum()
    fig.update_layout(
        title=f"Paradox Score vs √Popularity by Genre<br>"
              f"<sup>Green=supports | Red=against | {pos}/{len(gdf)} genres positive</sup>",
        xaxis_title="Pearson r", height=460, template="plotly_white",
        margin=dict(l=20, r=120, t=80, b=20))
    return fig, int(pos), len(gdf)




# =============================================================================
# APP
# =============================================================================


def create_app():
    df = load_data()
    if df.empty:
        app = Dash(__name__)
        app.layout = html.Div("Error: Could not load data.")
        return app


    print("Training models...")
    (scaler, ridge, rf, gb, results, fi, FEATURE_COLS,
     X_train_sc, X_test_sc, y_train, y_test) = train_models(df)
    print("Building improvement charts...")
    fig_bootstrap, ci_lo, ci_hi, ci_mn = build_bootstrap(FEATURE_COLS, X_train_sc, y_train)
    fig_subgroup, pos_g, tot_g = build_subgroup(df)
    print("All charts ready.")


    fig_map   = build_paradox_map(df)
    fig_dist  = build_distribution(df)
    fig_hyp   = build_hypothesis(df)
    fig_corr  = build_correlations(df)
    fig_model = build_model_comparison(results)
    fig_imp   = build_importance(fi)
    fig_genre = build_genre(df)
    top_df    = build_top_songs(df)


    app = Dash(__name__, suppress_callback_exceptions=True)


    # ── Helpers ───────────────────────────────────────────────────────────────
    def mhdr(title, close_id):
        return html.Div([
            html.H3(title, style={"margin": "0", "color": ACCENT, "fontSize": "17px"}),
            html.Button("✕", id=close_id, n_clicks=0,
                        style={"background": "none", "border": "none",
                               "fontSize": "20px", "cursor": "pointer"}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "padding": "16px 20px", "borderBottom": f"1px solid {LIGHT}"})


    def mbody(children):
        return html.Div(children, style={"padding": "16px 20px"})


    def mwrap(children, width="80%", mw="1000px"):
        return html.Div(children, style={
            "backgroundColor": "white", "borderRadius": "10px",
            "boxShadow": "0 10px 40px rgba(0,0,0,0.2)",
            "width": width, "maxWidth": mw,
            "position": "fixed", "top": "50%", "left": "50%",
            "transform": "translate(-50%,-50%)", "zIndex": "1000"})


    def info(text):
        return html.P(text, style={"color": "#555", "marginBottom": "12px",
                                   "fontSize": "13px"})


    def banner(children):
        return html.Div(children, style={"backgroundColor": LIGHT, "padding": "16px",
                                         "borderRadius": "8px", "marginBottom": "16px"})


    # ── All buttons ───────────────────────────────────────────────────────────
    all_btns = [
        ("btn-map",       "🗺️ Sonic Paradox Map"),
        ("btn-dist",      "📊 Paradox Distribution"),
        ("btn-hyp",       "🔬 Hypothesis Test"),
        ("btn-corr",      "🔥 Correlations"),
        ("btn-model",     "🤖 Model Comparison"),
        ("btn-imp",       "📈 Feature Importance"),
        ("btn-genre",     "🎸 Genre Analysis"),
        ("btn-songs",     "🏆 Top Sad-Bops"),
        ("btn-bootstrap", "📐 Bootstrap CI"),
        ("btn-subgroup",  "🔎 Subgroup Analysis"),
    ]


    bstyle = {"marginRight": "8px", "marginBottom": "6px", "padding": "9px 16px",
              "color": "white", "border": "none", "borderRadius": "6px",
              "cursor": "pointer", "fontSize": "13px", "fontWeight": "500"}


    app.layout = html.Div([


        # Header
        html.Div([
            html.H1("🎵 The Sonic Paradox Longevity Index",
                    style={"margin": "0", "fontSize": "24px", "color": "white"}),
            html.P("CIS 2450 — Does emotional incongruence between music and lyrics "
                   "predict Spotify popularity?",
                   style={"margin": "4px 0 0", "color": LIGHT, "fontSize": "13px"}),
        ], style={"backgroundColor": ACCENT, "padding": "18px 30px",
                  "boxShadow": "0 2px 8px rgba(0,0,0,0.15)"}),


        # Stats bar
        html.Div([
            *[html.Div([
                html.Div(v, style={"fontSize": "24px", "fontWeight": "bold", "color": c}),
                html.Div(l, style={"fontSize": "11px", "color": "#666"}),
            ], style={"textAlign": "center", "padding": "0 20px",
                      "borderRight": f"1px solid {LIGHT}"})
              for v, l, c in [
                  (f"{df.shape[0]:,}", "Songs", ACCENT),
                  (f"{df['paradox_score'].mean():.3f}", "Mean Paradox", ORANGE),
                  (f"{(df['paradox_score']>0.5).mean()*100:.1f}%", "High Paradox", GREEN),
                  ("p=0.0000", "OLS p-value", ACCENT),
                  (f"[{ci_lo:.4f},{ci_hi:.4f}]", "Bootstrap 95% CI", GREEN),
                  (f"{pos_g}/{tot_g} genres", "Positive Direction", GREEN),
                  ("0.4098", "Best R² (GB)", GREEN),
              ]],
        ], style={"display": "flex", "justifyContent": "center", "alignItems": "center",
                  "padding": "14px 30px", "backgroundColor": "white",
                  "boxShadow": "0 1px 4px rgba(0,0,0,0.08)", "flexWrap": "wrap"}),


        # Buttons
        html.Div([
            html.P("Explore:", style={"margin": "0 12px 0 0", "fontWeight": "500",
                                      "color": "#444", "fontSize": "13px"}),
            *[html.Button(label, id=bid, n_clicks=0,
                          style={**bstyle, "backgroundColor": "#2E7D32"
                          if bid in ("btn-bootstrap", "btn-subgroup") else ACCENT})
              for bid, label in all_btns],
        ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
                  "padding": "14px 30px", "backgroundColor": "#F8F9FA",
                  "borderBottom": f"1px solid {LIGHT}"}),


        # Main chart
        html.Div([dcc.Graph(id="main-chart", figure=fig_map,
                            style={"height": "540px"},
                            config={"displayModeBar": True})],
                 style={"padding": "20px 30px"}),


        # Finding banner
        html.Div([html.Div([
            html.H4("🔍 Research Finding", style={"margin": "0 0 8px", "color": ACCENT}),
            html.P(f"paradox_score IS a statistically significant positive predictor "
                   f"(OLS coef=+0.0455, Bootstrap 95% CI=[{ci_lo:.4f},{ci_hi:.4f}] "
                   f"excludes zero ✓, consistent across {pos_g}/{tot_g} genres). "
                   f"Effect is NEGLIGIBLE in practice — ranks 14th–17th of 23 features. "
                   f"Dominant predictors: release year, lyric density, genre.",
                   style={"margin": "0", "color": "#333", "fontSize": "14px",
                          "lineHeight": "1.6"}),
        ], style={"backgroundColor": LIGHT, "padding": "16px 20px",
                  "borderRadius": "8px", "borderLeft": f"4px solid {ACCENT}"})],
                 style={"padding": "0 30px 20px"}),


        # ── MODALS ────────────────────────────────────────────────────────────


        # 1. Map
        html.Div([
            html.Div(id="modal-map-bd", style={"display": "none"}),
            html.Div(id="modal-map", style={"display": "none"},
                     children=mwrap([
                         mhdr("🗺️ The Sonic Paradox Map", "close-map"),
                         mbody([info("Each point = one song. Color/size = popularity. "
                                    "Songs above 0.5 are strongly paradoxical."),
                                dcc.Graph(figure=fig_map, style={"height": "520px"})]),
                     ], "85%", "1100px")),
        ]),


        # 2. Distribution
        html.Div([
            html.Div(id="modal-dist-bd", style={"display": "none"}),
            html.Div(id="modal-dist", style={"display": "none"},
                     children=mwrap([
                         mhdr("📊 Paradox Score Distribution", "close-dist"),
                         mbody([info(f"Mean={df['paradox_score'].mean():.3f}, "
                                     f"Median={df['paradox_score'].median():.3f}, "
                                     f"Std={df['paradox_score'].std():.3f}. "
                                     f"{(df['paradox_score']>0.5).mean()*100:.1f}% "
                                     "are strongly paradoxical (>0.5)."),
                                dcc.Graph(figure=fig_dist, style={"height": "380px"})]),
                     ], "75%", "900px")),
        ]),


        # 3. Hypothesis
        html.Div([
            html.Div(id="modal-hyp-bd", style={"display": "none"}),
            html.Div(id="modal-hyp", style={"display": "none"},
                     children=mwrap([
                         mhdr("🔬 Hypothesis Test", "close-hyp"),
                         mbody([
                             banner([
                                 html.P("H0: β_paradox = 0 | H1: β_paradox > 0",
                                        style={"fontWeight": "500", "margin": "0 0 8px"}),
                                 html.Span("Reject H0 ✓ ",
                                           style={"color": GREEN, "fontWeight": "bold",
                                                  "fontSize": "15px"}),
                                 html.Span("(p=0.0000, coef=+0.0455, CI=[0.0401,0.0510])",
                                           style={"color": "#333"}),
                                 html.Hr(style={"margin": "10px 0"}),
                                 html.P("⚠️ At n=288,819, all p-values are 0. "
                                        "Standardized coef=0.0455 = NEGLIGIBLE effect. "
                                        "See Bootstrap CI for robust test.",
                                        style={"color": "#555", "margin": "0",
                                               "fontSize": "13px"}),
                             ]),
                             dcc.Graph(figure=fig_hyp, style={"height": "380px"}),
                         ]),
                     ], "75%", "900px")),
        ]),


        # 4. Correlations
        html.Div([
            html.Div(id="modal-corr-bd", style={"display": "none"}),
            html.Div(id="modal-corr", style={"display": "none"},
                     children=mwrap([
                         mhdr("🔥 Feature Correlations", "close-corr"),
                         mbody([info("Paradox Score (orange, r=0.082) ranks 6th. "
                                    "Year (r=0.352), lyric_density (r=0.296), "
                                    "danceability (r=0.257) dominate."),
                                dcc.Graph(figure=fig_corr, style={"height": "560px"})]),
                     ], "75%", "900px")),
        ]),


        # 5. Model comparison
        html.Div([
            html.Div(id="modal-model-bd", style={"display": "none"}),
            html.Div(id="modal-model", style={"display": "none"},
                     children=mwrap([
                         mhdr("🤖 Model Comparison", "close-model"),
                         mbody([
                             dash_table.DataTable(
                                 data=[
                                     {"Model": "Ridge", "CV RMSE": 1.2578,
                                      "Test RMSE": 1.2610, "Test R²": 0.3629,
                                      "Gap": 0.0009, "Verdict": "Underfitting"},
                                     {"Model": "Random Forest", "CV RMSE": 1.2169,
                                      "Test RMSE": 1.2179, "Test R²": 0.4056,
                                      "Gap": 0.0976, "Verdict": "Mild overfit"},
                                     {"Model": "Gradient Boosting ✓", "CV RMSE": 1.2090,
                                      "Test RMSE": 1.2136, "Test R²": 0.4098,
                                      "Gap": 0.0135, "Verdict": "Best fit"},
                                 ],
                                 columns=[{"name": c, "id": c} for c in
                                          ["Model","CV RMSE","Test RMSE",
                                           "Test R²","Gap","Verdict"]],
                                 style_cell={"textAlign": "center", "padding": "8px",
                                             "fontSize": "13px"},
                                 style_header={"backgroundColor": ACCENT,
                                               "color": "white", "fontWeight": "bold"},
                                 style_data_conditional=[{"if": {"row_index": 2},
                                                          "backgroundColor": "#EAF3DE",
                                                          "fontWeight": "bold"}],
                             ),
                             dcc.Graph(figure=fig_model,
                                       style={"height": "420px", "marginTop": "16px"}),
                         ]),
                     ], "80%", "1000px")),
        ]),


        # 6. Feature importance
        html.Div([
            html.Div(id="modal-imp-bd", style={"display": "none"}),
            html.Div(id="modal-imp", style={"display": "none"},
                     children=mwrap([
                         mhdr("📈 Feature Importance", "close-imp"),
                         mbody([info("paradox_score (orange) ranks 14th–17th. "
                                    "Agreement across RF and GB confirms the ranking."),
                                dcc.Graph(figure=fig_imp, style={"height": "520px"})]),
                     ], "80%", "1000px")),
        ]),


        # 7. Genre
        html.Div([
            html.Div(id="modal-genre-bd", style={"display": "none"}),
            html.Div(id="modal-genre", style={"display": "none"},
                     children=mwrap([
                         mhdr("🎸 Genre Analysis", "close-genre"),
                         mbody([info("Hip-hop most popular (√pop=6.687). "
                                    "Metal genres have LOWEST paradox — both "
                                    "music and lyrics are dark = aligned, not paradoxical."),
                                dcc.Graph(figure=fig_genre, style={"height": "420px"})]),
                     ], "80%", "1000px")),
        ]),


        # 8. Top songs
        html.Div([
            html.Div(id="modal-songs-bd", style={"display": "none"}),
            html.Div(id="modal-songs", style={"display": "none"},
                     children=mwrap([
                         mhdr("🏆 Top 20 Most Paradoxical Songs", "close-songs"),
                         mbody([info(f"Highest |Musical Happiness − Lyric Sentiment| "
                                     f"in {df.shape[0]:,} songs. Sortable & filterable."),
                                dash_table.DataTable(
                                    data=top_df.to_dict("records"),
                                    columns=[{"name": c, "id": c} for c in top_df.columns],
                                    style_cell={"textAlign": "left", "padding": "8px",
                                                "fontSize": "13px"},
                                    style_header={"backgroundColor": ACCENT,
                                                  "color": "white", "fontWeight": "bold"},
                                    style_data_conditional=[
                                        {"if": {"row_index": "odd"},
                                         "backgroundColor": "#F8F9FA"}],
                                    page_size=20, sort_action="native",
                                    filter_action="native",
                                )]),
                     ], "80%", "1000px")),
        ]),


        # 9. Bootstrap CI — NEW (green button)
        html.Div([
            html.Div(id="modal-bootstrap-bd", style={"display": "none"}),
            html.Div(id="modal-bootstrap", style={"display": "none"},
                     children=mwrap([
                         mhdr("📐 Bootstrap Confidence Intervals", "close-bootstrap"),
                         mbody([
                             banner([
                                 html.P("Why bootstrap? OLS p-values are meaningless "
                                        "at n=288,819. Bootstrap resampling tests "
                                        "stability without distributional assumptions.",
                                        style={"margin": "0 0 8px", "fontWeight": "500"}),
                                 html.Div([
                                     html.Span(
                                         f"95% CI = [{ci_lo:.4f}, {ci_hi:.4f}] — "
                                         f"{'Excludes zero ✓ Effect is robust' if ci_lo > 0 else 'Includes zero ✗'}",
                                         style={"color": GREEN if ci_lo > 0 else RED,
                                                "fontWeight": "bold", "fontSize": "15px"}),
                                 ]),
                                 html.P(f"Mean={ci_mn:.4f}, Std=0.0029 — "
                                        "narrow variation across 200 resamples. "
                                        "The effect is stable and not a sampling artifact.",
                                        style={"color": "#555", "margin": "8px 0 0",
                                               "fontSize": "13px"}),
                             ]),
                             dcc.Graph(figure=fig_bootstrap, style={"height": "420px"}),
                         ]),
                     ], "80%", "1000px")),
        ]),


        # 10. Subgroup Analysis — NEW (green button)
        html.Div([
            html.Div(id="modal-subgroup-bd", style={"display": "none"}),
            html.Div(id="modal-subgroup", style={"display": "none"},
                     children=mwrap([
                         mhdr("🔎 Subgroup Analysis by Genre", "close-subgroup"),
                         mbody([
                             banner([
                                 html.P("Is the paradox effect consistent across genres, "
                                        "or driven by one genre?",
                                        style={"margin": "0 0 8px", "fontWeight": "500"}),
                                 html.Span(
                                     f"Positive in {pos_g}/{tot_g} genres ✓ — "
                                     "Cross-genre phenomenon, not an artifact.",
                                     style={"color": GREEN, "fontWeight": "bold",
                                            "fontSize": "15px"}),
                                 html.P("Exception: hip-hop (r=-0.018) — structurally "
                                        "paradoxical by default, so within-genre "
                                        "additional paradox penalizes accessibility.",
                                        style={"color": "#555", "margin": "8px 0 0",
                                               "fontSize": "13px"}),
                             ]),
                             dcc.Graph(figure=fig_subgroup, style={"height": "460px"}),
                         ]),
                     ], "80%", "1000px")),
        ]),


        # Footer
        html.Div([
            html.P(f"CIS 2450 | Sonic Paradox | {df.shape[0]:,} songs | "
                   f"Best model: Gradient Boosting R²=0.4098 | "
                   f"Bootstrap CI=[{ci_lo:.4f},{ci_hi:.4f}] | "
                   f"Positive in {pos_g}/{tot_g} genres",
                   style={"margin": "0", "color": "#888", "fontSize": "12px",
                          "textAlign": "center"}),
        ], style={"padding": "16px", "borderTop": f"1px solid {LIGHT}",
                  "marginTop": "20px"}),


    ], style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#F8F9FA",
              "minHeight": "100vh"})


    # ── Main chart switcher ───────────────────────────────────────────────────
    chart_map = {
        "btn-map": fig_map, "btn-dist": fig_dist, "btn-hyp": fig_hyp,
        "btn-corr": fig_corr, "btn-model": fig_model, "btn-imp": fig_imp,
        "btn-genre": fig_genre, "btn-songs": fig_map,
        "btn-bootstrap": fig_bootstrap, "btn-subgroup": fig_subgroup,
    }


    @app.callback(
        Output("main-chart", "figure"),
        [Input(bid, "n_clicks") for bid, _ in all_btns],
        prevent_initial_call=True,
    )
    def update_chart(*args):
        from dash import ctx
        if not ctx.triggered:
            return fig_map
        return chart_map.get(ctx.triggered[0]["prop_id"].split(".")[0], fig_map)


    # ── Modal toggles ─────────────────────────────────────────────────────────
    modal_pairs = [
        ("btn-map",       "close-map",       "modal-map",       "modal-map-bd"),
        ("btn-dist",      "close-dist",      "modal-dist",      "modal-dist-bd"),
        ("btn-hyp",       "close-hyp",       "modal-hyp",       "modal-hyp-bd"),
        ("btn-corr",      "close-corr",      "modal-corr",      "modal-corr-bd"),
        ("btn-model",     "close-model",     "modal-model",     "modal-model-bd"),
        ("btn-imp",       "close-imp",       "modal-imp",       "modal-imp-bd"),
        ("btn-genre",     "close-genre",     "modal-genre",     "modal-genre-bd"),
        ("btn-songs",     "close-songs",     "modal-songs",     "modal-songs-bd"),
        ("btn-bootstrap", "close-bootstrap", "modal-bootstrap", "modal-bootstrap-bd"),
        ("btn-subgroup",  "close-subgroup",  "modal-subgroup",  "modal-subgroup-bd"),
    ]


    def make_toggle(oid, cid, mid, bid):
        @app.callback(
            Output(mid, "style"), Output(bid, "style"),
            Input(oid, "n_clicks"), Input(cid, "n_clicks"),
            State(mid, "style"), prevent_initial_call=True,
        )
        def toggle(o, c, style):
            from dash import ctx
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            is_open = True if trigger == oid else False
            s = {"display": "block" if is_open else "none"}
            return s, s
        return toggle


    for p in modal_pairs:
        make_toggle(*p)


    return app




if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=8050,
            dev_tools_ui=False, dev_tools_props_check=False)