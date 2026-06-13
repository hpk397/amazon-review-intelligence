"""
Amazon Review Intelligence System
Author: Harmanpreet Kaur
Stack : Python · Pandas · Scikit-learn · NLTK · Streamlit · Plotly

Run:  streamlit run app.py
"""

import re
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

from model.train import clean_text, fake_review_score, train, DATA_PATH
from utils.insights import get_sentiment_keywords

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
MODEL_PATH = BASE / "model" / "sentiment_model.pkl"
VEC_PATH   = BASE / "model" / "tfidf_vectorizer.pkl"
METRICS_PATH = BASE / "model" / "metrics.pkl"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Review Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  body, .stApp { background: #0f1117; color: #e8eaf0; }
  .block-container { padding: 2rem 2.5rem 3rem; max-width: 1200px; }

  /* Metric cards */
  .metric-card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #f0b429; }
  .metric-label { font-size: 0.78rem; color: #8b8fa8; text-transform: uppercase;
                  letter-spacing: 0.08em; margin-top: 0.25rem; }

  /* Result badge */
  .badge-pos {
    display: inline-block;
    background: #0d3d2e; border: 1px solid #10b981;
    color: #10b981; border-radius: 8px;
    padding: 0.6rem 1.4rem; font-size: 1.1rem; font-weight: 600;
  }
  .badge-neg {
    display: inline-block;
    background: #3d0d0d; border: 1px solid #ef4444;
    color: #ef4444; border-radius: 8px;
    padding: 0.6rem 1.4rem; font-size: 1.1rem; font-weight: 600;
  }
  .badge-fake {
    display: inline-block;
    background: #3d2d0d; border: 1px solid #f59e0b;
    color: #f59e0b; border-radius: 8px;
    padding: 0.4rem 1rem; font-size: 0.9rem; font-weight: 600; margin-top: 0.5rem;
  }

  /* Section headers */
  .section-title {
    font-size: 1.15rem; font-weight: 700; color: #c9cde0;
    border-left: 3px solid #f0b429; padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
  }

  /* History table */
  .hist-row {
    display: flex; gap: 1rem; align-items: center;
    background: #1a1d27; border: 1px solid #2a2d3a;
    border-radius: 8px; padding: 0.6rem 1rem; margin-bottom: 0.5rem;
    font-size: 0.88rem;
  }
  .hist-text { flex: 1; color: #c9cde0; }
  .hist-pos  { color: #10b981; font-weight: 600; min-width: 80px; }
  .hist-neg  { color: #ef4444; font-weight: 600; min-width: 80px; }
  .hist-conf { color: #8b8fa8; min-width: 60px; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #13151f !important; }
</style>
""", unsafe_allow_html=True)


# ── Load / train model ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on Amazon reviews…")
def load_model():
    if not MODEL_PATH.exists():
        train()
    with open(MODEL_PATH,   "rb") as f: model = pickle.load(f)
    with open(VEC_PATH,     "rb") as f: vec   = pickle.load(f)
    with open(METRICS_PATH, "rb") as f: mets  = pickle.load(f)
    return model, vec, mets


@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_json(DATA_PATH, lines=True)
    df = df[df["overall"] != 3].copy()
    df["label"] = (df["overall"] >= 4).astype(int)
    df["sentiment"] = df["label"].map({1: "Positive", 0: "Negative"})
    return df


model, vectorizer, metrics = load_model()
df_data = load_dataset()

pos_reviews = df_data[df_data["label"] == 1]["reviewText"].tolist()
neg_reviews = df_data[df_data["label"] == 0]["reviewText"].tolist()


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(text: str):
    cleaned  = clean_text(text)
    vec_text = vectorizer.transform([cleaned])
    proba    = model.predict_proba(vec_text)[0]
    label    = model.predict(vec_text)[0]
    fake     = fake_review_score(text)
    return {
        "label":      "Positive" if label == 1 else "Negative",
        "confidence": round(max(proba) * 100, 1),
        "pos_prob":   round(proba[1] * 100, 1),
        "neg_prob":   round(proba[0] * 100, 1),
        "fake_score": round(fake * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Amazon Review Intelligence")
    st.caption("NLP · Machine Learning · Streamlit")
    st.divider()

    st.markdown("**Model Performance**")
    for label, val in [("Accuracy", metrics["accuracy"]),
                       ("Precision", metrics["precision"]),
                       ("Recall",    metrics["recall"]),
                       ("F1 Score",  metrics["f1"])]:
        st.metric(label, f"{val}%")
    st.divider()

    st.markdown("**Dataset**")
    st.caption(f"Total reviews: {len(df_data):,}")
    st.caption(f"Positive: {df_data['label'].sum():,}")
    st.caption(f"Negative: {(df_data['label']==0).sum():,}")
    st.divider()

    st.markdown("**Tech Stack**")
    st.caption("Python · Pandas · NumPy")
    st.caption("Scikit-learn · NLTK")
    st.caption("TF-IDF · Logistic Regression")
    st.caption("Plotly · WordCloud · Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🛒 Amazon Review Intelligence System")
st.caption("Sentiment analysis · Keyword insights · Fake review detection")
st.markdown("---")

# ── Metric overview ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, (label, val) in zip(
    [c1, c2, c3, c4],
    [("Accuracy", f"{metrics['accuracy']}%"),
     ("F1 Score",  f"{metrics['f1']}%"),
     ("Precision", f"{metrics['precision']}%"),
     ("Recall",    f"{metrics['recall']}%")]
):
    col.markdown(
        f'<div class="metric-card"><div class="metric-value">{val}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔎 Predict Review", "📊 Dataset Insights", "☁️ Word Clouds", "📜 History"]
)


# ── TAB 1: Predict ────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Enter an Amazon Review</div>', unsafe_allow_html=True)

    example_reviews = [
        "",
        "The battery life on this product is amazing! Lasts all day.",
        "Completely broke after two days. Terrible quality.",
        "BEST PRODUCT EVER BUY NOW!!!!! MUST HAVE!!!!!",
        "Decent product for the price. A few minor issues but overall okay.",
    ]
    example = st.selectbox("Or pick an example →", example_reviews)

    review_text = st.text_area(
        "Review text",
        value=example,
        height=130,
        placeholder="Type a product review here…",
        label_visibility="collapsed",
    )

    col_btn, col_clr = st.columns([1, 6])
    predict_btn = col_btn.button("Predict", type="primary", use_container_width=True)

    if predict_btn and review_text.strip():
        result = predict(review_text)

        # Store in history
        st.session_state.history.insert(0, {
            "text":       review_text[:80] + ("…" if len(review_text) > 80 else ""),
            "sentiment":  result["label"],
            "confidence": result["confidence"],
            "fake":       result["fake_score"],
        })

        st.markdown("---")
        r1, r2 = st.columns([1, 1])

        with r1:
            badge_cls = "badge-pos" if result["label"] == "Positive" else "badge-neg"
            icon = "😊" if result["label"] == "Positive" else "😞"
            st.markdown(
                f'<div class="{badge_cls}">{icon} {result["label"]} Review</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Confidence:** {result['confidence']}%")

            if result["fake_score"] >= 50:
                st.markdown(
                    f'<div class="badge-fake">⚠️ Likely Fake Review ({result["fake_score"]}%)</div>',
                    unsafe_allow_html=True,
                )
            elif result["fake_score"] >= 25:
                st.markdown(
                    f'<div class="badge-fake">🟡 Possibly Suspicious ({result["fake_score"]}%)</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("✅ **Authenticity:** Looks genuine")

        with r2:
            fig = go.Figure(go.Bar(
                x=[result["pos_prob"], result["neg_prob"]],
                y=["Positive", "Negative"],
                orientation="h",
                marker_color=["#10b981", "#ef4444"],
                text=[f"{result['pos_prob']}%", f"{result['neg_prob']}%"],
                textposition="auto",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c9cde0",
                margin=dict(l=0, r=0, t=10, b=0),
                height=160,
                showlegend=False,
                xaxis=dict(range=[0, 100], showgrid=False),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Keywords from this review
        words = re.sub(r"[^a-z\s]", " ", review_text.lower()).split()
        stop  = set(stopwords.words("english"))
        words = [w for w in words if w not in stop and len(w) > 3]
        if words:
            from collections import Counter
            top = Counter(words).most_common(5)
            st.markdown('<div class="section-title">Key Words Detected</div>', unsafe_allow_html=True)
            kw_cols = st.columns(len(top))
            for col, (word, freq) in zip(kw_cols, top):
                col.metric(word, freq)

    elif predict_btn:
        st.warning("Please enter a review first.")


# ── TAB 2: Dataset Insights ───────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        counts = df_data["sentiment"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            marker_colors=["#10b981", "#ef4444"],
            hole=0.45,
            textfont_size=13,
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9cde0",
            margin=dict(l=0, r=0, t=30, b=0),
            height=280,
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Top Positive Keywords</div>', unsafe_allow_html=True)
        pos_kw = get_sentiment_keywords(pos_reviews[:500], 8)
        fig_pos = px.bar(
            x=[k for k, _ in pos_kw], y=[v for _, v in pos_kw],
            color_discrete_sequence=["#10b981"],
        )
        fig_pos.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9cde0", margin=dict(l=0, r=0, t=10, b=0),
            height=220, showlegend=False,
            xaxis_title="", yaxis_title="Frequency",
        )
        st.plotly_chart(fig_pos, use_container_width=True)

    st.markdown('<div class="section-title">Top Complaint Keywords (Negative Reviews)</div>',
                unsafe_allow_html=True)
    neg_kw = get_sentiment_keywords(neg_reviews[:500], 12)
    fig_neg = px.bar(
        x=[k for k, _ in neg_kw], y=[v for _, v in neg_kw],
        color_discrete_sequence=["#ef4444"],
    )
    fig_neg.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c9cde0", margin=dict(l=0, r=0, t=10, b=0),
        height=240, showlegend=False,
        xaxis_title="", yaxis_title="Frequency",
    )
    st.plotly_chart(fig_neg, use_container_width=True)

    # Confusion matrix
    st.markdown('<div class="section-title">Model Confusion Matrix</div>', unsafe_allow_html=True)
    cm = np.array(metrics["conf_matrix"])
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=["Pred Negative", "Pred Positive"],
        y=["True Negative", "True Positive"],
        colorscale=[[0, "#1a1d27"], [1, "#f0b429"]],
        text=cm, texttemplate="%{text}",
        showscale=False,
    ))
    fig_cm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#c9cde0",
        margin=dict(l=0, r=0, t=20, b=0), height=260,
    )
    st.plotly_chart(fig_cm, use_container_width=True)


# ── TAB 3: Word Clouds ────────────────────────────────────────────────────────
with tab3:
    stop = set(stopwords.words("english")) | {"product", "item", "amazon", "one", "got", "use", "used", "really", "also"}

    def make_wordcloud(reviews, colormap):
        text = " ".join(reviews[:800])
        wc = WordCloud(
            width=700, height=350,
            background_color="#1a1d27",
            colormap=colormap,
            stopwords=stop,
            max_words=80,
            prefer_horizontal=0.7,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.patch.set_facecolor("#1a1d27")
        return fig

    wc1, wc2 = st.columns(2)
    with wc1:
        st.markdown('<div class="section-title">😊 Positive Review Keywords</div>',
                    unsafe_allow_html=True)
        st.pyplot(make_wordcloud(pos_reviews, "Greens"))
    with wc2:
        st.markdown('<div class="section-title">😞 Negative Review Keywords</div>',
                    unsafe_allow_html=True)
        st.pyplot(make_wordcloud(neg_reviews, "Reds"))


# ── TAB 4: History ────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No predictions yet. Go to 'Predict Review' and analyse a review!")
    else:
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

        for item in st.session_state.history:
            cls = "hist-pos" if item["sentiment"] == "Positive" else "hist-neg"
            icon = "😊" if item["sentiment"] == "Positive" else "😞"
            fake_note = f" | ⚠️ Fake {item['fake']}%" if item["fake"] >= 50 else ""
            st.markdown(
                f'<div class="hist-row">'
                f'<span class="hist-text">{item["text"]}</span>'
                f'<span class="{cls}">{icon} {item["sentiment"]}</span>'
                f'<span class="hist-conf">{item["confidence"]}%{fake_note}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Amazon Review Intelligence System · Built with Python & Streamlit · Harmanpreet Kaur")
