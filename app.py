import io
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from emmis.config import settings
from emmis.models.vision_model import ImageAnomalyDetector
from emmis.database import MyMongoDBClient
from emmis.encryption import Cipher
from emmis.models.risk_model import RiskScoringModel
from emmis.models.language_model import (
    TextProcessor,
    CallSentimentPipeline,
    format_timestamp,
    truncate,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EMMIS · Risk Intelligence",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

/* ── header banner ────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0d0d14 0%, #0f1729 50%, #0d0d14 100%);
    border: 1px solid #1e2d4a;
    border-radius: 14px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 80% at 70% 50%, #1a4a8a18 0%, transparent 70%);
    pointer-events: none;
}
.hero h1 { color: #e8eaf6; font-size: 1.85rem; font-weight: 700; margin: 0 0 .35rem; letter-spacing: -.02em; }
.hero p  { color: #7986a8; font-size: .95rem; margin: 0; }
.hero .badge {
    display: inline-block; background: #1e3a6e; color: #5c9aff;
    border: 1px solid #2d5aaa; padding: 2px 10px; border-radius: 20px;
    font-size: .78rem; font-weight: 600; margin-bottom: .6rem;
}

/* ── risk cards ───────────────────────────────────────────────────────────── */
.risk-banner {
    border-radius: 12px; padding: 1.6rem 2rem;
    text-align: center; margin: 1rem 0;
    border: 1.5px solid;
}
.risk-HIGH   { background: #1a0606; border-color: #e53935; }
.risk-MEDIUM { background: #1a1006; border-color: #fb8c00; }
.risk-LOW    { background: #051a0a; border-color: #43a047; }

/* ── cipher box ───────────────────────────────────────────────────────────── */
.cipher-box {
    background: #060d1a;
    border: 1px solid #1e3358;
    border-radius: 8px;
    padding: .9rem 1.1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: .8rem;
    color: #4fc3f7;
    word-break: break-all;
    line-height: 1.6;
}

/* ── stat card ────────────────────────────────────────────────────────────── */
.stat-card {
    background: #080f1e;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-card .val { font-size: 1.9rem; font-weight: 700; color: #e8eaf6; line-height: 1; }
.stat-card .lbl { font-size: .78rem; color: #5c7095; margin-top: .3rem; letter-spacing: .05em; text-transform: uppercase; }

/* ── keyword pill ─────────────────────────────────────────────────────────── */
.kw-pill {
    display: inline-block;
    background: #2a0c0c; color: #ef9a9a;
    border: 1px solid #5d1c1c;
    border-radius: 20px; padding: 3px 12px;
    font-size: .78rem; margin: 3px 3px 3px 0;
}

/* ── section header ───────────────────────────────────────────────────────── */
.section-hdr {
    font-size: .78rem; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; color: #4a6494;
    border-bottom: 1px solid #1a2540; padding-bottom: .4rem;
    margin: 1.4rem 0 .8rem;
}

/* ── record row ───────────────────────────────────────────────────────────── */
.rec-row {
    background: #07111f;
    border: 1px solid #162035;
    border-radius: 8px;
    padding: .75rem 1rem;
    margin-bottom: .5rem;
}

/* ── Streamlit overrides ──────────────────────────────────────────────────── */
section[data-testid="stSidebar"] { background: #060c18 !important; }
div[data-testid="stMetricValue"]  { font-family: 'Space Grotesk', sans-serif !important; }
.stTabs [data-baseweb="tab"]      { font-size: .88rem; }
div[data-testid="stExpander"]     { border: 1px solid #1a2540 !important; border-radius: 8px !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Component initialisation (cached)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading AI models …")
def _load_components():
    pipeline = CallSentimentPipeline(model_name=settings.MODEL_NAME)
    cipher = Cipher()
    text_proc = TextProcessor(pipeline)
    img_det = ImageAnomalyDetector()
    r_model = RiskScoringModel()
    db = MyMongoDBClient(
        uri=settings.MONGODB_URI,
        db_name=settings.DATABASE_NAME,
        collection_name=settings.COLLECTION_NAME,
    )
    return cipher, text_proc, img_det, r_model, db


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────


def _risk_color(level: str) -> str:
    return {"HIGH": "#e53935", "MEDIUM": "#fb8c00", "LOW": "#43a047"}.get(
        level, "#607d8b"
    )


def _risk_icon(level: str) -> str:
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(level, "⚪")


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

_PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#8a9cc5",
    margin=dict(l=12, r=12, t=36, b=12),
)


def _gauge(score: float, title: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(score * 100, 1),
            title={"text": title, "font": {"size": 13, "color": "#8a9cc5"}},
            number={"suffix": "%", "font": {"size": 22, "color": "#e8eaf6"}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"color": "#4a6494"}},
                "bar": {"color": "#3a7bd5", "thickness": 0.28},
                "bgcolor": "#060d1a",
                "bordercolor": "#1e3358",
                "steps": [
                    {"range": [0, 35], "color": "#0a2a0a"},
                    {"range": [35, 65], "color": "#1f1a06"},
                    {"range": [65, 100], "color": "#250606"},
                ],
                "threshold": {
                    "line": {"color": _risk_color("HIGH"), "width": 2},
                    "thickness": 0.75,
                    "value": score * 100,
                },
            },
        )
    )
    fig.update_layout(height=200, **_PLOT_BASE)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────


def _sidebar() -> str:
    with st.sidebar:
        st.markdown(
            "<p style='color:#3a7bd5;font-weight:700;font-size:1.1rem;"
            "letter-spacing:.05em;margin-bottom:.2rem'>⚠️ EMMIS</p>"
            "<p style='color:#4a6494;font-size:.78rem;margin-top:0'>"
            "Encrypted Multi-Modal Intelligence</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🔬  Analyze", "📊  Dashboard", "📋  Records"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("<p class='section-hdr'>System Status</p>", unsafe_allow_html=True)
        try:
            _, _, _, _, db = _load_components()
            st.success("✅ Components loaded")
            st.success("🗄️ MongoDB")
            st.caption(f"Records: {db.get_total_count()}")
        except Exception as exc:
            st.error(f"Init error: {exc}")

        st.markdown("---")
        st.markdown(
            "<p style='color:#2a3a5a;font-size:.72rem;text-align:center'>"
            "v1.0.0 · Encrypted Multi-Modal System</p>",
            unsafe_allow_html=True,
        )

    return page.split("  ")[1]  # strip emoji prefix


# ─────────────────────────────────────────────────────────────────────────────
# Page: Analyze
# ─────────────────────────────────────────────────────────────────────────────


def _page_analyze(cipher, text_proc, img_det, r_model, db):
    st.markdown(
        """
        <div class="hero">
            <span class="badge">MULTI-MODAL · ENCRYPTED · REAL-TIME</span>
            <h1>⚠️ Encrypted Multi-Modal Intelligence System</h1>
            <p>Submit encrypted text and image data — the system decrypts, analyses, and scores risk automatically.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Text input ────────────────────────────────────────────────────────────
    with col_left:
        st.markdown("<p class='section-hdr'>Text Input</p>", unsafe_allow_html=True)
        mode = st.radio(
            "mode",
            ["✏️  Enter plain text (auto-encrypt)", "🔒  Paste pre-encrypted text"],
            horizontal=True,
            label_visibility="collapsed",
        )
        plain_text = ""
        encrypted_input = ""

        if "plain" in mode:
            plain_text = st.text_area(
                "Plain text",
                placeholder="e.g.  Critical overheating in motor unit 4 — immediate shutdown required.",
                height=110,
                label_visibility="collapsed",
            )
            if plain_text:
                enc_preview = cipher.encrypt_text(plain_text)
                st.markdown("**Encrypted preview:**")
                st.markdown(
                    f'<div class="cipher-box">{enc_preview[:96]}…</div>',
                    unsafe_allow_html=True,
                )
        else:
            encrypted_input = st.text_area(
                "Encrypted text",
                placeholder="Paste encrypted cipher-text here …",
                height=110,
                label_visibility="collapsed",
            )

    # ── Image input ───────────────────────────────────────────────────────────
    with col_right:
        st.markdown(
            "<p class='section-hdr'>Image Input (optional)</p>", unsafe_allow_html=True
        )
        uploaded = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.image(Image.open(uploaded), caption="Uploaded image")

    # ── Analyse button ────────────────────────────────────────────────────────
    st.markdown(" ")
    run = st.button("🚀  Run Analysis", type="primary", use_container_width=True)

    if not run:
        return

    # ── Validation ────────────────────────────────────────────────────────────
    if not plain_text and not encrypted_input:
        st.error("Please provide text input.")
        return

    final_enc = (
        cipher.encrypt_text(plain_text)
        if "plain" in mode and plain_text
        else encrypted_input.strip()
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    with st.spinner("🔓 Decrypting …"):
        try:
            decrypted = cipher.decrypt_text(final_enc)
        except Exception as exc:
            st.error(f"Decryption failed: {exc}")
            return

    with st.spinner("🧠 NLP analysis …"):
        nlp = text_proc.process(decrypted)

    cv: dict = {
        "success": False,
        "anomaly_score": 0.0,
        "anomaly_regions": 0,
        "anomaly_detected": False,
    }
    annotated_bytes = None
    if uploaded:
        with st.spinner("👁️ Anomaly detection …"):
            uploaded.seek(0)
            raw_img = uploaded.read()
            cv = img_det.detect(raw_img)
            annotated_bytes = img_det.annotate(raw_img)

    with st.spinner("⚖️ Risk scoring …"):
        risk = r_model.predict(
            nlp_risk_score=nlp["nlp_risk_score"],
            sentiment_score=nlp["sentiment"]["risk_contribution"],
            anomaly_score=cv.get("anomaly_score", 0.0),
            keyword_count=nlp["risk_keywords"]["keyword_count"],
            anomaly_regions=cv.get("anomaly_regions", 0),
        )

    # Persist
    db.insert_analysis(
        {
            "decrypted_text": decrypted,
            "image_analyzed": uploaded is not None,
            "nlp_risk_score": nlp["nlp_risk_score"],
            "anomaly_score": cv.get("anomaly_score", 0.0),
            "unified_risk_score": risk["unified_risk_score"],
            "risk_level": risk["risk_level"],
        }
    )

    # ── Risk banner ────────────────────────────────────────────────────────────
    level = risk["risk_level"]
    score = risk["unified_risk_score"]
    color = _risk_color(level)
    icon = _risk_icon(level)

    st.success("✅ Analysis complete")
    st.markdown(
        f"""
        <div class="risk-banner risk-{level}">
            <div style="color:{color};font-size:2rem;font-weight:700;letter-spacing:-.02em">
                {icon} {level} RISK
            </div>
            <div style="color:#c0c8e0;font-size:1.05rem;margin-top:.4rem">
                Unified Score: <b style="color:{color}">{score:.1%}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3 = st.tabs(["📊 Risk Breakdown", "🧠 NLP Analysis", "👁️ CV Results"])

    # ── Tab 1: Risk breakdown ─────────────────────────────────────────────────
    with t1:
        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(
                _gauge(nlp["nlp_risk_score"], "NLP Risk"), use_container_width=True
            )
        with g2:
            st.plotly_chart(
                _gauge(cv.get("anomaly_score", 0), "Anomaly"), use_container_width=True
            )
        with g3:
            st.plotly_chart(_gauge(score, "Unified Risk"), use_container_width=True)

        # Feature contributions
        contrib = risk["feature_importance"]
        df_c = pd.DataFrame(
            {
                "Feature": list(contrib.keys()),
                "Contribution": list(contrib.values()),
            }
        ).sort_values("Contribution")

        fig_c = px.bar(
            df_c,
            x="Contribution",
            y="Feature",
            orientation="h",
            title="Feature Contributions to Unified Risk Score",
            color="Contribution",
            color_continuous_scale=[[0, "#1a4a1a"], [0.5, "#4a3a08"], [1, "#4a0808"]],
        )
        fig_c.update_layout(height=250, showlegend=False, **_PLOT_BASE)
        st.plotly_chart(fig_c, use_container_width=True)

        # Probability donut
        proba = risk["risk_probabilities"]
        fig_p = go.Figure(
            go.Pie(
                labels=list(proba.keys()),
                values=list(proba.values()),
                hole=0.55,
                marker_colors=[_risk_color(k) for k in proba],
            )
        )
        fig_p.update_layout(
            title="Class Probability Distribution",
            height=250,
            **_PLOT_BASE,
        )
        st.plotly_chart(fig_p, use_container_width=True)

    # ── Tab 2: NLP ────────────────────────────────────────────────────────────
    with t2:
        sent = nlp["sentiment"]
        s_color = _risk_color("HIGH" if sent["label"] == "NEGATIVE" else "LOW")

        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("<p class='section-hdr'>Sentiment</p>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="background:#06100e;border:1px solid {s_color};
                            border-radius:10px;padding:1rem;text-align:center">
                    <div style="color:{s_color};font-size:1.3rem;font-weight:700">
                        {sent["label"]}
                    </div>
                    <div style="color:#8a9cc5;font-size:.85rem;margin-top:.4rem">
                        Confidence: <b style="color:#e8eaf6">{sent["confidence"]:.1%}</b>
                    </div>
                    <div style="color:#8a9cc5;font-size:.85rem">
                        Risk contrib: <b style="color:{s_color}">{sent["risk_contribution"]:.1%}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c_b:
            st.markdown(
                "<p class='section-hdr'>Risk Keywords</p>", unsafe_allow_html=True
            )
            kws = nlp["risk_keywords"]["found_keywords"]
            if kws:
                pills = "".join(f'<span class="kw-pill">{kw}</span>' for kw in kws)
                st.markdown(pills, unsafe_allow_html=True)
                st.caption(f"{len(kws)} keyword(s) detected")
            else:
                st.success("No risk keywords found")

        st.markdown("<p class='section-hdr'>Decrypted Text</p>", unsafe_allow_html=True)
        st.info(decrypted)

        ents = nlp.get("entities", {})
        if ents:
            st.markdown(
                "<p class='section-hdr'>Extracted Entities</p>", unsafe_allow_html=True
            )
            for etype, vals in ents.items():
                st.markdown(f"**{etype}:** {', '.join(vals)}")

    # ── Tab 3: CV ─────────────────────────────────────────────────────────────
    with t3:
        if not cv.get("success"):
            st.info("No image was analysed.  Upload an image to see CV results.")
        else:
            c_x, c_y = st.columns(2)
            detected = cv["anomaly_detected"]
            a_color = _risk_color("HIGH") if detected else _risk_color("LOW")
            a_label = "ANOMALY DETECTED" if detected else "NO ANOMALY"

            with c_x:
                st.markdown(
                    f"""
                    <div style="background:#060c18;border:1px solid {a_color};
                                border-radius:10px;padding:1.2rem;text-align:center">
                        <div style="color:{a_color};font-size:1.1rem;font-weight:700">
                            {a_label}
                        </div>
                        <div style="color:#8a9cc5;margin-top:.5rem">
                            Score: <b style="color:#e8eaf6">{cv['anomaly_score']:.1%}</b>
                        </div>
                        <div style="color:#8a9cc5">
                            Regions: <b style="color:#e8eaf6">{cv['anomaly_regions']}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                stats = cv.get("image_stats", {})
                if stats:
                    st.markdown(
                        "<p class='section-hdr'>Image Statistics</p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"**Dimensions:** {stats.get('width')} × {stats.get('height')} px"
                    )
                    st.markdown(f"**Mean pixel:** {stats.get('mean_pixel')}")
                    st.markdown(f"**Std dev:** {stats.get('std_pixel')}")
                    st.markdown(
                        f"**Anomalous area:** {stats.get('anomaly_area_px')} px²"
                    )

            with c_y:
                if annotated_bytes:
                    st.markdown(
                        "<p class='section-hdr'>Annotated Image</p>",
                        unsafe_allow_html=True,
                    )
                    st.image(
                        Image.open(io.BytesIO(annotated_bytes)),
                        caption="Red contours = anomalous regions",
                    )


# ─────────────────────────────────────────────────────────────────────────────
# Page: Dashboard
# ─────────────────────────────────────────────────────────────────────────────


def _page_dashboard(db):
    st.markdown(
        """
        <div class="hero">
            <span class="badge">LIVE ANALYTICS</span>
            <h1>📊 Risk Intelligence Dashboard</h1>
            <p>Aggregated view of all stored analysis records.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    records = db.get_recent_records(limit=100)
    if not records:
        st.info("No records yet.  Run some analyses to populate the dashboard.")
        return

    df = pd.DataFrame(records)

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="stat-card"><div class="val">{len(df)}</div>'
            f'<div class="lbl">Total Analyses</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        avg = df["unified_risk_score"].mean() if "unified_risk_score" in df else 0
        st.markdown(
            f'<div class="stat-card"><div class="val">{avg:.0%}</div>'
            f'<div class="lbl">Avg Risk Score</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        hi = len(df[df["risk_level"] == "HIGH"]) if "risk_level" in df else 0
        st.markdown(
            f'<div class="stat-card"><div class="val" style="color:#e53935">{hi}</div>'
            f'<div class="lbl">High Risk Cases</div></div>',
            unsafe_allow_html=True,
        )
    with k4:
        rate = hi / len(df) * 100 if len(df) else 0
        st.markdown(
            f'<div class="stat-card"><div class="val">{rate:.0f}%</div>'
            f'<div class="lbl">High Risk Rate</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        if "risk_level" in df.columns:
            counts = df["risk_level"].value_counts().reset_index()
            counts.columns = ["Risk Level", "Count"]
            fig = px.pie(
                counts,
                names="Risk Level",
                values="Count",
                title="Risk Level Distribution",
                color="Risk Level",
                color_discrete_map={
                    "HIGH": "#e53935",
                    "MEDIUM": "#fb8c00",
                    "LOW": "#43a047",
                },
                hole=0.5,
            )
            fig.update_layout(height=300, **_PLOT_BASE)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "unified_risk_score" in df.columns:
            fig = px.histogram(
                df,
                x="unified_risk_score",
                nbins=20,
                title="Risk Score Distribution",
                color_discrete_sequence=["#3a7bd5"],
            )
            fig.update_layout(height=300, **_PLOT_BASE)
            st.plotly_chart(fig, use_container_width=True)

    # ── Risk over time ────────────────────────────────────────────────────────
    if "timestamp" in df.columns and "unified_risk_score" in df.columns:
        df_t = df.sort_values("timestamp")
        fig = px.line(
            df_t,
            x="timestamp",
            y="unified_risk_score",
            title="Unified Risk Score Over Time",
            color_discrete_sequence=["#3a7bd5"],
        )
        for thresh, color, label in [
            (0.65, "#e53935", "High threshold"),
            (0.35, "#fb8c00", "Medium threshold"),
        ]:
            fig.add_hline(
                y=thresh,
                line_dash="dash",
                line_color=color,
                annotation_text=label,
                annotation_font_color=color,
            )
        fig.update_layout(height=280, **_PLOT_BASE)
        st.plotly_chart(fig, use_container_width=True)

    # ── Scatter: NLP vs Anomaly ───────────────────────────────────────────────
    if "nlp_risk_score" in df.columns and "anomaly_score" in df.columns:
        fig = px.scatter(
            df,
            x="nlp_risk_score",
            y="anomaly_score",
            color="risk_level",
            title="NLP Risk Score vs Anomaly Score",
            color_discrete_map={
                "HIGH": "#e53935",
                "MEDIUM": "#fb8c00",
                "LOW": "#43a047",
            },
            size_max=10,
        )
        fig.update_layout(height=300, **_PLOT_BASE)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Records
# ─────────────────────────────────────────────────────────────────────────────


def _page_records(db):
    st.markdown(
        """
        <div class="hero">
            <span class="badge">AUDIT TRAIL</span>
            <h1>📋 Analysis Records</h1>
            <p>Browse every stored analysis result.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    limit = st.slider("Records to load", 5, 50, 15)
    records = db.get_recent_records(limit=limit)

    if not records:
        st.info("No records found.")
        return

    st.caption(f"Showing {len(records)} most recent records")

    for rec in records:
        level = rec.get("risk_level", "UNKNOWN")
        score = rec.get("unified_risk_score", 0.0)
        icon = _risk_icon(level)
        ts = format_timestamp(rec.get("timestamp"))

        with st.expander(f"{icon}  {level}  ·  {score:.1%}  ·  {ts}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Text:** {truncate(rec.get('decrypted_text','—'), 100)}")
                st.markdown(f"**NLP score:** {rec.get('nlp_risk_score',0):.1%}")
                st.markdown(f"**Anomaly score:** {rec.get('anomaly_score',0):.1%}")
            with c2:
                rid = rec.get("_id") or rec.get("record_id", "—")
                st.markdown(f"**Record ID:** `{str(rid)[:24]}…`")
                st.markdown(f"**Image analysed:** {rec.get('image_analyzed', False)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Application entry point."""
    page = _sidebar()

    try:
        cipher, text_proc, img_det, r_model, db = _load_components()
    except Exception as exc:
        st.error(f"Failed to load components: {exc}")
        return

    if page == "Analyze":
        _page_analyze(cipher, text_proc, img_det, r_model, db)
    elif page == "Dashboard":
        _page_dashboard(db)
    elif page == "Records":
        _page_records(db)


if __name__ == "__main__":
    main()
