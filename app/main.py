from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 1. CONFIGURATION  (auto-detect project root)
# =========================================================

_here = Path(__file__).resolve()
_candidates = [
    _here.parent,
    _here.parent.parent,
    _here.parent.parent.parent,
    Path.cwd(),
]

DATASET_NAME = "smartship_modeling_dataset.csv"

ROOT = None
for base in _candidates:
    if (base / "data" / "processed" / DATASET_NAME).exists():
        ROOT = base
        break

if ROOT is None:
    for base in _candidates:
        if (base / "data").exists() or (base / "models").exists():
            ROOT = base
            break

if ROOT is None:
    ROOT = _here.parent

MODEL_PATH      = ROOT / "models"  / "smartship_pipeline.joblib"
DATA_PATH       = ROOT / "data"    / "processed" / DATASET_NAME
EVALUATION_PATH = ROOT / "reports" / "results"   / "evaluation_summary.json"

# Debug (اختياري — شيله بعد التأكد)
print(f"[SmartShip] ROOT        = {ROOT}")
print(f"[SmartShip] DATA_PATH   = {DATA_PATH}  (exists={DATA_PATH.exists()})")
print(f"[SmartShip] MODEL_PATH  = {MODEL_PATH}  (exists={MODEL_PATH.exists()})")
print(f"[SmartShip] EVAL_PATH   = {EVALUATION_PATH}  (exists={EVALUATION_PATH.exists()})")

st.set_page_config(
    page_title="SmartShip | Smart Delivery Platform",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# 2. SHARED STYLING
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
.stApp {
    background: linear-gradient(180deg, #F6F8FB 0%, #EEF2F7 100%);
    color: #1F2A44;
}
.block-container {
    max-width: 1340px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }

/* ============ TYPOGRAPHY ============ */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #0B2A1F !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}
[data-testid="stMarkdownContainer"] p { color: #4A5568; line-height: 1.55; }
[data-testid="stCaptionContainer"] { color: #8A94A6 !important; font-size: 12.5px !important; }

/* ============ HERO HEADER ============ */
.hero-wrap {
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px;
    padding: 6px 0 20px 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 24px;
}
.hero-left  { display: flex; align-items: center; gap: 16px; }
.hero-icon {
    width: 52px; height: 52px;
    border-radius: 14px;
    background: linear-gradient(135deg, #0F7B4D 0%, #16A06B 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    box-shadow: 0 6px 18px rgba(15, 123, 77, 0.22);
    flex-shrink: 0;
}
.hero-eyebrow {
    font-size: 12.5px; font-weight: 700;
    color: #0F7B4D; letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 2px;
}
.hero-title {
    font-size: 26px; font-weight: 800;
    color: #0B2A1F; line-height: 1.2; margin-bottom: 4px;
}
.hero-sub { font-size: 13.5px; color: #6B7A90; }

/* ============ CARDS ============ */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E7ECF2 !important;
    border-radius: 14px !important;
    padding: 18px !important;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03),
                0 4px 14px rgba(16, 24, 40, 0.04);
    transition: box-shadow 0.2s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03),
                0 6px 22px rgba(16, 24, 40, 0.06);
}

.card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.card-icon {
    width: 34px; height: 34px;
    border-radius: 9px;
    background: #E8F6EF;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px; flex-shrink: 0;
}
.card-title {
    font-size: 15px; font-weight: 700;
    color: #0B2A1F; line-height: 1.2;
}
.card-sub {
    font-size: 12.5px; color: #8A94A6;
    margin: 2px 0 14px 0; padding-left: 44px;
}

/* ============ INPUTS ============ */
[data-testid="stWidgetLabel"] p {
    color: #344563 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
.stNumberInput input, .stTextInput input {
    color: #172B4D !important;
    background-color: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: #0F7B4D !important;
    background-color: #FFFFFF !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
}
.stSelectbox div[data-baseweb="select"] * {
    color: #172B4D !important;
    font-size: 14px !important;
}

/* ============ SLIDER ============ */
.stSlider [data-baseweb="slider"] > div > div > div { background: #0F7B4D !important; }
.stSlider [role="slider"] {
    background: #0F7B4D !important;
    border-color: #0F7B4D !important;
    box-shadow: 0 2px 8px rgba(15, 123, 77, 0.3) !important;
}

/* ============ BUTTONS ============ */
.stButton button, .stFormSubmitButton button {
    background: linear-gradient(135deg, #0F7B4D 0%, #12895A 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 14.5px !important;
    min-height: 48px;
    box-shadow: 0 4px 12px rgba(15, 123, 77, 0.24);
    transition: all 0.2s ease;
}
.stButton button:hover, .stFormSubmitButton button:hover {
    background: linear-gradient(135deg, #0B5C3A 0%, #0F7B4D 100%) !important;
    box-shadow: 0 6px 18px rgba(15, 123, 77, 0.34);
    transform: translateY(-1px);
    color: #FFFFFF !important;
}

/* ============ PROGRESS ============ */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #0F7B4D 0%, #16A06B 100%) !important;
}
.stProgress > div > div { background-color: #E8EFF6 !important; border-radius: 6px; }

/* ============ WARNING BOX ============ */
.warn-box {
    background: #FFF8E9;
    border: 1px solid #F5D08A;
    border-radius: 9px;
    padding: 10px 12px;
    font-size: 12px; color: #8A5A00;
    margin-top: 12px; line-height: 1.5;
}

/* ============ METRICS ============ */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E7ECF2;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
}
[data-testid="stMetricLabel"] p {
    color: #8A94A6 !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
    color: #0B2A1F !important;
    font-weight: 800;
    font-size: 26px !important;
}

/* ============ ALERTS & DIVIDERS ============ */
[data-testid="stAlert"] { border-radius: 10px !important; }
[data-testid="stAlert"] p { color: inherit !important; }
hr { border-color: #E7ECF2 !important; }

/* ============ SUMMARY CARD LAYOUT ============ */
.summary-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0;
    border-bottom: 1px dashed #E7ECF2;
    font-size: 13.5px;
}
.summary-row:last-child { border-bottom: none; }
.summary-key { color: #7B899D; }
.summary-val { color: #1F2A44; font-weight: 700; }

/* ============ STATUS PILL ============ */
.status-pill {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 30px;
    background: #E8F6EF;
    color: #0F7B4D;
    font-size: 11px; font-weight: 800;
}

/* ============ SECTION KICKER / PANEL TITLE ============ */
.section-kicker {
    color: #7D8DA4; font-size: 11px;
    font-weight: 800; letter-spacing: .13em;
    text-transform: uppercase;
}
.panel-title {
    color: #0B2A1F; font-weight: 800;
    font-size: 17px; margin: 2px 0 0;
}

/* ============ FOOTER ============ */
.footer-text {
    text-align: center; color: #8A94A6;
    font-size: 12px; padding-top: 8px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. CACHED LOADERS
# =========================================================

@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH, parse_dates=["order_purchase_timestamp"])
    df["purchase_month_label"] = df["order_purchase_timestamp"].dt.strftime("%Y-%m")
    return df


@st.cache_data(show_spinner=False)
def load_evaluation():
    if not EVALUATION_PATH.exists():
        return {}
    with open(EVALUATION_PATH, encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# 4. NAVIGATION STATE
# =========================================================

if "current_page" not in st.session_state:
    st.session_state.current_page = "predict"


def goto(page: str):
    st.session_state.current_page = page
    st.rerun()


# =========================================================
# 5. HERO HELPER
# =========================================================

def render_hero(eyebrow: str, title: str, subtitle: str,
                nav_label: str, nav_page: str, nav_arrow: str = "→"):
    left, right = st.columns([5, 1])
    with left:
        st.markdown(f"""
        <div class="hero-wrap" style="border-bottom:0;margin-bottom:0;padding-bottom:0;">
            <div class="hero-left">
                <div class="hero-icon">🚚</div>
                <div>
                    <div class="hero-eyebrow">{eyebrow}</div>
                    <div class="hero-title">{title}</div>
                    <div class="hero-sub">{subtitle}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        if st.button(f"{nav_label}  {nav_arrow}", key=f"nav_{nav_page}", width="stretch"):
            goto(nav_page)
    st.markdown(
        '<div style="border-bottom:1px solid #E2E8F0;margin:20px 0 24px 0;"></div>',
        unsafe_allow_html=True,
    )


# =========================================================
# 6. PAGE: PREDICTION
# =========================================================

def render_prediction_page():
    artifact = load_artifact()
    if artifact is None:
        st.error(
            "Trained model not found. Run:\n\n"
            "`python -m src.features.build_dataset`\n\n"
            "`python -m src.models.train`"
        )
        st.stop()

    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]
    preprocessor = pipeline.named_steps["preprocessor"]
    num_features = artifact["numeric_features"]
    cat_features = artifact["categorical_features"]

    num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]

    num_defaults = dict(zip(num_features, num_imputer.statistics_))
    cat_options = {
        name: [str(v) for v in vals if str(v) != "nan"]
        for name, vals in zip(cat_features, cat_encoder.categories_)
    }

    render_hero(
        eyebrow="Smart Delivery Platform",
        title="Order Risk Assessment",
        subtitle="Enter order details and shipping information to evaluate readiness before dispatch.",
        nav_label="View Dashboard",
        nav_page="dashboard",
        nav_arrow="→",
    )

    values = {}

    with st.form("order_form"):
        main_col, side_col = st.columns([3, 1], gap="large")

        with main_col:
            # CARD 1
            with st.container(border=True):
                st.markdown("""
                <div class="card-header">
                    <div class="card-icon">🛒</div>
                    <div class="card-title">Order &amp; Products</div>
                </div>
                <div class="card-sub">Quantities, pricing and seller details in the order.</div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    values["item_count"] = st.number_input("Item count", min_value=1, max_value=30, value=1)
                with c2:
                    values["product_count"] = st.number_input("Distinct products", min_value=1, max_value=30, value=1)

                c1, c2 = st.columns(2)
                with c1:
                    values["seller_count"] = st.number_input("Seller count", min_value=1, max_value=10, value=1)
                with c2:
                    values["total_price"] = st.number_input("Total item price", min_value=0.0, value=150.0, step=10.0)

                c1, c2 = st.columns(2)
                with c1:
                    values["avg_item_price"] = st.number_input("Average item price", min_value=0.0, value=150.0, step=10.0)
                with c2:
                    values["total_freight"] = st.number_input("Total freight cost", min_value=0.0, value=25.0, step=5.0)

            st.write("")

            # CARD 2
            with st.container(border=True):
                st.markdown("""
                <div class="card-header">
                    <div class="card-icon">📦</div>
                    <div class="card-title">Preparation &amp; Shipping</div>
                </div>
                <div class="card-sub">Weights, dimensions and order preparation deadlines.</div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    values["avg_product_weight_g"] = st.number_input("Average product weight (g)", min_value=0.0, value=1200.0, step=100.0)
                with c2:
                    values["total_product_weight_g"] = st.number_input("Total product weight (g)", min_value=0.0, value=1200.0, step=100.0)

                c1, c2 = st.columns(2)
                with c1:
                    values["avg_product_volume_cm3"] = st.number_input("Average product volume (cm³)", min_value=0.0, value=5000.0, step=500.0)
                with c2:
                    values["avg_shipping_limit_days"] = st.number_input("Average shipping deadline (days)", min_value=0.0, value=3.0, step=0.5)

                c1, c2 = st.columns(2)
                with c1:
                    values["min_shipping_limit_days"] = st.number_input("Minimum shipping deadline (days)", min_value=0.0, value=3.0, step=0.5)
                with c2:
                    values["estimated_lead_days"] = st.number_input("Estimated delivery lead (days)", min_value=1.0, value=24.0, step=1.0)

            st.write("")

            # CARD 3
            with st.container(border=True):
                st.markdown("""
                <div class="card-header">
                    <div class="card-icon">📍</div>
                    <div class="card-title">Destination &amp; Purchase Timing</div>
                </div>
                <div class="card-sub">Customer location, seller location and purchase timing.</div>
                """, unsafe_allow_html=True)

                states = ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT",
                          "PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"]

                c1, c2 = st.columns(2)
                with c1:
                    values["customer_state"] = st.selectbox("Customer state", states, index=states.index("SP"))
                with c2:
                    seller_states = cat_options.get("seller_state", states)
                    values["seller_state"] = st.selectbox(
                        "Seller state",
                        seller_states,
                        index=(seller_states.index("SP") if "SP" in seller_states else 0),
                    )

                category_options = cat_options.get("dominant_category", ["unknown"])
                values["dominant_category"] = st.selectbox("Main product category", category_options)

                c1, c2 = st.columns(2)
                weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                with c1:
                    values["purchase_weekday"] = st.selectbox("Purchase weekday", weekdays)
                with c2:
                    values["purchase_month"] = int(st.selectbox("Purchase month", [str(i) for i in range(1,13)], index=5))

                values["purchase_hour"] = st.slider("Purchase hour", 0, 23, 14, format="%d:00")
                values["is_weekend"] = int(values["purchase_weekday"] in ["Saturday", "Sunday"])

        with side_col:
            # Assessment Summary
            with st.container(border=True):
                st.markdown("""
                <div class="card-header">
                    <div class="card-icon">📋</div>
                    <div class="card-title">Assessment Summary</div>
                </div>
                <div class="card-sub" style="padding-left:0;">Order overview before shipping</div>
                """, unsafe_allow_html=True)

                prob_val = st.session_state.get("last_probability", None)

                st.markdown(
                    "<div style='font-size:12.5px;font-weight:600;color:#344563;margin-bottom:6px;'>"
                    "Late-delivery probability</div>",
                    unsafe_allow_html=True,
                )

                if prob_val is None:
                    st.progress(0.0)
                    st.markdown(
                        "<div style='font-size:22px;font-weight:800;color:#8A94A6;'>— %</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption("Assessment not run yet")
                    st.markdown(
                        "<div class='warn-box'>⚠️ Compute the probability so shipping is not blocked — currently not desired.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.progress(min(max(prob_val, 0.0), 1.0))
                    color = "#C0392B" if prob_val >= artifact.get("threshold", 0.5) else "#0F7B4D"
                    st.markdown(
                        f"<div style='font-size:22px;font-weight:800;color:{color};'>"
                        f"{prob_val:.1%}</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "🔴 High risk" if prob_val >= artifact.get("threshold", 0.5)
                        else "🟢 Lower risk"
                    )

            st.write("")

            # Shipment Details
            with st.container(border=True):
                st.markdown("""
                <div class="card-header">
                    <div class="card-icon">📦</div>
                    <div class="card-title">Shipment Details</div>
                </div>
                """, unsafe_allow_html=True)

                rows = [
                    ("Items",       str(values.get("item_count", 1))),
                    ("Destination", str(values.get("customer_state", "SP"))),
                    ("Lead time",   f"{int(values.get('estimated_lead_days', 24))} days"),
                    ("Purchase",    f"{values.get('purchase_weekday','Monday')[:3]} · {values.get('purchase_hour', 14):02d}:00"),
                ]
                html_rows = "".join(
                    f"<div class='summary-row'><span class='summary-key'>{k}</span>"
                    f"<span class='summary-val'>{v}</span></div>"
                    for k, v in rows
                )
                st.markdown(html_rows, unsafe_allow_html=True)
                st.caption("✅ Order data available for review")

        st.write("")

        submitted = st.form_submit_button(
            "➡️  Run Risk Assessment",
            type="primary",
            width="stretch",
        )

    # --- Prediction results ---
    if submitted:
        row = {col: values.get(col, num_defaults.get(col, np.nan)) for col in feature_columns}
        input_df = pd.DataFrame([row], columns=feature_columns)

        probability = float(pipeline.predict_proba(input_df)[0, 1])
        threshold = float(artifact.get("threshold", 0.5))
        prediction = int(probability >= threshold)
        status = "AT RISK" if prediction else "LOWER RISK"

        st.session_state["last_probability"] = probability

        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        st.caption("AI-powered delivery risk assessment based on the submitted order.")

        r1, r2 = st.columns(2, gap="large")
        with r1:
            with st.container(border=True):
                st.metric("Late Delivery Probability", f"{probability:.1%}")
                st.caption("Estimated probability of arriving after the promised date.")
        with r2:
            with st.container(border=True):
                st.metric("Model Decision", status)
                st.caption("Classification based on the configured decision threshold.")

        st.write("")
        st.markdown("#### Delivery Risk Probability")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"Decision threshold: {threshold:.2f}")

        if prediction:
            st.warning(
                "⚠️ Potential Late Delivery\n\n"
                "The model identifies this order as being at risk of late delivery. "
                "Consider reviewing seller readiness, shipping arrangements and logistics planning."
            )
        else:
            st.success(
                "✓ Lower Delivery Risk\n\n"
                "The model does not classify this order as being at risk of late delivery "
                "at the current decision threshold. This prediction is not a delivery guarantee."
            )


# =========================================================
# 7. PAGE: DASHBOARD
# =========================================================

def render_dashboard_page():
    df = load_data()
    evaluation = load_evaluation()

    if df is None:
        st.error(
            "The processed modeling dataset was not found.\n\n"
            f"Expected at: `{DATA_PATH}`\n\n"
            "Run from the project root:\n"
            "```\npython -m src.features.build_dataset\npython -m src.models.train\n```"
        )
        st.stop()

    total_orders = len(df)
    late_orders = int(df["is_late"].sum())
    on_time_orders = total_orders - late_orders
    late_rate = late_orders / total_orders if total_orders else 0
    on_time_rate = 1 - late_rate
    metrics = evaluation.get("test_metrics", {})

    render_hero(
        eyebrow="Smart Delivery Platform",
        title="Delivery Intelligence Overview",
        subtitle="Historical delivery performance and machine-learning signals across the Olist dataset.",
        nav_label="Run Risk Assessment",
        nav_page="predict",
        nav_arrow="→",
    )

    head_l, head_r = st.columns([4, 1])
    with head_l:
        st.markdown("### Overview")
        st.caption("Historical dataset performance · Olist Brazilian E-Commerce")
    with head_r:
        st.markdown(
            '<div style="text-align:right;padding-top:8px">'
            '<span class="status-pill">● DATA LOADED</span></div>',
            unsafe_allow_html=True,
        )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Orders analyzed", f"{total_orders:,}",
              help="Orders with complete outcomes in the processed modeling dataset.")
    k2.metric("On-time delivery", f"{on_time_rate:.1%}",
              help="Share delivered on or before the estimated delivery date.")
    k3.metric("Late deliveries", f"{late_rate:.1%}",
              delta=f"{late_orders:,} orders", delta_color="inverse",
              help="Share delivered after the estimated delivery date.")
    k4.metric("Late-order recall",
              f"{metrics.get('recall_late', 0):.1%}" if metrics else "N/A",
              help="Test-set recall at the selected decision threshold.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.55, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="section-kicker">DELIVERY PERFORMANCE</div>'
                '<div class="panel-title">Monthly late-delivery trend</div>',
                unsafe_allow_html=True,
            )
            monthly = (
                df.groupby("purchase_month_label", as_index=False)
                  .agg(orders=("is_late", "size"), late_rate=("is_late", "mean"))
            )
            monthly["late_rate_pct"] = monthly["late_rate"] * 100

            fig = px.line(monthly, x="purchase_month_label", y="late_rate_pct", markers=True)
            fig.update_traces(
                line=dict(color="#0F7B4D", width=3),
                marker=dict(size=7, color="#0F7B4D"),
                hovertemplate="%{x}<br>Late rate: %{y:.1f}%<extra></extra>",
            )
            fig.update_layout(
                height=300,
                margin=dict(l=8, r=8, t=18, b=8),
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis_title=None, yaxis_title="Late orders (%)",
                font=dict(family="Inter", color="#718096", size=11),
                yaxis=dict(gridcolor="#EEF2F7", ticksuffix="%", rangemode="tozero"),
                xaxis=dict(showgrid=False, tickangle=-35),
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption("Monthly rate is calculated from observed delivery outcomes in the processed dataset.")

    with right:
        with st.container(border=True):
            st.markdown(
                '<div class="section-kicker">ORDER OUTCOMES</div>'
                '<div class="panel-title">Delivery distribution</div>',
                unsafe_allow_html=True,
            )
            pie = pd.DataFrame({
                "Outcome": ["On time", "Late"],
                "Orders":  [on_time_orders, late_orders],
            })
            fig2 = px.pie(
                pie, names="Outcome", values="Orders", hole=.72,
                color="Outcome",
                color_discrete_map={"On time": "#0F7B4D", "Late": "#F0A64A"},
            )
            fig2.update_traces(
                textinfo="none",
                hovertemplate="%{label}<br>%{value:,} orders (%{percent})<extra></extra>",
                marker=dict(line=dict(color="white", width=4)),
            )
            fig2.update_layout(
                height=300,
                margin=dict(l=5, r=5, t=12, b=5),
                paper_bgcolor="white",
                showlegend=True,
                legend=dict(orientation="h", y=-.05, x=.18),
                annotations=[dict(
                    text=f"<b>{total_orders:,}</b><br>"
                         f"<span style='font-size:11px;color:#718096'>orders</span>",
                    x=.5, y=.5, showarrow=False,
                    font=dict(size=20, color="#0B2A1F"),
                )],
            )
            st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})
            st.caption(f"{late_orders:,} late orders and {on_time_orders:,} on-time orders.")

    lower_left, lower_right = st.columns([1.25, 1], gap="large")

    with lower_left:
        with st.container(border=True):
            st.markdown(
                '<div class="section-kicker">GEOGRAPHIC SIGNAL</div>'
                '<div class="panel-title">Late-delivery rate by customer state</div>',
                unsafe_allow_html=True,
            )
            state = (
                df.groupby("customer_state", as_index=False)
                  .agg(orders=("is_late", "size"), late_rate=("is_late", "mean"))
            )
            state = state[state["orders"] >= 250].sort_values("late_rate", ascending=True).tail(12)
            state["late_rate_pct"] = state["late_rate"] * 100

            fig3 = px.bar(
                state, x="late_rate_pct", y="customer_state",
                orientation="h",
                text=state["late_rate_pct"].map(lambda x: f"{x:.1f}%"),
            )
            fig3.update_traces(
                marker_color="#16A06B", textposition="outside",
                hovertemplate="%{y}<br>Late rate: %{x:.1f}%<extra></extra>",
            )
            fig3.update_layout(
                height=330,
                margin=dict(l=5, r=30, t=15, b=5),
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis_title="Late orders (%)", yaxis_title=None,
                font=dict(family="Inter", color="#718096", size=11),
                xaxis=dict(gridcolor="#EEF2F7", ticksuffix="%"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})
            st.caption("States with at least 250 observed orders; descriptive rates, not causal comparisons.")

    with lower_right:
        with st.container(border=True):
            st.markdown(
                '<div class="section-kicker">MODEL SNAPSHOT</div>'
                '<div class="panel-title">Test-set performance</div>',
                unsafe_allow_html=True,
            )
            if metrics:
                m1, m2 = st.columns(2)
                m1.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
                m2.metric("PR-AUC",  f"{metrics.get('pr_auc', 0):.3f}")
                m3, m4 = st.columns(2)
                m3.metric("Precision · late", f"{metrics.get('precision_late', 0):.1%}")
                m4.metric("F1 · late",        f"{metrics.get('f1_late', 0):.1%}")
                st.markdown("---")
                st.markdown("**Selected model**")
                st.markdown(
                    f"<span style='font-size:18px;font-weight:800;color:#0B2A1F'>"
                    f"{evaluation.get('selected_model_family', 'Saved model')}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Stratified holdout results measure within-dataset discrimination; "
                    "they do not establish future-period generalization."
                )
            else:
                st.info("Evaluation summary is not available.")


# =========================================================
# 8. ROUTER
# =========================================================

page = st.session_state.get("current_page", "predict")

if page == "dashboard":
    render_dashboard_page()
else:
    render_prediction_page()


# =========================================================
# 9. FOOTER
# =========================================================

st.divider()
st.markdown(
    "<div class='footer-text'>SmartShip · Smart Delivery Platform</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='footer-text'>Machine Learning · Predictive Analytics · E-commerce</div>",
    unsafe_allow_html=True,
)