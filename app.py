import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "churn_model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"

INTERNET_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]
SERVICE_COLS = ["PhoneService", "MultipleLines"] + INTERNET_COLS
RAW_COLUMNS = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
               "PhoneService", "MultipleLines", "InternetService"] + INTERNET_COLS + \
              ["Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.markdown("""
<style>
.stApp {
    background: #000000;
}
[data-testid="stSidebar"] {
    background: #121212;
}
[data-testid="stHeader"] {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)

TRANSPARENT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                   plot_bgcolor="rgba(255,255,255,0.03)", font=dict(color="#FFFFFF"))


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)


def prepare_input(data):
    """Same feature engineering steps used in training (see notebook)."""
    data = data.copy()
    for col in INTERNET_COLS:
        data[col] = data[col].replace("No internet service", "No")
    data["MultipleLines"] = data["MultipleLines"].replace("No phone service", "No")
    data["NumServices"] = (data[SERVICE_COLS] == "Yes").sum(axis=1)
    return data


def clean_uploaded(data):
    """Apply the data cleaning steps to a raw uploaded CSV."""
    data = data.copy()
    data.columns = data.columns.str.strip()
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].str.strip()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)
    if pd.api.types.is_numeric_dtype(data["SeniorCitizen"]):
        data["SeniorCitizen"] = data["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return data


def risk_level(prob):
    if prob < 0.3:
        return "Low", "#2E9E5B"
    if prob < 0.6:
        return "Medium", "#E8A33C"
    return "High", "#E8604C"


def gauge(prob, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Churn Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "rgba(46,158,91,0.15)"},
                {"range": [30, 60], "color": "rgba(232,163,60,0.15)"},
                {"range": [60, 100], "color": "rgba(232,96,76,0.15)"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10), **TRANSPARENT)
    return fig


def recommendations(c):
    tips = []
    if c["Contract"] == "Month-to-month":
        tips.append("Offer a discount for switching to a **one-year or two-year contract**.")
    if c["tenure"] <= 12:
        tips.append("New customer: enroll them in an **onboarding / welcome program**.")
    if c["InternetService"] == "Fiber optic":
        tips.append("Fiber optic users churn more: check **service quality and pricing**.")
    if c["PaymentMethod"] == "Electronic check":
        tips.append("Encourage an **automatic payment method** (bank transfer / credit card).")
    if c["InternetService"] != "No":
        if c["OnlineSecurity"] == "No":
            tips.append("Offer a free trial of **Online Security**.")
        if c["TechSupport"] == "No":
            tips.append("Offer **Tech Support** to improve the customer experience.")
    return tips


model = load_model()
meta = load_metadata()
options = meta["categorical_options"]
ranges = meta["numeric_ranges"]
threshold = meta["threshold"]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("Churn Predictor")
    st.write("Predicts whether a telecom customer is likely to **leave the company**.")
    st.divider()
    st.subheader("Model")
    st.write("**Gradient Boosting** (tuned) + SMOTE")
    m = meta["metrics"]
    c1, c2 = st.columns(2)
    c1.metric("Recall", f"{m['Recall']:.0%}")
    c2.metric("ROC-AUC", f"{m['ROC-AUC']:.2f}")
    c1.metric("F1", f"{m['F1']:.2f}")
    c2.metric("Accuracy", f"{m['Accuracy']:.0%}")
    st.caption("Trained on the Kaggle Telco Customer Churn dataset (7,021 customers).")

st.title("Telecom Customer Churn Prediction")

tab_single, tab_batch, tab_insights = st.tabs(["Single Customer", "Batch Prediction", "Model Insights"])

# ---------------- Single prediction ----------------
with tab_single:
    st.subheader("Enter customer details")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### Demographics")
        gender = st.selectbox("Gender", options["gender"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", options["Partner"])
        dependents = st.selectbox("Dependents", options["Dependents"])

        st.markdown("##### Account")
        contract = st.selectbox("Contract", options["Contract"])
        paperless = st.selectbox("Paperless Billing", options["PaperlessBilling"], index=1)
        payment = st.selectbox("Payment Method", options["PaymentMethod"])

    with col2:
        st.markdown("##### Phone & Internet")
        phone = st.selectbox("Phone Service", options["PhoneService"], index=1)
        if phone == "No":
            multiple = "No phone service"
            st.selectbox("Multiple Lines", ["No phone service"], disabled=True)
        else:
            multiple = st.selectbox("Multiple Lines", ["No", "Yes"])

        internet = st.selectbox("Internet Service", options["InternetService"])
        addons = {}
        for col, label in zip(INTERNET_COLS, ["Online Security", "Online Backup", "Device Protection",
                                              "Tech Support", "Streaming TV", "Streaming Movies"]):
            if internet == "No":
                addons[col] = "No internet service"
                st.selectbox(label, ["No internet service"], disabled=True, key=f"dis_{col}")
            else:
                addons[col] = st.selectbox(label, ["No", "Yes"], key=col)

    with col3:
        st.markdown("##### Charges")
        tenure = st.slider("Tenure (months)", 0, int(ranges["tenure"]["max"]), 12)
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0,
                                  value=70.0, step=0.5)
        total = st.number_input("Total Charges ($)", min_value=0.0, max_value=20000.0,
                                value=float(round(tenure * monthly, 2)), step=10.0,
                                help="Defaults to tenure × monthly charges.")

    customer = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
        **addons,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }

    st.divider()
    if st.button("Predict Churn", type="primary", width="stretch"):
        X = prepare_input(pd.DataFrame([customer]))
        prob = float(model.predict_proba(X)[0, 1])
        level, color = risk_level(prob)

        r1, r2 = st.columns([1, 1])
        with r1:
            st.plotly_chart(gauge(prob, color), width="stretch")
        with r2:
            if prob >= threshold:
                st.error(f"### This customer is likely to CHURN\nRisk level: **{level}**")
            else:
                st.success(f"### This customer is likely to STAY\nRisk level: **{level}**")

            tips = recommendations(customer)
            if tips:
                st.markdown("**Retention recommendations**")
                for t in tips:
                    st.markdown(f"- {t}")
            else:
                st.info("No specific risk factors detected for this customer.")

# ---------------- Batch prediction ----------------
with tab_batch:
    st.subheader("Predict churn for many customers at once")
    st.write("Upload a CSV file with the same columns as the original dataset "
             "(`customerID` and `Churn` are optional).")

    template = pd.DataFrame([{
        "customerID": "0001-DEMO", "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5, "TotalCharges": 191.0,
    }])
    st.download_button("Download CSV template", template.to_csv(index=False),
                       "churn_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        missing = [c for c in RAW_COLUMNS if c not in raw.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            data = clean_uploaded(raw)
            probs = model.predict_proba(prepare_input(data[RAW_COLUMNS]))[:, 1]

            result = raw.copy()
            result["Churn_Probability"] = probs.round(4)
            result["Prediction"] = ["Churn" if p >= threshold else "Stay" for p in probs]
            result["Risk_Level"] = [risk_level(p)[0] for p in probs]

            k1, k2, k3 = st.columns(3)
            k1.metric("Customers", len(result))
            k2.metric("Predicted to churn", int((result["Prediction"] == "Churn").sum()))
            k3.metric("Churn rate", f"{(result['Prediction'] == 'Churn').mean():.1%}")

            fig = px.histogram(result, x="Churn_Probability", nbins=20, color="Prediction",
                               color_discrete_map={"Stay": "#4C9BE8", "Churn": "#E8604C"},
                               title="Distribution of churn probabilities")
            fig.update_layout(**TRANSPARENT)
            st.plotly_chart(fig, width="stretch")

            st.dataframe(result.sort_values("Churn_Probability", ascending=False),
                         width="stretch")
            st.download_button("Download predictions", result.to_csv(index=False),
                               "churn_predictions.csv", "text/csv")

# ---------------- Insights ----------------
with tab_insights:
    st.subheader("Model performance on the test set")
    m = meta["metrics"]
    cols = st.columns(5)
    for col, (name, val) in zip(cols, m.items()):
        col.metric(name, f"{val:.3f}")

    st.subheader("Top 10 most important features")
    fi = pd.DataFrame(meta["top_features"]).sort_values("Importance")
    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                 color_discrete_sequence=["#4DA3E0"])
    fig.update_layout(height=450, **TRANSPARENT)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Key insights")
    st.markdown("""
- **Contract type** is the strongest churn driver: month-to-month customers churn the most.
- **New customers** (low tenure) are much more likely to leave.
- **Fiber optic** internet users and customers paying by **electronic check** have higher churn.
- Customers **without Online Security / Tech Support** churn more.
""")
