import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_URL = st.secrets.get("FASTAPI_URL", "http://localhost:8000")

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0;
    }
    .sub-header {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .result-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        background-color: #ff8c00;
        border: 1px solid #d2691e;
    }
    .risk-label {
        font-size: 0.85rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
    }
    .risk-label-high { background-color: #7f1d1d; color: #ffffff; }
    .risk-label-medium { background-color: #78350f; color: #ffffff; }
    .risk-label-low { background-color: #14532d; color: #ffffff; }
    .prob-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: #000000 !important;
    }
    .explanation-box {
        background-color: #fff3e0;
        border-left: 3px solid #d2691e;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Transaction fraud checker</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time fraud detection powered by XGBoost and SHAP explainability</p>', unsafe_allow_html=True)

with st.expander("About this system", expanded=False):
    st.write(
        "This tool scores a transaction's fraud probability using a model trained on "
        "6M+ mobile money transactions. Every prediction comes with a plain-language "
        "explanation of the top contributing factors, powered by SHAP."
    )

st.divider()

tab1, tab2 = st.tabs(["Check transaction", "Analytics"])

with tab1:
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("Transaction details")

        with st.form("transaction_form"):
            transaction_type = st.selectbox(
                "Transaction type",
                ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
            )
            amount = st.number_input("Amount ($)", min_value=0.0, value=5000.0, step=100.0)

            st.markdown("**Sender account**")
            sc1, sc2 = st.columns(2)
            with sc1:
                oldbalanceOrg = st.number_input("Balance before", min_value=0.0, value=10000.0, key="ob_orig")
            with sc2:
                newbalanceOrig = st.number_input("Balance after", min_value=0.0, value=5000.0, key="nb_orig")

            st.markdown("**Receiver account**")
            rc1, rc2 = st.columns(2)
            with rc1:
                oldbalanceDest = st.number_input("Balance before", min_value=0.0, value=2000.0, key="ob_dest")
            with rc2:
                newbalanceDest = st.number_input("Balance after", min_value=0.0, value=7000.0, key="nb_dest")

            hour_of_day = st.slider("Hour of day", 0, 23, 14)

            submitted = st.form_submit_button("Check for fraud", use_container_width=True, type="primary")

    with col2:
        st.subheader("Result")

        if submitted:
            payload = {
                "transaction_type": transaction_type,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
                "hour_of_day": hour_of_day,
                "step": 1
            }

            try:
                with st.spinner("Analyzing transaction..."):
                    pred_response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
                    pred_response.raise_for_status()
                    pred_data = pred_response.json()

                    explain_response = requests.post(f"{API_URL}/explain", json=payload, timeout=15)
                    explain_response.raise_for_status()
                    explain_data = explain_response.json()

                prob = pred_data["fraud_probability"]

                if prob > 0.7:
                    label_class, risk_text = "risk-label-high", "High risk"
                elif prob > 0.3:
                    label_class, risk_text = "risk-label-medium", "Medium risk"
                else:
                    label_class, risk_text = "risk-label-low", "Low risk"

                st.markdown(f"""
                    <div class="result-card">
                        <span class="risk-label {label_class}">{risk_text}</span>
                        <div class="prob-number">{prob:.1%}</div>
                        <div style="color: #333333; font-size: 0.85rem;">fraud probability</div>
                        <div class="explanation-box">{explain_data.get("explanation", "No explanation available.")}</div>
                    </div>
                """, unsafe_allow_html=True)

                top_features = explain_data.get("top_features", [])
                if top_features:
                    feat_df = pd.DataFrame(top_features)
                    feat_df["abs_value"] = feat_df["shap_value"].abs()
                    feat_df = feat_df.sort_values("abs_value", ascending=True)
                    fig_shap = px.bar(
                        feat_df, x="shap_value", y="feature", orientation="h",
                        color="shap_value", color_continuous_scale=["#16a34a", "#dc2626"],
                        labels={"shap_value": "Impact on fraud score", "feature": ""},
                        title="What drove this prediction"
                    )
                    fig_shap.update_layout(height=250, showlegend=False, coloraxis_showscale=False,
                                            margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig_shap, use_container_width=True)

                st.session_state.history.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Type": transaction_type,
                    "Amount": amount,
                    "Fraud probability": prob,
                    "Risk": risk_text
                })

            except requests.exceptions.RequestException as e:
                st.error(f"Couldn't reach the backend: {e}")
        else:
            st.info("Fill out the transaction details and click Check for fraud to see the result.")

with tab2:
    if not st.session_state.history:
        st.info("Check a few transactions first to see analytics here.")
    else:
        df = pd.DataFrame(st.session_state.history)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total checked", len(df))
        c2.metric("Flagged high risk", (df["Risk"] == "High risk").sum())
        c3.metric("Avg fraud probability", f"{df['Fraud probability'].mean():.1%}")

        col_a, col_b = st.columns(2)

        with col_a:
            fig_trend = px.line(
                df, x="Time", y="Fraud probability", markers=True,
                title="Fraud probability over time"
            )
            fig_trend.update_layout(yaxis_tickformat=".0%", height=320)
            st.plotly_chart(fig_trend, use_container_width=True)

        with col_b:
            risk_counts = df["Risk"].value_counts().reset_index()
            risk_counts.columns = ["Risk", "Count"]
            fig_pie = px.pie(
                risk_counts, names="Risk", values="Count",
                color="Risk",
                color_discrete_map={"High risk": "#dc2626", "Medium risk": "#d97706", "Low risk": "#16a34a"},
                title="Risk level distribution", hole=0.4
            )
            fig_pie.update_layout(height=320)
            st.plotly_chart(fig_pie, use_container_width=True)

        fig_type = px.bar(
            df.groupby("Type")["Fraud probability"].mean().reset_index(),
            x="Type", y="Fraud probability",
            title="Average fraud probability by transaction type",
            color="Fraud probability", color_continuous_scale="Reds"
        )
        fig_type.update_layout(yaxis_tickformat=".0%", height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_type, use_container_width=True)

        st.subheader("Transaction log")
        display_df = df.copy()
        display_df["Amount"] = display_df["Amount"].apply(lambda x: f"${x:,.2f}")
        display_df["Fraud probability"] = display_df["Fraud probability"].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
