import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="wide")


@st.cache_resource
def load_artifacts():
    bundle = joblib.load("model.joblib")
    with open("metrics.json") as f:
        metrics = json.load(f)
    return bundle, metrics


@st.cache_data
def load_data():
    d = load_breast_cancer(as_frame=True)
    df = d.data[[c for c in d.data.columns if c.startswith("mean ")]].copy()
    df["diagnosis"] = d.target.map({0: "Malignant", 1: "Benign"})
    return df


bundle, metrics = load_artifacts()
model, features, ranges = bundle["model"], bundle["features"], bundle["ranges"]
df = load_data()

st.title("🩺 Breast Cancer Diagnosis Predictor")
st.caption("A machine learning app built with scikit-learn and Streamlit · "
           "Dataset: Breast Cancer Wisconsin (Diagnostic)")
st.warning("Educational project only. This is NOT a medical tool and must not be used for real diagnosis.")

tab_pred, tab_perf, tab_data = st.tabs(["🔮 Predict", "📊 Model performance", "🔎 Data explorer"])

# ---------------------------------------------------------------- Predict
with tab_pred:
    st.subheader("Enter the tumor measurements")
    st.write("Defaults are the dataset medians. Move the sliders and press **Predict**.")

    cols = st.columns(2)
    values = {}
    for i, feat in enumerate(features):
        lo, hi, med = ranges[feat]
        with cols[i % 2]:
            values[feat] = st.slider(feat.title(), float(lo), float(hi), float(med),
                                     step=(hi - lo) / 200)

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([values])[features]
        proba_benign = float(model.predict_proba(row)[0, 1])
        label = "Benign" if proba_benign >= 0.5 else "Malignant"
        conf = proba_benign if label == "Benign" else 1 - proba_benign

        c1, c2 = st.columns(2)
        c1.metric("Prediction", label)
        c2.metric("Model confidence", f"{conf:.1%}")
        st.progress(conf)
        if label == "Malignant":
            st.error("The model predicts MALIGNANT for these measurements.")
        else:
            st.success("The model predicts BENIGN for these measurements.")

# ------------------------------------------------------------ Performance
with tab_perf:
    st.subheader(f"Best model: {metrics['best_model']}")
    st.write(f"Trained on {metrics['n_samples']} samples with {metrics['n_features']} features. "
             "The best model was chosen by 5-fold cross-validation on the training set; "
             "the test set (20%) was used only for the final scores.")

    res = pd.DataFrame(metrics["results"]).T.drop(columns="confusion_matrix")
    st.dataframe(res.style.format("{:.3f}").highlight_max(axis=0, color="#d4edda"),
                 width="stretch")

    left, right = st.columns(2)
    with left:
        cm = metrics["results"][metrics["best_model"]]["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Pred. Malignant", "Pred. Benign"],
                        y=["Actual Malignant", "Actual Benign"],
                        title="Confusion matrix (test set)")
        st.plotly_chart(fig, width="stretch")
    with right:
        imp = (pd.Series(metrics["feature_importance"]).sort_values()
               .rename(index=str.title).reset_index())
        imp.columns = ["Feature", "Importance"]
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                     title="Feature importance")
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------- Data explorer
with tab_data:
    st.subheader("Explore the dataset")
    a, b = st.columns(2)
    x_feat = a.selectbox("X axis", features, index=0)
    y_feat = b.selectbox("Y axis", features, index=1)
    fig = px.scatter(df, x=x_feat, y=y_feat, color="diagnosis",
                     color_discrete_map={"Malignant": "#d62728", "Benign": "#2ca02c"},
                     opacity=0.7, title=f"{y_feat.title()} vs {x_feat.title()}")
    st.plotly_chart(fig, width="stretch")

    st.write("Class balance")
    st.bar_chart(df["diagnosis"].value_counts())
    with st.expander("Show raw data"):
        st.dataframe(df, width="stretch")

st.divider()
st.caption("Built by Mahmoud Emad · AI & Data Science, Al-Zagazig National University")
