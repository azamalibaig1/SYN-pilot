import streamlit as st
import numpy as np, joblib

st.set_page_config(page_title="Syrinx Outcomes Predictor", page_icon="🧠", layout="centered")
b=joblib.load("syrinx_outcomes_models_v01.joblib")

st.title("Syrinx Outcomes Predictor")
st.caption("Exploratory machine-learning research prototype")
st.error(
    "Not for clinical decision-making. The current pilot dataset is small and the models did not "
    "show useful cross-validated discrimination. The percentages below are experimental model outputs only."
)

with st.form("patient"):
    age=st.number_input("Age at first review",min_value=0.0,max_value=100.0,value=45.0,step=1.0)
    aet=st.selectbox("Aetiology",b["aetiology_options"])
    surg=st.selectbox("Surgical treatment",b["surgery_options"])
    go=st.form_submit_button("Estimate outcomes",type="primary")

if go:
    X=np.array([[float(age),aet,surg]],dtype=object)
    rp=float(b["radiological_model"].predict_proba(X)[0,1])
    cp=float(b["clinical_model"].predict_proba(X)[0,1])
    col1,col2=st.columns(2)
    col1.metric("Radiological improvement",f"{rp*100:.1f}%")
    col2.metric("Clinical improvement",f"{cp*100:.1f}%")
    st.warning("These are exploratory outputs and should not be interpreted as validated individual patient probabilities.")

with st.expander("Current model performance"):
    r=b["radiological_metrics"]; c=b["clinical_metrics"]
    st.write(f"Radiological model: N={r['n']}, improvement events={r['events']}, cross-validated ROC AUC={r['auc']:.3f}.")
    st.write(f"Clinical model: N={c['n']}, improvement events={c['events']}, cross-validated ROC AUC={c['auc']:.3f}.")
    st.write("The next model should add objective baseline neurological and MRI variables before clinical use.")
