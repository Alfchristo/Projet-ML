import streamlit as st
from evaluation.evaluator import evaluate_model

# -------------------------------------------------------------------------------------------------------
# Étape 5: Evaluation Modele
# Sidebar Section: Model Evaluation
st.subheader("Bloc d'Évaluation")
st.write(st.session_state.state['model'])
# Model Evaluation
if st.button("Évaluer le modèle"):
    if st.session_state.state['model'] is not None:
        evaluate_model(st.session_state.state['model'], selected_model, X_test, y_test)