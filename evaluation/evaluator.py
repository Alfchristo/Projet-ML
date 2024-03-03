from turtle import st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.decomposition import PCA
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Function to evaluate the trained model
def evaluate_model(model, selected_model, X_test, y_test):
    if selected_model == "Dimensionality Reduction Algorithms":
        st.warning("Impossible d'évaluer un modèle de réduction de dimension (PCA) de cette manière.")
    else:
        if not isinstance(model, PCA):
            if selected_model == "Linear Regression":
                st.warning("La régression linéaire n'est pas adaptée à la classification. Choisissez un modèle de classification approprié.")
            else:
                y_pred = model.predict(X_test)

                if selected_model == "Logistic Regression":
                    # Convert predictions into classes (0 or 1) for binary classification
                    y_pred = np.round(y_pred)

                accuracy = accuracy_score(y_test, y_pred)
                st.write("Précision du modèle :", accuracy)

                # Confusion matrix
                st.write("Matrice de confusion :")
                confusion_matrix_display = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
                st.pyplot(confusion_matrix_display.figure)
        else:
            st.warning("Impossible d'évaluer un modèle de réduction de dimension (PCA) de cette manière.")

