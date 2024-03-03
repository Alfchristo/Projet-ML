#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
# Importez les bibliothèques nécessaires au début de votre script
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import expon
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import streamlit as st

from evaluation.evaluator import evaluate_model
from modelisation.model import train_machine_learning_model, get_user_input
from traitement.distributions import visualize_normal_distribution, visualize_exponential_distribution
from traitement.nettoyage import *

st.title("Mon Application Streamlit")
st.write("Bonjour depuis Streamlit!")

# Étape 1: Construction de Streamlit
st.sidebar.title("Paramètres")

st.title("Application Machine Learning")
# Étape 2: Chargement du jeu de données
df = load_data()

if st.checkbox('Afficher les valeurs uniques dans chaque colonne'):
    count_unique_values(df)

# ------------------------------------------------------------------------------
# Étape 3: Traitement des données
selected_columns, target_column = show_data_processing_options(df)

st.sidebar.write("Traitements variables:")

# Analyse descriptive
descriptive_analysis(df[selected_columns])

# Graphique de distribution et pairplot
distribution_pairplot(df, selected_columns, target_column)

# Corrélation avec la cible
correlation_with_target(df[selected_columns], df[target_column])

# Standardisation
standardization(df[selected_columns])

st.sidebar.write("Traitements variable cible:")

# Fréquences
column_frequencies(df, target_column)

# Visualisation de la distribution normale
visualize_normal_distribution(df, target_column)

# Visualisation de la distribution exponentielle
visualize_exponential_distribution(df, target_column)

# --------------------------------------------------------------------------
# Étape 4: Machine Learning
# Sidebar Section: Model Training
st.sidebar.subheader("Bloc de Machine Learning")

# Initialization of the model to None
model = None

# Training the model
train_model = st.sidebar.checkbox("Entraîner le modèle")

if train_model:
    selected_model = st.sidebar.selectbox("Sélectionnez le modèle", [" ", "Linear Regression", "Logistic Regression",
                                                                     "Decision Tree", "SVM", "Naive Bayes",
                                                                     "Random Forest",
                                                                     "Dimensionality Reduction Algorithms"])

    if selected_columns and target_column:
        X_train, X_test, y_train, y_test = train_test_split(df[selected_columns], df[target_column], test_size=0.2,
                                                            random_state=42)

        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

    # Model Training
    model = train_machine_learning_model(selected_model, X_train, y_train)

# Sidebar Section: Predictions on New Data
if st.sidebar.checkbox("Prédictions sur de nouvelles données"):
    st.write("Entrez les nouvelles données à prédire :")
    new_data = get_user_input(selected_columns)

    if model is not None and not isinstance(model, PCA):
        prediction = model.predict(new_data)
        st.write("Prédiction :", prediction)
    elif isinstance(model, PCA):
        st.warning("Impossible de faire des prédictions avec un modèle de réduction de dimension (PCA).")
    else:
        st.warning("Aucun modèle n'est sélectionné.")
# -------------------------------------------------------------------------------------------------------
# Étape 5: Evaluation Modele
# Sidebar Section: Model Evaluation
st.sidebar.subheader("Bloc d'Évaluation")

# Model Evaluation
if st.sidebar.button("Évaluer le modèle"):
    if model is not None:
        evaluate_model(model, selected_model, X_test, y_test)

# ------------------------------------------------------------------------------------------------------------------------------------
# Étape 6: Fonctionnalités supplémentaires
st.sidebar.subheader("Fonctionnalités Supplémentaires")

# Lazy Predict
if st.sidebar.checkbox("Lazy Predict"):
    st.write("Lazy Predict :")
    lazy_predict(X_train, X_test, y_train, y_test)

# GridSearchCV
if st.sidebar.checkbox("GridSearchCV"):
    st.write("GridSearchCV :")
    grid_search_cv(X_train, y_train)

# Modèle de Deep Learning (Exemple avec Keras)
if st.sidebar.checkbox("Modèle de Deep Learning (Keras)"):
    st.write("Modèle de Deep Learning avec Keras :")
    keras_model(X_train, X_test, y_train, y_test)

# In[ ]:
