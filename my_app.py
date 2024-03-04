import streamlit as st
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from evaluation.evaluator import evaluate_model
from modelisation.model import get_user_input, train_machine_learning_model
from traitement.description import load_data, count_unique_values, detect_na, show_data_processing_options, \
    descriptive_analysis
from traitement.distributions import distribution_pairplot, visualize_normal_distribution, \
    visualize_exponential_distribution
from traitement.nettoyage import drop_empty_columns, detect_and_drop_similar_to_index_columns, handle_missing_values, \
    detect_outliers, treat_outliers, correlation_with_target, standardization, column_frequencies, apply_normalization, \
    apply_encoding_methods

st.title("Mon Application Streamlit")
st.write("Bonjour depuis Streamlit!")

# ----------------------------------------------------------------------------------------------------------------------
# Étape 1: Construction de Streamlit
st.sidebar.title("Paramètres")

st.title("Application Machine Learning")

# ----------------------------------------------------------------------------------------------------------------------
# Étape 2: Chargement du jeu de données
# df = load_data()
st.session_state.df = load_data()
df = st.session_state.df
if df is not None:
    if st.checkbox('Détecter les valeurs manquantes'):
        detect_na(df)

    if st.checkbox('Détecter les outliers'):
        outliers = detect_outliers(df)
        if outliers.empty:
            st.write("Aucun outlier détecté.")
        else:
            st.write("Outliers détectés :")
            st.write(outliers)

    if st.checkbox('Traiter les outliers'):
        threshold = st.slider("Choisir le seuil de Z-score :", min_value=1, max_value=10, value=3, step=1)
        df = treat_outliers(df, threshold)
        st.write("Après le traitement des outliers :")
        st.write(df.head())

    if st.checkbox('Supprimer les colonnes vides'):
        df = drop_empty_columns(df)

    if st.checkbox('Detecter et supprimer les colonnes similaires à l\'index'):
        df = detect_and_drop_similar_to_index_columns(df)

    if st.checkbox('Afficher les valeurs uniques dans chaque colonne'):
        count_unique_values(df)

    df = handle_missing_values(df)

    st.write("Après suppression des colonnes vides et similaires à l'index :")
    st.write(df.head())

    selected_columns, target_column = show_data_processing_options(df)

    st.session_state.df = df

# ----------------------------------------------------------------------------------------------------------------------
# Étape 3: Traitement des données

st.sidebar.write("Traitements variables:")

# Analyse descriptive
descriptive_analysis(df[selected_columns])

# Graphique de distribution et pairplot
distribution_pairplot(df, selected_columns, target_column)

# Corrélation avec la cible
correlation_with_target(selected_columns, target_column)
"""
# Standardisation
# User selects normalization technique
if df is not None:

    # User selects normalization technique
    normalization_technique = st.selectbox("Choose a normalization technique:",
                                           ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer',
                                            'Log Transformation', 'Normalization by Division'])

    try:
        # Apply normalization
        df = apply_normalization(df, normalization_technique)

        # Display normalized data
        st.write("Normalized Data:")
        st.write(df.head())

    except ValueError as e:
        st.error(f"Error: {str(e)}")

# Multiselect box for encoding methods
selected_encoding_methods = st.multiselect('Select Encoding Methods', ['One-Hot Encoding', 'Label Encoding'])
if df is not None:
    encoded_df = apply_encoding_methods(df, selected_encoding_methods)
    st.write('Encoded DataFrame:')
    st.write(encoded_df)
"""
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
#model = None
# Initialize session state
if 'state' not in st.session_state:
    st.session_state.state = {
        'model': None
    }
train_model = st.sidebar.toggle("Entraîner le modèle")

if train_model:
    selected_model = st.sidebar.selectbox("Sélectionnez le modèle", [" ", "Linear Regression", "Logistic Regression",
                                                                     "Decision Tree", "SVM", "Naive Bayes",
                                                                     "Random Forest",
                                                                     "Dimensionality Reduction Algorithms"])

    num_epochs = st.sidebar.number_input("Nombre d'époques", min_value=1, value=10)

    if selected_model and selected_columns and target_column:
        X_train, X_test, y_train, y_test = train_test_split(df[selected_columns], df[target_column], test_size=0.2,
                                                            random_state=42)
        # Button to trigger model training
        if st.sidebar.button("Lancer l'entraînement"):
            # Model Training
            st.session_state.state['model'] = train_machine_learning_model(selected_model, X_train, y_train, num_epochs)

        # from sklearn.preprocessing import LabelEncoder
        # label_encoder = LabelEncoder()
        # y_train = label_encoder.fit_transform(y_train)
        # y_test = label_encoder.transform(y_test)

    # Model Training
    #model = train_machine_learning_model(selected_model, X_train, y_train, num_epochs)

# -------------------------------------------------------------------------------------------------------
# Étape 5: Evaluation Modele
# Sidebar Section: Model Evaluation
st.sidebar.subheader("Bloc d'Évaluation")

# Model Evaluation
if st.sidebar.button("Évaluer le modèle"):
    if st.session_state.state['model'] is not None:
        evaluate_model(st.session_state.state['model'], selected_model, X_test, y_test)




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
