import streamlit as st

# ----------------------------------------------------------------------------------------------------------------------
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