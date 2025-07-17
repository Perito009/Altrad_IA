import streamlit as st
import pandas as pd
import pickle
import os

# Define the file paths for the saved model and label encoders
model_path = 'Data/multioutput_rf_model.pkl'
label_encoders_path = 'Data/label_encoders.pkl'

# Check if files exist (important for Streamlit deployment)
if not os.path.exists(model_path) or not os.path.exists(label_encoders_path):
    st.error("Model or label encoder files not found. Please ensure they are in the specified Google Drive paths.")
else:
    # Load the trained model and label encoders
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(label_encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        st.success("Model and label encoders loaded successfully.")

        # Get the expected feature columns from the loaded label encoders (excluding target columns)
        # Assuming all columns in label_encoders except 'genre' and 'Clavier' are features
        feature_columns = [col for col in label_encoders.keys() if col not in ['genre', 'Clavier']]

        def predict_user_data(input_df):
            """
            Preprocesses input data, makes predictions, and inverse transforms them.
            """
            processed_df = input_df.copy()

            # Preprocess categorical columns using loaded encoders
            for col in feature_columns:
                if col in processed_df.columns:
                    le = label_encoders[col]
                    # Convert to string to handle potential mixed types
                    processed_df[col] = processed_df[col].astype(str)
                    # Handle unseen categories by mapping to a placeholder if necessary
                    # For simplicity here, we'll map unseen to the encoding of 'Aucun' if it exists, or -1
                    try:
                        processed_df[col] = le.transform(processed_df[col])
                    except ValueError:
                         # If 'Aucun' was in the training data, map unseen to its encoding
                        if 'Aucun' in le.classes_:
                            unseen_encoding = le.transform(['Aucun'])[0]
                            processed_df[col] = processed_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else unseen_encoding)
                        else:
                             # Otherwise, map to -1 (handle this in model if necessary)
                             processed_df[col] = processed_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

                else:
                    # Handle missing feature columns in input data (e.g., add with a default value)
                    processed_df[col] = -1 # Or some other appropriate default
                    st.warning(f"Missing feature column in input data: {col}. Added with default value.")


            # Ensure columns match the training data features and are in the correct order
            # Drop any columns in processed_df that are not in feature_columns
            extra_cols = [col for col in processed_df.columns if col not in feature_columns]
            processed_df = processed_df.drop(columns=extra_cols)

            # Reindex to ensure the order of columns matches the training data
            processed_df = processed_df.reindex(columns=feature_columns, fill_value=-1) # Use -1 for any truly missing columns after reindexing


            # Make predictions
            predictions_encoded = model.predict(processed_df)

            # Inverse transform predictions
            predicted_genre_encoded = predictions_encoded[:, 0]
            predicted_clavier_encoded = predictions_encoded[:, 1]

            genre_le = label_encoders['genre']
            clavier_le = label_encoders['Clavier']

            # Handle potential unseen encoded values in predictions during inverse transform
            # This can happen if the model predicts an encoding that wasn't in the original classes
            predicted_genre = [genre_le.inverse_transform([val])[0] if val in genre_le.transform(genre_le.classes_) else 'Unknown' for val in predicted_genre_encoded]
            predicted_Clavier = [clavier_le.inverse_transform([val])[0] if val in clavier_le.transform(clavier_le.classes_) else 'Unknown' for val in predicted_clavier_encoded]


            # Add predictions to the original input DataFrame
            input_df['predicted_genre'] = predicted_genre
            input_df['predicted_Clavier'] = predicted_Clavier

            return input_df

        # Streamlit App
        st.title("User Material Prediction App")

        st.write("Upload a CSV file containing user material data to predict 'genre' and 'Clavier'.")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.write("Original Data:")
                st.dataframe(input_df)

                # Make predictions and get the DataFrame with predictions
                output_df = predict_user_data(input_df.copy()) # Use a copy to avoid modifying the original uploaded_file data

                st.write("Data with Predictions:")
                st.dataframe(output_df)

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

    except Exception as e:
        st.error(f"An error occurred during model or label encoder loading: {e}")
