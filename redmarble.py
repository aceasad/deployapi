import pandas as pd
import json
import numpy as np
import os
from pathlib import Path
from ast import literal_eval
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import classification_report
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Define paths for data and models
DATA_DIR = Path('./data')
MODELS_DIR = Path('./models')
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

joblib.load('models/model.pkl')

def preprocess_data(json_data):
    data.dropna(inplace=True)
    data['winner'] = data['winner'].astype(int)
    data['player_1'] = data['player_1'].str.replace('&lt;', '<').str.replace('&gt;', '>').str.replace('<sp/>', '')
    data['player_2'] = data['player_2'].str.replace('&lt;', '<').str.replace('&gt;', '>').str.replace('<sp/>', '')
    return data

def get_column_types(data):
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    object_cols = data.select_dtypes(include='object').columns.tolist()
    return numeric_cols, object_cols

def parse_and_count_units(unit_str):
    unit_counts = Counter(unit_str)
    return unit_counts

def feature_engineering(df_cleaned):

    # Apply the function to both player_1_units and player_2_units columns
    player_1_units_counts = df_cleaned['player_1_units'].apply(parse_and_count_units)
    player_2_units_counts = df_cleaned['player_2_units'].apply(parse_and_count_units)

    # Combine all unit counts to identify all unique units in the dataset
    all_unit_counts = pd.concat([player_1_units_counts, player_2_units_counts])
    all_unique_units = set(unit for counts in all_unit_counts for unit in counts)

    player_1_cols = [f'player_1_{unit}' for unit in all_unique_units]
    player_2_cols = [f'player_2_{unit}' for unit in all_unique_units]
    new_cols_df = pd.DataFrame(0, index=df_cleaned.index, columns=player_1_cols + player_2_cols)
    df_cleaned = pd.concat([df_cleaned, new_cols_df], axis=1)

    # Populate the unit count columns for each player
    for index, row in df_cleaned.iterrows():
        for unit, count in player_1_units_counts.loc[index].items():
            if unit not in ['Larva', 'Zergling', 'Drone', 'Probe', 'SCV', 'Marine', 'Baneling','Overlord', 'Roach', 'Broodling']: continue
            df_cleaned.at[index, f'player_1_{unit}'] = count
        for unit, count in player_2_units_counts.loc[index].items():
            if unit not in ['Larva', 'Zergling', 'Drone', 'Probe', 'SCV', 'Marine', 'Baneling','Overlord', 'Roach', 'Broodling']: continue
            df_cleaned.at[index, f'player_2_{unit}'] = count

    # One-hot encode the 'map' variable
    ohe = OneHotEncoder(sparse_output=False)
    map_encoded = ohe.fit_transform(df_cleaned[['map']])
    map_encoded_df = pd.DataFrame(map_encoded, columns=ohe.get_feature_names_out(['map']), index=df_cleaned.index)

    # Drop the original 'map' column and concatenate the one-hot encoded map columns
    df_cleaned = pd.concat([df_cleaned.drop('map', axis=1), map_encoded_df], axis=1)

    # Scaling numerical features (excluding 'winner', 'build', and any string columns)
    numerical_cols = [col for col in df_cleaned.columns if df_cleaned[col].dtype in ['int64', 'float64'] and col not in ['winner', 'build']]
    scaler = StandardScaler()
    df_cleaned[numerical_cols] = scaler.fit_transform(df_cleaned[numerical_cols])

    # Feature hashing for 'player_1' and 'player_2' columns
    combined_players_list = df_cleaned[['player_1', 'player_2']].values.tolist()
    fh = FeatureHasher(n_features=10, input_type='string')
    hashed_features = fh.transform(combined_players_list).toarray()
    hashed_features_df = pd.DataFrame(hashed_features, columns=[f'player_hash_{i}' for i in range(10)], index=df_cleaned.index)
    df_cleaned_final = pd.concat([df_cleaned.drop(['player_1', 'player_2'], axis=1), hashed_features_df], axis=1)
    
    
    unit_columns = [col for col in df_cleaned_final.columns if 'player_1_' in col and col.replace('player_1_', 'player_2_') in df_cleaned_final.columns and col != 'player_1_units']
    for unit_col in unit_columns:
        
        player_2_col = unit_col.replace('player_1_', 'player_2_')
        differential_col = unit_col.replace('player_1_', 'diff_')
        df_cleaned_final[differential_col] = df_cleaned_final[unit_col] - df_cleaned_final[player_2_col]
    return df_cleaned_final


app = Flask(__name__)

# Assuming the paths are defined as follows:
MODELS_DIR = Path('./models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Global variable to hold the loaded model
current_model = None

def load_latest_model():
    """
    Load the most recent model from the models directory.
    """
    model_files = list(MODELS_DIR.glob('*.pkl'))
    print(model_files)
    if not model_files:
        return None
    latest_model_file = max(model_files, key=os.path.getmtime)
    print(latest_model_file)
    return joblib.load(latest_model_file)

@app.route('/update_model', methods=['GET'])
def update_model():
    """
    Endpoint to load the most recent model into the application.
    """
    global current_model
    try:
        current_model = load_latest_model()
        if current_model is None:
            return jsonify({'error': 'No model files found.'}), 404
        return jsonify({'message': 'Model updated successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/predict', methods=['POST'])
def predict():
    json_input = request.json
    data = pd.json_normalize(json_input)  # Convert JSON to DataFrame
    data = load_and_preprocess_data(data)  # Assuming preprocessing function can handle DataFrame input
    data = feature_engineering(data)
    
    # Drop unnecessary columns for prediction
    data = data.drop(['path', 'winner', 'gameloop', 'player_1_units', 'player_2_units'], axis=1, errors='ignore')
    
    # Load the trained model
    model_path = MODELS_DIR / 'model.pkl'
    if not model_path.exists():
        return jsonify({'error': 'Model not found.'}), 404
    model = joblib.load(model_path)
    
    predictions = model.predict(data)
    return jsonify(predictions.tolist())


@app.route('/',methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/train', methods=['POST'])
def train():
    json_input = request.json
   
    file_path = DATA_DIR / f"data_{len(list(DATA_DIR.iterdir())) + 1}.json"
    with open(file_path, 'w') as file:
        json.dump(json_input, file)
    
    # Load and preprocess data from all JSON files in DATA_DIR
    dfs = []
    for file in DATA_DIR.iterdir():
        print(file)
        # df = pd.read_json(file)
        
        df = load_and_preprocess_data(file)

        print("processed: ", df.shape)
        dfs.append(df)
    full_df = pd.concat(dfs)
    print("full df: ", full_df.shape)
    full_df = feature_engineering(full_df)
    
    X = full_df.drop(['path', 'winner', 'gameloop', 'player_1_units', 'player_2_units'], axis=1, errors='ignore')
    y = full_df['winner'] - 1  # Adjust target to 0-based
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, MODELS_DIR / 'model.pkl')
    
    return jsonify({'message': 'Model trained and saved successfully.'})

if __name__ == '__main__':
    app.run(port=5000, debug=True, host='0.0.0.0')

