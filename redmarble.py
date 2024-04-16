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
from sentence_transformers import SentenceTransformer


app = Flask(__name__)

# Define paths for data and models
DATA_DIR = Path('./data')
MODELS_DIR = Path('./models')
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)



VESPENE_UNITS = ["Assimilator", "Extractor", "Refinery"]

SUPPLY_UNITS = ["Overlord", "Overseer", "Pylon", "SupplyDepot"]

WORKER_UNITS = ["Drone", "Probe", "SCV", "MULE"]

BASE_UNITS = ["CommandCenter", "Nexus", "Hatchery", "Lair", "Hive", "PlanetaryFortress", "OrbitalCommand"]

GROUND_UNITS = ["Barracks", "Factory", "GhostAcademy", "Armory", "RoboticsBay", "RoboticsFacility", "TemplarArchive",
                "DarkShrine", "WarpGate", "SpawningPool", "RoachWarren", "HydraliskDen", "BanelingNest", "UltraliskCavern",
                "LurkerDen", "InfestationPit"]

AIR_UNITS = ["Starport", "FusionCore", "RoboticsFacility", "Stargate", "FleetBeacon", "Spire", "GreaterSpire"]

TECH_UNITS = ["EngineeringBay", "Armory", "GhostAcademy", "TechLab", "FusionCore", "Forge", "CyberneticsCore",
              "TwilightCouncil", "RoboticsFacility", "RoboticsBay", "FleetBeacon", "TemplarArchive", "DarkShrine",
              "SpawningPool", "RoachWarren", "HydraliskDen", "BanelingNest", "UltraliskCavern", "LurkerDen", "Spire",
              "GreaterSpire", "EvolutionChamber", "InfestationPit"]

ARMY_UNITS = ["Marine", "Colossus", "InfestorTerran", "Baneling", "Mothership", "MothershipCore", "Changeling", "SiegeTank", "Viking", "Reaper",
              "Ghost", "Marauder", "Thor", "Hellion", "Hellbat", "Cyclone", "Liberator", "Medivac", "Banshee", "Raven", "Battlecruiser", "Nuke", "Zealot",
              "Stalker", "HighTemplar", "Disruptor", "DarkTemplar", "Sentry", "Phoenix", "Carrier", "Oracle", "VoidRay", "Tempest", "WarpPrism", "Observer",
              "Immortal", "Adept", "Zergling", "Overlord", "Hydralisk", "Mutalisk", "Ultralisk", "Roach", "Infestor", "Corruptor",
              "BroodLord", "Queen", "Overseer", "Archon", "Broodling", "InfestedTerran", "Ravager", "Viper", "SwarmHost"]

ARMY_AIR = ["Mothership", "MothershipCore", "Viking", "Liberator", "Medivac", "Banshee", "Raven", "Battlecruiser",
            "Viper", "Mutalisk", "Phoenix", "Oracle", "Carrier", "VoidRay", "Tempest", "Observer", "WarpPrism", "BroodLord",
            "Corruptor", "Observer", "Overseer"]


train_cols = ['build', 'p1_unique_units', 'p2_unique_units', 'unique_diff',
       'p1_unit_count', 'p2_unit_count', 'unit_count_diff', 'p1_SUPPLY_UNITS',
       'p1_WORKER_UNITS', 'p1_ARMY_UNITS', 'p1_ARMY_AIR', 'p1_VESPENE_UNITS',
       'p1_TECH_UNITS', 'p1_GROUND_UNITS', 'p1_AIR_UNITS', 'p2_SUPPLY_UNITS',
       'p2_WORKER_UNITS', 'p2_ARMY_UNITS', 'p2_ARMY_AIR', 'p2_VESPENE_UNITS',
       'p2_TECH_UNITS', 'p2_GROUND_UNITS', 'p2_AIR_UNITS',
       'p1_InvisibleTargetDummy', 'p1_Larva', 'p1_Zergling',
       'p1_BroodlingEscort', 'p1_Drone', 'p1_Broodling', 'p1_Baneling',
       'p1_CreepTumorBurrowed', 'p2_InvisibleTargetDummy', 'p2_Larva',
       'p2_Zergling', 'p2_BroodlingEscort', 'p2_Drone', 'p2_Broodling',
       'p2_Baneling', 'p2_CreepTumorBurrowed', 'map_2000 Atmospheres LE',
       'map_Beckett Industries LE', 'map_Blackburn LE', 'map_Cosmic Sapphire',
       'map_Data-C', 'map_Deathaura LE', 'map_Ephemeron LE',
       'map_Eternal Empire LE', 'map_Ever Dream LE', 'map_Inside and Out',
       'map_Jagannatha LE', 'map_Lightshade LE', 'map_Moondance',
       'map_Nightshade LE', 'map_Oxide LE', 'map_Pillars of Gold LE',
       'map_Romanticide LE', 'map_Simulacrum LE', 'map_Submarine LE',
       'map_Triton LE', 'map_Tropical Sacrifice', 'map_Zen LE',
       'map_[ESL] Cosmic Sapphire', 'map_[ESL] Data-C',
       'map_[ESL] Inside and Out', 'map_[ESL] Moondance',
       'map_[ESL] Stargazers', 'map_[ESL] Tropical Sacrifice',
       'map_[ESL] Waterfall', 'map_others']


scale_cols = ['p2_Zergling', 'p2_BroodlingEscort', 'p1_GROUND_UNITS', 'p1_unique_units', 'p2_ARMY_AIR', 'p2_unit_count', 'p1_BroodlingEscort', 'p1_AIR_UNITS', 'p1_unit_count', 'p1_Broodling', 'p2_AIR_UNITS', 'p2_VESPENE_UNITS', 'p2_unique_units', 'p2_GROUND_UNITS', 'p1_SUPPLY_UNITS', 'p2_WORKER_UNITS', 'p1_ARMY_UNITS', 'p2_CreepTumorBurrowed', 'p1_Baneling', 'p1_Zergling', 'build', 'p2_TECH_UNITS', 'p2_Larva', 'p1_InvisibleTargetDummy', 'unique_diff', 'p2_ARMY_UNITS', 'p1_ARMY_AIR', 'p2_InvisibleTargetDummy', 'unit_count_diff', 'p2_Drone', 'p2_Baneling', 'p1_CreepTumorBurrowed', 'p1_WORKER_UNITS', 'p1_TECH_UNITS', 'p1_Larva', 'p1_VESPENE_UNITS', 'p1_Drone', 'p2_Broodling', 'p2_SUPPLY_UNITS']

def count_unit_type(player_units):
    count_dict = {}
    unit_types = ['SUPPLY_UNITS','WORKER_UNITS','ARMY_UNITS','ARMY_AIR', 'VESPENE_UNITS', 'TECH_UNITS', 'GROUND_UNITS', 'AIR_UNITS']

    for unit_type in unit_types:
        count_dict[unit_type] = 0
        
    for unit in player_units.keys():
        if unit in VESPENE_UNITS: 
            count_dict['VESPENE_UNITS'] = count_dict['VESPENE_UNITS'] + player_units[unit]
        if unit in AIR_UNITS:
            count_dict['AIR_UNITS'] = count_dict['AIR_UNITS'] + player_units[unit]
        
        if unit in TECH_UNITS:
            count_dict['TECH_UNITS'] = count_dict['TECH_UNITS'] + player_units[unit]
            
        if unit in GROUND_UNITS:
            count_dict['GROUND_UNITS'] = count_dict['GROUND_UNITS'] + player_units[unit]

        if unit in SUPPLY_UNITS:
            count_dict['SUPPLY_UNITS'] = count_dict['SUPPLY_UNITS'] + player_units[unit]

        if unit in WORKER_UNITS:
            count_dict['WORKER_UNITS'] = count_dict['WORKER_UNITS'] + player_units[unit]

        if unit in ARMY_UNITS:
            count_dict['ARMY_UNITS'] = count_dict['ARMY_UNITS'] + player_units[unit]

        if unit in ARMY_AIR:
            count_dict['ARMY_AIR'] = count_dict['ARMY_AIR'] + player_units[unit]
            
    # total = sum(count_dict.values())
    # if count_dict['ARMY_UNITS'] != 0: count_dict['ARMY_UNITS'] = round(count_dict['ARMY_UNITS']/total, 2)
    # if count_dict['WORKER_UNITS'] != 0: count_dict['WORKER_UNITS'] = round(count_dict['WORKER_UNITS']/total, 2)

    # if count_dict['SUPPLY_UNITS'] != 0: count_dict['SUPPLY_UNITS'] =round(count_dict['SUPPLY_UNITS']/total, 2)
    # if count_dict['ARMY_AIR'] != 0: count_dict['ARMY_AIR'] = round(count_dict['ARMY_AIR']/total, 2)
    return count_dict
        
        


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


def map_filter(value):
    map_names = ['Romanticide LE', 'Oxide LE', 'Lightshade LE', '2000 Atmospheres LE',
       'Jagannatha LE', '[ESL] Data-C', '[ESL] Inside and Out', 'Deathaura LE',
       'Pillars of Gold LE', 'Blackburn LE', '[ESL] Cosmic Sapphire',
       'Nightshade LE', 'Eternal Empire LE', '[ESL] Moondance',
       '[ESL] Tropical Sacrifice', 'Data-C', 'Inside and Out', 'Simulacrum LE',
       '[ESL] Waterfall', 'Moondance', 'Ephemeron LE', 'Triton LE', 'Zen LE',
       'Ever Dream LE', 'Cosmic Sapphire', '[ESL] Stargazers',
       'Beckett Industries LE', 'Tropical Sacrifice', 'Submarine LE']
    
    if value in map_names: return value
    else: return 'others'

def preprocess_data(df):
    df = df[df['map'] != 'TEST__DOCUMENT']
    df['player_1'] = df['player_1'].str.replace('&lt;', '<').str.replace('&gt;', '>').str.replace('<sp/>', '')
    df['player_2'] = df['player_2'].str.replace('&lt;', '<').str.replace('&gt;', '>').str.replace('<sp/>', '')
    # df.drop(['p1_embedding', 'p2_embedding'], axis=1, inplace=True)
    cols = ['map', 'player_1_units', 'player_1', 'player_2_units', 'player_2', 'build', 'path']
    df = df[cols]
    return df


def get_unit_cols(df):
    df['p1_unique_units'] = [len(set(x)) for x in df['player_1_units'].values]
    df['p2_unique_units'] = [len(set(x)) for x in df['player_2_units'].values]
    df['unique_diff'] = df['p1_unique_units'] - df['p2_unique_units']


    df['p1_unit_count'] = [len(x) for x in df['player_1_units'].values]
    df['p2_unit_count'] = [len(x) for x in df['player_2_units'].values]
    df['unit_count_diff'] = df['p1_unit_count'] - df['p2_unit_count']
    return df
def get_unit_category_counts(df):
    player_1_units_counts = df['player_1_units'].apply(parse_and_count_units)
    player_2_units_counts = df['player_2_units'].apply(parse_and_count_units)

    p1_unit_types = pd.json_normalize(player_1_units_counts.apply(count_unit_type))
    p1_unit_types.columns = [f'p1_{col}' for col in p1_unit_types]

    p2_unit_types = pd.json_normalize(player_2_units_counts.apply(count_unit_type))
    p2_unit_types.columns = [f'p2_{col}' for col in p2_unit_types]

    df[p1_unit_types.columns] = p1_unit_types.values
    df[p2_unit_types.columns] = p2_unit_types.values
    
    return df

def get_unit_counts(df):
    create_unit_cols = ['InvisibleTargetDummy', 'Larva', 'Zergling', 'BroodlingEscort', 'Drone', 'Broodling', 'Baneling', 'CreepTumorBurrowed']

    player_1_units_counts = df['player_1_units'].apply(parse_and_count_units)
    player_2_units_counts = df['player_2_units'].apply(parse_and_count_units)

    p1_c = pd.json_normalize(player_1_units_counts).fillna(0)
    p1_c = p1_c[[col for col in create_unit_cols if col in p1_c.columns]]
    p1_c.columns = [f'p1_{col}' for col in p1_c.columns]

    p2_c = pd.json_normalize(player_2_units_counts).fillna(0)
    p2_c = p2_c[[col for col in create_unit_cols if col in p2_c.columns]]
    p2_c.columns = [f'p2_{col}' for col in p2_c.columns]

    p_unit_counts = pd.concat([p1_c, p2_c], axis=1).fillna(0)
    df[p_unit_counts.columns] = p_unit_counts.values
    
    return df


def get_embedding(text, embeding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')):
    return embeding_model.encode(text)

def parse_and_count_units(unit_str):
    unit_counts = Counter(unit_str)
    return unit_counts

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

import pickle

# model, target_encoder, scaler = joblib.load('model_and_preprocessing.pkl')
with open('models/model.pkl', 'rb') as file:  
    model = pickle.load(file)
    
with open('models/encoder.pkl', 'rb') as file:  
    encoder = pickle.load(file)
    
with open('models/scaler.pkl', 'rb') as file:  
    scaler = pickle.load(file)
    
@app.route('/predict', methods=['POST'])
def predict():
    json_input = json.loads(request.json)
    df = pd.DataFrame(json_input)  # Convert JSON to DataFrame
    df = df.reset_index(drop=True)
    
    processed_df = preprocess_data(df)
    processed_df = get_unit_cols(processed_df)
    processed_df = get_unit_category_counts(processed_df)
    processed_df = get_unit_counts(processed_df)
    processed_df['map'] = processed_df['map'].apply(map_filter)
    
    numerical_features = processed_df.select_dtypes(include='number').columns.tolist()
    categorical_features = processed_df.select_dtypes(include= 'object').columns.tolist()
    
    processed_df[scale_cols] = scaler.transform(processed_df[scale_cols])
    
    map_encoded = encoder.transform(processed_df[['map']])
    map_encoded_df = pd.DataFrame(map_encoded, columns=encoder.get_feature_names_out(['map']), index=processed_df.index)
    processed_df = pd.concat([processed_df.drop('map', axis=1), map_encoded_df], axis=1)
    processed_df = processed_df[train_cols]
    # print(processed_df.columns)
    predictions = model.predict(processed_df)
    
    return jsonify(predictions.tolist())
    

@app.route('/predict', methods=['POST'])
def predict_old():
    embeding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    json_input = json.loads(request.json)
    
    df = pd.DataFrame(json_input)  # Convert JSON to DataFrame
    df = df.reset_index(drop=True)
    
    df['p1_embedding'] = df['player_1_units'].apply(lambda x: get_embedding(','.join(x), embeding_model=embeding_model))
    df['p2_embedding'] = df['player_2_units'].apply(lambda x: get_embedding(','.join(x), embeding_model=embeding_model))
    df['build'] = df['build'].astype(str)
    df['embed_dif'] = df['p1_embedding'] - df['p2_embedding']
    
    player_1_units_counts = df['player_1_units'].apply(parse_and_count_units)
    player_2_units_counts = df['player_2_units'].apply(parse_and_count_units)


    p1_unit_types = pd.json_normalize(player_1_units_counts.apply(count_unit_type))
    p1_unit_types.columns = [f'p1_{col}' for col in p1_unit_types]

    p2_unit_types = pd.json_normalize(player_2_units_counts.apply(count_unit_type))
    p2_unit_types.columns = [f'p2_{col}' for col in p2_unit_types]
    
    # if 'winner' in df.columns:
    #     final_df = df[['embed_dif', 'map', 'build', 'total_gameloops', 'winner']].copy()
    #     final_df.columns = ['embeddings', 'map', 'build', 'total_gameloops', 'winner']
    # else:
    final_df = df[['embed_dif', 'map', 'build', 'total_gameloops']].copy()
    final_df.columns = ['embeddings', 'map', 'build', 'total_gameloops']
    
    final_df = pd.concat([final_df, p1_unit_types, p2_unit_types], axis=1)
    final_df.columns = final_df.columns.astype(str)
    final_df = final_df.dropna()
    
    embedding_df = pd.DataFrame(final_df['embeddings'].tolist())
    final_df = pd.concat([final_df.drop('embeddings', axis=1), embedding_df], axis=1)
    final_df.columns = final_df.columns.astype(str)
    cols = [str(col) for col in final_df.columns if col not in  ['winner', 'map', 'build']]
    final_df[cols] = scaler.transform(final_df[cols])
    
    target_encoding_cols = ['map', 'build']
    final_df[target_encoding_cols] = target_encoder.transform(final_df[target_encoding_cols])
    
    
    col_order = ['map', 'build', 'total_gameloops', 'p1_SUPPLY_UNITS', 'p1_WORKER_UNITS', 'p1_ARMY_UNITS', 'p1_ARMY_AIR', 'p2_SUPPLY_UNITS', 'p2_WORKER_UNITS', 'p2_ARMY_UNITS', 'p2_ARMY_AIR', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', '300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '330', '331', '332', '333', '334', '335', '336', '337', '338', '339', '340', '341', '342', '343', '344', '345', '346', '347', '348', '349', '350', '351', '352', '353', '354', '355', '356', '357', '358', '359', '360', '361', '362', '363', '364', '365', '366', '367', '368', '369', '370', '371', '372', '373', '374', '375', '376', '377', '378', '379', '380', '381', '382', '383']
    predictions = model.predict(final_df[col_order])
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

