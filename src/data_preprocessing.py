import pandas as pd
import numpy as np
import os
import librosa
import torch

def load_data(filepath):
    """Loads the music dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data from {filepath}, shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def create_subset(df, n_per_language=500, random_state=42):
    """
    Creates a balanced subset of the data.
    """
    if 'language' not in df.columns:
        print("Warning: 'language' column not found. Returning random sample.")
        return df.sample(min(len(df), n_per_language * 2), random_state=random_state)
    
    langs = df['language'].unique()
    dfs = []
    for lang in langs:
        lang_df = df[df['language'] == lang]
        sample_size = min(len(lang_df), n_per_language)
        dfs.append(lang_df.sample(sample_size, random_state=random_state))
        
    return pd.concat(dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)

def extract_audio_features(file_path, n_mfcc=13, mean_only=False):
    """
    Extracts MFCC features from an audio file.
    Returns a flattened array of mean and std of MFCCs.
    """
    try:
        if not os.path.exists(file_path):
             return None
             
        y, sr = librosa.load(file_path, sr=None, duration=30) # Limit to 30s
        if len(y) == 0:
            return None
            
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Calculate statistics (Mean and Std) to get fixed-size vector
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent)
        cent_std = np.std(cent)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # MFCC(26) + Centroid(2) + ZCR(2) = 30 features per song.
        
        features = np.concatenate([mfcc_mean, mfcc_std, [cent_mean, cent_std, zcr_mean, zcr_std]])
        return features
        
    except Exception as e:
        # print(f"Error processing {file_path}: {e}")
        return None

def process_audio_data(df, audio_col='audio_path'):
    """
    Process DataFrame to extract audio features.
    """
    features_list = []
    valid_indices = []
    
    print(f"Extracting features from {len(df)} files...")
    for idx, row in df.iterrows():
        path = row[audio_col]
        # Clean path if needed (sometimes windows paths in CSV need care)
        # CSV example: e:\uni\...
        
        feat = extract_audio_features(path)
        if feat is not None:
            features_list.append(feat)
            valid_indices.append(idx)
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(df)}")
            
    if not features_list:
        return None, None
        
    X = np.array(features_list)
    # Normalize features for VAE training.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df.loc[valid_indices].reset_index(drop=True)

def extract_spectrogram(file_path, target_shape=(64, 64)):
    """
    Extracts Mel Spectrogram and resizes/pads to target_shape.
    """
    try:
        if not os.path.exists(file_path):
            return None
        
        # Load audio (limit duration)
        y, sr = librosa.load(file_path, sr=22050, duration=30)
        
        # Compute Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_shape[0])
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize/Crop to target width
        # If shorter, pad. If longer, crop or resize.
        # Simple resize using cv2 is strictly not "correct" for spectrograms but okay for VAE features.
        # Or just array slicing/padding.
        
        current_width = mel_spec_db.shape[1]
        target_width = target_shape[1]
        
        if current_width < target_width:
            pad_width = target_width - current_width
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
             # Center crop or just take first N
            mel_spec_db = mel_spec_db[:, :target_width]
            
        # Normalize to [0, 1]
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Add channel dim (1, 64, 64)
        return mel_spec_db[np.newaxis, :, :]
        
    except Exception as e:
        return None

def process_spectrograms(df, audio_col='audio_path'):
    specs_list = []
    valid_indices = []
    
    print(f"Extracting Spectrograms from {len(df)} files...")
    for idx, row in df.iterrows():
        path = row[audio_col]
        spec = extract_spectrogram(path)
        if spec is not None:
            specs_list.append(spec)
            valid_indices.append(idx)
            
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(df)}")
            
    if not specs_list:
        return None, None
        
    X = np.array(specs_list) # (N, 1, 64, 64)
    return torch.FloatTensor(X), df.loc[valid_indices].reset_index(drop=True)

from sklearn.feature_extraction.text import TfidfVectorizer

def process_hybrid_data(df, audio_col='audio_path', text_col='lyrics', max_features=1000):
    """
    Extracts aligned Audio (MFCC) and Text (TF-IDF) features.
    """
    # First extract audio features (most likely to fail)
    print("Extracting Audio features for Hybrid...")
    X_audio, df_valid = process_audio_data(df, audio_col=audio_col)
    
    if X_audio is None:
        return None, None, None
        
    # Now extract Text features from the VALID df
    print("Extracting Text features (TF-IDF)...")
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    # Simple cleaning: fillna
    texts = df_valid[text_col].fillna("").tolist()
    X_text = tfidf.fit_transform(texts).toarray()
    
    # Use raw TF-IDF (approx 0-1 range).
    
    return X_audio, X_text, df_valid

from sklearn.preprocessing import OneHotEncoder

def process_multimodal_data(df, audio_col='audio_path', text_col='lyrics', genre_col='genre', max_features=1000):
    """
    Extracts Audio, Text, and Genre features.
    Returns concatenated features for baselines, and individual features for VAEs.
    """
    # 1. Audio & Text (re-use hybrid pipeline)
    X_audio, X_text, df_valid = process_hybrid_data(df, audio_col, text_col, max_features)
    
    if X_audio is None:
        return None, None, None, None
        
    # 2. Genre (One-Hot)
    print("Encoding Genre features...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Use df_valid to match rows
    genres = df_valid[[genre_col]].fillna("Unknown")
    X_genre = encoder.fit_transform(genres)
    
    # Concatenate for baselines
    # Scale Audio is already scaled. Text is TF-IDF. Genre is 0/1.
    # To treat them equally, maybe scale all?
    # For now, just concat.
    X_fused = np.hstack([X_audio, X_text, X_genre])
    
    return X_fused, X_audio, X_text, X_genre, df_valid
