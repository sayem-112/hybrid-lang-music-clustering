
import os
import sys
import torch
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_preprocessing import load_data, create_subset, process_audio_data
from src.vae import VAE
from src.analysis import train_vae, extract_features, perform_clustering, visualize_clusters, evaluate_clustering, run_baseline

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, 'data', 'lyrics', 'hybrid_songs.csv')
    results_dir = os.path.join(base_dir, 'results', 'easy')
    os.makedirs(results_dir, exist_ok=True)
    
    print("=== Step 1: Data Preparation (Audio) ===")
    df = load_data(data_path)
    if df is None:
        return
        
    print("Using full dataset (User requested all songs)...")
    df_subset = df
    print(f"Total songs to process: {len(df_subset)}")
    
    print("Extracting Audio Features (MFCCs)...")
    X_audio, df_final = process_audio_data(df_subset, audio_col='audio_path')
    
    if X_audio is None or len(X_audio) == 0:
        print("Failed to extract any audio features. Check file paths.")
        return
        
    print(f"Feature shape: {X_audio.shape}")
    print(f"Final dataframe shape: {df_final.shape}")
    input_dim = X_audio.shape[1]
    
    print("\n=== Step 2: VAE Training ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    latent_dim = 8
    
    vae = VAE(input_dim=input_dim, hidden_dim=64, latent_dim=latent_dim)
    
    vae = train_vae(vae, X_audio, epochs=50, batch_size=16, device=device)
    
    print("\n=== Step 3: Feature Extraction (VAE) ===")
    vae_features = extract_features(vae, X_audio, device=device)
    print(f"Latent features shape: {vae_features.shape}")
    
    print("\n=== Step 4: Clustering (VAE) ===")
    n_clusters = 5
    vae_labels, _ = perform_clustering(vae_features, n_clusters=n_clusters)
    
    # Eval
    vae_sil, vae_ch = evaluate_clustering(vae_features, vae_labels)
    print(f"VAE clustering metrics: Silhouette={vae_sil:.4f}, Calinski-Harabasz={vae_ch:.4f}")
    
    # Visualize
    visualize_clusters(vae_features, vae_labels, 
                       title=f'VAE Latent Space (Audio K={n_clusters})', 
                       save_path=os.path.join(results_dir, 'vae_audio_clusters.png'))
    
    # Visualize by Language
    lang_labels = df_final['language'].astype('category').cat.codes
    visualize_clusters(vae_features, lang_labels, 
                       title='VAE Latent Space (Audio Color by Language)', 
                       save_path=os.path.join(results_dir, 'vae_audio_language.png'))

    print("\n=== Step 5: Baseline Comparison (PCA) ===")
    baseline_features, baseline_labels, base_sil, base_ch = run_baseline(X_audio, n_clusters=n_clusters)
    print(f"Baseline metrics: Silhouette={base_sil:.4f}, Calinski-Harabasz={base_ch:.4f}")
    
    visualize_clusters(baseline_features, baseline_labels,
                       title=f'Baseline PCA (Audio K={n_clusters})', 
                       save_path=os.path.join(results_dir, 'baseline_audio_clusters.png'))
                       
    print("\n=== Summary Evaluation ===")
    results_df = pd.DataFrame({
        'Method': ['Audio VAE', 'Baseline (Audio PCA)'],
        'Silhouette Score': [vae_sil, base_sil],
        'Calinski-Harabasz': [vae_ch, base_ch]
    })
    print(results_df)
    
    # Save metrics to CSV
    csv_path = os.path.join(results_dir, 'clustering_metrics.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

if __name__ == '__main__':
    main()
