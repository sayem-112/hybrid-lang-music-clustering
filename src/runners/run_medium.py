
import os
import sys
import torch
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_preprocessing import load_data, process_spectrograms, process_hybrid_data
from src.models.cvae import CVAE, cvae_loss_function
from src.models.hybrid_vae import HybridVAE, hybrid_loss_function
from src.analysis import perform_clustering, visualize_clusters, evaluate_clustering, perform_advanced_clustering, evaluate_advanced_metrics

def train_cvae(model, data, epochs=30, batch_size=16, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Starting CVAE training on {device}...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = cvae_loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader.dataset):.4f}")
    return model

def train_hybrid_vae(model, audio_data, text_data, epochs=30, batch_size=16, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    # Check shape
    if not isinstance(audio_data, torch.Tensor):
        audio_data = torch.FloatTensor(audio_data)
    if not isinstance(text_data, torch.Tensor):
        text_data = torch.FloatTensor(text_data)
        
    dataset = torch.utils.data.TensorDataset(audio_data, text_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Starting Hybrid VAE training on {device}...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x_a = batch[0].to(device)
            x_t = batch[1].to(device)
            
            optimizer.zero_grad()
            recon_a, recon_t, mu, logvar = model(x_a, x_t)
            
            loss = hybrid_loss_function(recon_a, x_a, recon_t, x_t, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader.dataset):.4f}")
    return model

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, 'data', 'lyrics', 'hybrid_songs.csv')
    results_dir = os.path.join(base_dir, 'results', 'medium')
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    df = load_data(data_path)
    if df is None: return
    
    print("Creating subset for Medium Task (Spectrograms are large)...")

    
    print("\n=== CVAE: Spectrogram Feature Extraction ===")
    X_specs, df_specs = process_spectrograms(df, audio_col='audio_path')
    
    if X_specs is not None:
        print(f"Spectrograms shape: {X_specs.shape}") # (N, 1, 64, 64)
        cvae = CVAE(latent_dim=16)
        cvae = train_cvae(cvae, X_specs, epochs=30, batch_size=16, device=device)
        
        # Extract Latent
        cvae.eval()
        with torch.no_grad():
            cvae_mu, _ = cvae.encode(X_specs.to(device))
            cvae_features = cvae_mu.cpu().numpy()
            
        # Cluster CVAE
        cvae_labels, _ = perform_clustering(cvae_features, n_clusters=5)
        visualize_clusters(cvae_features, cvae_labels, "CVAE Latent Space (Spectrograms)", 
                           os.path.join(results_dir, "cvae_clusters.png"))
        
        cvae_metrics = evaluate_advanced_metrics(cvae_features, cvae_labels)
        print("CVAE Metrics:", cvae_metrics)
    else:
        print("Skipping CVAE (No spectrograms extracted)")
        cvae_metrics = {}

    print("\n=== Hybrid VAE: Audio + Lyrics ===")
    X_audio, X_text, df_hybrid = process_hybrid_data(df, audio_col='audio_path', text_col='lyrics')
    
    if X_audio is not None and X_text is not None:
        hybrid_vae = HybridVAE(audio_dim=X_audio.shape[1], text_dim=X_text.shape[1], latent_dim=16)
        hybrid_vae = train_hybrid_vae(hybrid_vae, X_audio, X_text, epochs=30, batch_size=16, device=device)
        
        # Extract Latent
        hybrid_vae.eval()
        with torch.no_grad():
            mu, _ = hybrid_vae.encode(torch.FloatTensor(X_audio).to(device), torch.FloatTensor(X_text).to(device))
            hybrid_features = mu.cpu().numpy()
            
        print(f"Hybrid Features shape: {hybrid_features.shape}")
        
        print("\n=== Advanced Clustering Comparison (Hybrid VAE) ===")
        adv_results = perform_advanced_clustering(hybrid_features, n_clusters=5)
        
        labels_kmeans, _ = perform_clustering(hybrid_features, n_clusters=5)
        adv_results['K-Means'] = labels_kmeans
        
        true_labels = df_hybrid['language'].astype('category').cat.codes.values
        
        comparison_data = []
        
        for name, labels in adv_results.items():
            metrics = evaluate_advanced_metrics(hybrid_features, labels, true_labels=true_labels)
            metrics['Method'] = f"Hybrid + {name}"
            comparison_data.append(metrics)
            
            visualize_clusters(hybrid_features, labels, f"Hybrid VAE + {name}", 
                               os.path.join(results_dir, f"hybrid_{name}_clusters.png"))
            
        if 'Silhouette' in cvae_metrics:
            cvae_metrics['Method'] = "CVAE + K-Means"
            comparison_data.append(cvae_metrics)
        
        results_df = pd.DataFrame(comparison_data)
        print("\n=== Medium Task Results ===")
        print(results_df)
        results_df.to_csv(os.path.join(results_dir, "medium_metrics.csv"), index=False)
        
    else:
        print("Failed to process Hybrid data")

if __name__ == '__main__':
    main()
