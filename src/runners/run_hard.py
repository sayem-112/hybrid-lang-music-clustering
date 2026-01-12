
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_preprocessing import load_data, process_multimodal_data
from src.models.beta_vae import BetaVAE, beta_vae_loss_function
from src.models.autoencoder import Autoencoder, autoencoder_loss_function
from src.analysis import perform_clustering, visualize_clusters, evaluate_hard_metrics, run_spectral_clustering

def train_beta_vae(model, data, epochs=50, batch_size=32, device='cpu', beta=4.0):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Starting Beta-VAE (beta={beta}) training on {device}...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = beta_vae_loss_function(recon_x, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader.dataset):.4f}")
    return model

def train_autoencoder(model, data, epochs=50, batch_size=32, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Starting Autoencoder training on {device}...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x = model(x)
            loss = autoencoder_loss_function(recon_x, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader.dataset):.4f}")
    return model

def visualize_reconstruction(model, data, title, save_path, device='cpu', is_vae=True):
    """
    Visualizes original vs reconstructed features as a heatmap.
    """
    model.eval()
    with torch.no_grad():
        # Take a few samples
        samples = data[:10].to(device)
        if is_vae:
            recon, _, _ = model(samples)
        else:
            recon = model(samples)
        
    samples = samples.cpu().numpy()
    recon = recon.cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(samples, cmap='viridis', cbar=False)
    plt.title("Original Input")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(recon, cmap='viridis', cbar=False)
    plt.title(f"Reconstruction ({title})")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, 'data', 'lyrics', 'hybrid_songs.csv')
    results_dir = os.path.join(base_dir, 'results', 'hard')
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    df = load_data(data_path)
    if df is None: return
    
    print("Using full dataset for Hard Task...")
    
    print("\n=== Multi-modal Feature Extraction ===")
    X_fused, X_audio, X_text, X_genre, df_final = process_multimodal_data(
        df, audio_col='audio_path', text_col='lyrics', genre_col='genre', max_features=100
    )
    
    if X_fused is None:
        print("Failed to extract multimodal data.")
        return
        
    print(f"Fused Data Shape: {X_fused.shape}")
    input_dim = X_fused.shape[1]
    
    true_labels = df_final['language'].astype('category').cat.codes.values
    
    X_tensor = torch.FloatTensor(X_fused)
    
    results = []
    
    print("\n=== Training Beta-VAE ===")
    beta_vae = BetaVAE(input_dim=input_dim, hidden_dim=256, latent_dim=16)
    beta_vae = train_beta_vae(beta_vae, X_tensor, epochs=50, batch_size=32, device=device, beta=4.0)
    
    beta_vae.eval()
    with torch.no_grad():
        mu, _ = beta_vae.encode(X_tensor.to(device))
        beta_features = mu.cpu().numpy()
        
    beta_labels, _ = perform_clustering(beta_features, n_clusters=5)
    beta_metrics = evaluate_hard_metrics(beta_features, beta_labels, true_labels)
    beta_metrics['Method'] = "Beta-VAE"
    results.append(beta_metrics)
    
    visualize_clusters(beta_features, beta_labels, "Beta-VAE Latent Space", 
                       os.path.join(results_dir, "beta_vae_clusters.png"))
    
    visualize_reconstruction(beta_vae, X_tensor, "Beta-VAE", 
                             os.path.join(results_dir, "beta_vae_recon.png"), device=device)

    print("\n=== Training Autoencoder Baseline ===")
    ae = Autoencoder(input_dim=input_dim, hidden_dim=256, latent_dim=16)
    ae = train_autoencoder(ae, X_tensor, epochs=50, batch_size=32, device=device)
    
    # Extract Latent
    ae.eval()
    with torch.no_grad():
        ae_features, _ = ae.encode(X_tensor.to(device)) # AE encode returns (z, None)
        ae_features = ae_features.cpu().numpy()
        
    # Cluster
    ae_labels, _ = perform_clustering(ae_features, n_clusters=5)
    ae_metrics = evaluate_hard_metrics(ae_features, ae_labels, true_labels)
    ae_metrics['Method'] = "Autoencoder"
    results.append(ae_metrics)
    
    visualize_clusters(ae_features, ae_labels, "Autoencoder Latent Space", 
                       os.path.join(results_dir, "autoencoder_clusters.png"))
                       
    visualize_reconstruction(ae, X_tensor, "Autoencoder", 
                             os.path.join(results_dir, "autoencoder_recon.png"), device=device, is_vae=False)

    print("\n=== Spectral Clustering Baseline ===")
    spec_labels = run_spectral_clustering(X_fused, n_clusters=5)
    spec_metrics = evaluate_hard_metrics(X_fused, spec_labels, true_labels)
    spec_metrics['Method'] = "Spectral Clustering"
    results.append(spec_metrics)
    
    visualize_clusters(X_fused, spec_labels, "Spectral Clustering (Input Space)", 
                       os.path.join(results_dir, "spectral_clusters.png"))

    print("\n=== Hard Task Results ===")
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(os.path.join(results_dir, "hard_metrics.csv"), index=False)

if __name__ == '__main__':
    main()
