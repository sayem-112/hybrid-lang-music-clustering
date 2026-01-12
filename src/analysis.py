
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import os

def train_vae(model, data, epochs=50, batch_size=32, learning_rate=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    if hasattr(data, "toarray"):
        data_tensor = torch.FloatTensor(data.toarray())
    else:
        data_tensor = torch.FloatTensor(data)
        
    dataset = torch.utils.data.TensorDataset(data_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Starting training on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = from_vae_import_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader.dataset):.4f}")
            
    return model


def from_vae_import_loss(recon_x, x, mu, logvar):
    from src.vae import loss_function
    return loss_function(recon_x, x, mu, logvar)

def extract_features(model, data, device='cpu'):
    model.eval()
    if hasattr(data, "toarray"):
        data_tensor = torch.FloatTensor(data.toarray())
    else:
        data_tensor = torch.FloatTensor(data)
    
    data_tensor = data_tensor.to(device)
    with torch.no_grad():
        mu, _ = model.encode(data_tensor)
    return mu.cpu().numpy()

def perform_clustering(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    return labels, kmeans

def visualize_clusters(features, labels, title, save_path=None):
    if features.shape[1] > 50:
        pca = PCA(n_components=50)
        features_pca = pca.fit_transform(features)
    else:
        features_pca = features
        
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(features_pca)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedded[:,0], y=embedded[:,1], hue=labels, palette='viridis', legend='full')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.close()

def evaluate_clustering(features, labels):
    sil_score = silhouette_score(features, labels)
    ch_score = calinski_harabasz_score(features, labels)
    return sil_score, ch_score

def run_baseline(data, n_clusters=5, n_components=8):
    print("Running Baseline (PCA + K-Means)...")
    if hasattr(data, "toarray"):
        data_dense = data.toarray()
    else:
        data_dense = data
        
    # Ensure n_components <= n_features
    n_features = data_dense.shape[1]
    n_comps = min(n_components, n_features)
    print(f"PCA using n_components={n_comps}")
    
    pca = PCA(n_components=n_comps)
    reduced_data = pca.fit_transform(data_dense)
    
    labels, _ = perform_clustering(reduced_data, n_clusters=n_clusters)
    
    sil, ch = evaluate_clustering(reduced_data, labels)
    return reduced_data, labels, sil, ch

from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import davies_bouldin_score, adjusted_rand_score

def perform_advanced_clustering(features, n_clusters=5):
    results = {}
    
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg.fit_predict(features)
    results['Agglomerative'] = agg_labels
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    db_labels = dbscan.fit_predict(features)
    results['DBSCAN'] = db_labels
    
    return results

def evaluate_advanced_metrics(features, labels, true_labels=None):
    metrics = {}
    
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        return {'Silhouette': -1, 'Calinski-Harabasz': -1, 'Davies-Bouldin': -1, 'ARI': -1}
        
    metrics['Silhouette'] = silhouette_score(features, labels)
    metrics['Calinski-Harabasz'] = calinski_harabasz_score(features, labels)
    metrics['Davies-Bouldin'] = davies_bouldin_score(features, labels)
    
    if true_labels is not None:
        metrics['ARI'] = adjusted_rand_score(true_labels, labels)
    else:
        metrics['ARI'] = None
        
    return metrics

from sklearn.metrics import normalized_mutual_info_score, confusion_matrix
from sklearn.cluster import SpectralClustering

def compute_purity(y_true, y_pred):
    contingency_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def evaluate_hard_metrics(features, labels, true_labels):
    metrics = evaluate_advanced_metrics(features, labels, true_labels)
    if true_labels is not None:
        metrics['NMI'] = normalized_mutual_info_score(true_labels, labels)
        metrics['Purity'] = compute_purity(true_labels, labels)
    else:
        metrics['NMI'] = None
        metrics['Purity'] = None
        
    return metrics

def run_spectral_clustering(data, n_clusters=5):
    print("Running Spectral Clustering...")
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42, n_jobs=-1)
    labels = spectral.fit_predict(data)
    return labels
