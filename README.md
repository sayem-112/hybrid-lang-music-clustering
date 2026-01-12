# Multi-Modal Music Clustering with VAEs

This project implements various Variational Autoencoder (VAE) architectures to cluster music data using audio features (MFCCs, Spectrograms) and lyrics (TF-IDF). It compares VAE-based approaches against baselines like PCA + K-Means and Spectral Clustering.

## Project Structure

- `src/`: Source code for models, data processing, and analysis.
  - `models/`: VAE implementations (Basic VAE, CVAE, Hybrid VAE, Beta-VAE, Autoencoder).
  - `runners/`: Scripts to run different complexity levels of the project (Easy, Medium, Hard).
- `data/`: Directory for datasets (Audio files and CSV metadata).
- `results/`: Output directory for clustering plots and metrics.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The project is divided into three tasks increasing in complexity:

### 1. Easy Task: Audio Features VAE
Trains a simple VAE on MFCC audio features and compares clustering performance with a PCA + K-Means baseline.
```bash
python src/runners/run_easy.py
```

### 2. Medium Task: CVAE & Hybrid VAE
- **CVAE**: Uses Convolutional VAE on Mel Spectrograms.
- **Hybrid VAE**: Combines Audio (MFCC) and Lyrics (TF-IDF) features.
```bash
python src/runners/run_medium.py
```

### 3. Hard Task: Multi-Modal Beta-VAE
Implements a Beta-VAE for disentangled representation learning using Audio, Lyrics, and Genre information. Compares against Autoencoder and Spectral Clustering baselines.
```bash
python src/runners/run_hard.py
```

## Results
Metrics (Silhouette Score, Calinski-Harabasz, NMI, Purity) and visualizations are saved in the `results/` directory.
