
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridVAE(nn.Module):
    def __init__(self, audio_dim=30, text_dim=1000, hidden_dim=256, latent_dim=32):
        super(HybridVAE, self).__init__()
        
        # Audio Encoder
        self.audio_encoder = nn.Linear(audio_dim, hidden_dim)
        
        # Text Encoder
        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        
        # Fusion
        self.fc_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Latent Space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoders
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        
        # Audio Decoder
        self.audio_decoder = nn.Linear(hidden_dim, audio_dim)
        
        # Text Decoder
        self.text_decoder = nn.Linear(hidden_dim, text_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_audio, x_text):
        # Encode Audio
        h_audio = F.relu(self.audio_encoder(x_audio))
        
        # Encode Text
        h_text = F.relu(self.text_encoder(x_text))
        
        # Fuse
        h_fused = torch.cat([h_audio, h_text], dim=1)
        h_joint = F.relu(self.fc_fusion(h_fused))
        
        # Bottleneck
        mu = self.fc_mu(h_joint)
        logvar = self.fc_logvar(h_joint)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        h_decoded = F.relu(self.decoder_input(z))
        
        recon_audio = self.audio_decoder(h_decoded)
        recon_text = torch.sigmoid(self.text_decoder(h_decoded)) # Sigmoid for TF-IDF
        
        return recon_audio, recon_text, mu, logvar
    
    def encode(self, x_audio, x_text):
        h_audio = F.relu(self.audio_encoder(x_audio))
        h_text = F.relu(self.text_encoder(x_text))
        h_fused = torch.cat([h_audio, h_text], dim=1)
        h_joint = F.relu(self.fc_fusion(h_fused))
        mu = self.fc_mu(h_joint)
        return mu, None

def hybrid_loss_function(recon_audio, x_audio, recon_text, x_text, mu, logvar, alpha=0.5):
    # MSE for Audio
    BCE_audio = F.mse_loss(recon_audio, x_audio, reduction='sum')
    
    # MSE for Text (TF-IDF)
    BCE_text = F.mse_loss(recon_text, x_text, reduction='sum')
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return alpha * BCE_audio + (1-alpha) * BCE_text + KLD
