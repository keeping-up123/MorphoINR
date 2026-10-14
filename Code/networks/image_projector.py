import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageProjector(nn.Module):
    def __init__(self, image_dim=1024, latent_dim=64, hidden_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x):
        return self.net(x)


def symmetric_info_nce(z_st, z_img, temperature=0.1):
    """
    z_st:  [B, D], Gene AE latent
    z_img: [B, D], projected image latent
    """
    z_st = F.normalize(z_st, dim=-1)
    z_img = F.normalize(z_img, dim=-1)

    logits = z_st @ z_img.T / temperature
    labels = torch.arange(z_st.size(0), device=z_st.device)

    loss_st_to_img = F.cross_entropy(logits, labels)
    loss_img_to_st = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_st_to_img + loss_img_to_st)