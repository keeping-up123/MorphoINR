import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: list, dim_latent: int, act_fn=nn.ReLU):
        super().__init__()
        self.net = self._build_layers(dim_in, dim_hidden, dim_latent)

    def _build_layers(self, dim_in, dim_hidden, dim_out):
        if len(dim_hidden) == 0:
            return nn.Sequential(nn.Linear(dim_in, dim_out))

        return nn.Sequential(
            nn.Linear(dim_in, dim_hidden[0]),
            nn.ReLU(),
            self._build_layers(dim_hidden[0], dim_hidden[1:], dim_out),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: list, dim_out: int, act_fn=nn.ReLU):
        super().__init__()
        self.net = self._build_layers(dim_in, dim_hidden, dim_out)

    def _build_layers(self, dim_in, dim_hidden, dim_out):
        if len(dim_hidden) == 0:
            return nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.ReLU(),  # expression after log1p is non-negative
            )

        return nn.Sequential(
            nn.Linear(dim_in, dim_hidden[0]),
            nn.ReLU(),
            self._build_layers(dim_hidden[0], dim_hidden[1:], dim_out),
        )

    def forward(self, x):
        return self.net(x)


class AE(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_latent,
        in_logarithm=False,
        dice_lambda=0.01,
        dice_eps=1.0,
        act_fn=nn.ReLU,
    ):
        super().__init__()

        self.encoder = Encoder(dim_in, dim_hidden, dim_latent, act_fn)
        self.decoder = Decoder(dim_latent, dim_hidden[::-1], dim_in, act_fn)

        self.in_logarithm = in_logarithm
        self.dice_lambda = dice_lambda
        self.dice_eps = dice_eps

    def forward(self, x):
        if self.in_logarithm:
            x = torch.log1p(x)

        z = self.encoder(x)
        y_hat = self.decoder(z)

        return y_hat, z

    def dice_loss(self, pred, gt):
        """
        Dice loss for zero/non-zero expression pattern.

        pred: reconstructed expression, non-negative
        gt:   target expression, non-negative after log1p
        """
        gt_binary = (gt > 0).float()

        # Map non-negative prediction to soft non-zero probability.
        # pred = 0 -> 0, pred large -> close to 1
        pred_prob = 1.0 - torch.exp(-torch.clamp(pred, min=0.0))

        intersection = (pred_prob * gt_binary).sum()
        denominator = pred_prob.sum() + gt_binary.sum()

        dice = (2.0 * intersection + self.dice_eps) / (
            denominator + self.dice_eps
        )

        return 1.0 - dice

    def recon_loss(self, pred, gt):
        mse = F.mse_loss(pred, gt)
        dice = self.dice_loss(pred, gt)
        loss = mse + self.dice_lambda * dice

        return loss, {
            "mse_loss": mse,
            "dice_loss": dice,
            "recon_loss": loss,
        }

    def forward_loss(self, x):
        y_hat, z = self.forward(x)

        loss, loss_dict = self.recon_loss(y_hat, x)

        return loss, y_hat, z, loss_dict