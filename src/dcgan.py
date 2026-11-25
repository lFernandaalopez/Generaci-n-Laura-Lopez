"""
dcgan.py — Red Generativa Adversaria convolucional profunda (DCGAN; Radford et al., 2016) en PyTorch.

Generador G: z ~ N(0, I) (vector latente de 100 dimensiones) -> imagen 64x64.
    ConvTranspose 4x4 -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64, BatchNorm + ReLU, salida Tanh.
Discriminador D: imagen 64x64 -> probabilidad de que sea REAL.
    Conv 4x4 con stride 2 (sin pooling), BatchNorm + LeakyReLU(0.2), salida un logit.

Reglas de diseño DCGAN aplicadas: convoluciones con stride en lugar de pooling, BatchNorm en G y D
(excepto entrada de D y salida de G), sin capas densas ocultas, ReLU en G / LeakyReLU en D,
inicialización N(0, 0.02) y Adam con lr = 2e-4 y β1 = 0.5.

Entrenamiento adversarial (juego minimax):
    min_G max_D  E[log D(x)] + E[log(1 − D(G(z)))]
En la práctica: D minimiza la entropía cruzada binaria (reales = 1, falsas = 0) y G minimiza
BCE(D(G(z)), 1) — versión "no saturante" que da gradientes fuertes al inicio.

Autor: Jahir Ávila — Especialización en Inteligencia Artificial.
"""
from __future__ import annotations

import torch
import torch.nn as nn

DIM_Z = 100


def iniciar_pesos(m):
    """Inicialización recomendada por Radford et al. (2016)."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Generador(nn.Module):
    def __init__(self, dim_z=DIM_Z, ngf=48, canales=1):
        super().__init__()
        def bloque(c_in, c_out):
            return [nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True)]
        self.red = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),  # 4x4
            *bloque(ngf * 8, ngf * 4),   # 8x8
            *bloque(ngf * 4, ngf * 2),   # 16x16
            *bloque(ngf * 2, ngf),       # 32x32
            nn.ConvTranspose2d(ngf, canales, 4, 2, 1, bias=False), nn.Tanh())  # 64x64, valores en [-1, 1]

    def forward(self, z):
        return self.red(z.view(z.size(0), -1, 1, 1))


class Discriminador(nn.Module):
    def __init__(self, ndf=48, canales=1):
        super().__init__()
        def bloque(c_in, c_out):
            return [nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2, True)]
        self.red = nn.Sequential(
            nn.Conv2d(canales, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),  # 32x32
            *bloque(ndf, ndf * 2),       # 16x16
            *bloque(ndf * 2, ndf * 4),   # 8x8
            *bloque(ndf * 4, ndf * 8),   # 4x4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))  # 1x1: logit (la sigmoide va dentro de la pérdida)

    def forward(self, x):
        return self.red(x).view(-1)


def paso_entrenamiento(G, D, opt_G, opt_D, reales, dim_z=DIM_Z, etiqueta_real=0.9):
    """Un paso del entrenamiento adversarial. Devuelve pérdidas y salidas medias de D.

    1) Discriminador: maximizar log D(x) + log(1 − D(G(z)))  (BCE con reales = 0.9, falsas = 0)
       - 'label smoothing' unilateral (0.9) evita que D se vuelva demasiado confiado.
    2) Generador: maximizar log D(G(z))  (BCE de las falsas contra la etiqueta 1)
    """
    bce = nn.BCEWithLogitsLoss()
    B = reales.size(0)
    z = torch.randn(B, dim_z, device=reales.device)
    falsas = G(z)
    # --- 1. Discriminador ---
    opt_D.zero_grad()
    out_real = D(reales)
    out_falsa = D(falsas.detach())                 # detach: no propagar al generador
    loss_D = bce(out_real, torch.full_like(out_real, etiqueta_real)) + bce(out_falsa, torch.zeros_like(out_falsa))
    loss_D.backward()
    opt_D.step()
    # --- 2. Generador ---
    opt_G.zero_grad()
    out = D(falsas)
    loss_G = bce(out, torch.ones_like(out))
    loss_G.backward()
    opt_G.step()
    return {"loss_D": loss_D.item(), "loss_G": loss_G.item(),
            "D_x": torch.sigmoid(out_real).mean().item(), "D_G_z": torch.sigmoid(out_falsa).mean().item()}
