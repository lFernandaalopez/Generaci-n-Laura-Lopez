# Generación — DCGAN y modelos de difusión con diffusers

**Especialización en Inteligencia Artificial · Laura Lopez**

| # | Notebook | Contenido | Estado |
|---|---|---|---|
| 01 | `01_DCGAN_rostros.ipynb` | DCGAN en PyTorch que genera rostros 64×64 (AT&T/ORL): entrenamiento adversarial, evolución, interpolación latente, memorización y diversidad | Ejecutado (CPU) |
| 02 | `02_Difusion_diffusers_finetuning.ipynb` | diffusers: proceso de difusión, U-Net + DDPMScheduler, preentrenamiento en Fashion-MNIST sin bolsos, fine-tuning **LoRA** (peft) con 100 bolsos vs. ajuste completo | Ejecutado (CPU) |
| 03 | `03_StableDiffusion_LoRA_Colab.ipynb` | Fine-tuning LoRA de **Stable Diffusion 1.5** con diffusers y peft (VAE, CLIP, U-Net condicionada) | Listo para Colab con GPU T4 |

**Ejecutar en Colab:** abra el notebook desde GitHub (Archivo → Abrir cuaderno → GitHub) y, en la primera celda, reemplace
`REPO_URL` por la URL de este repositorio.

## Resultados principales

| Experimento | Resultado |
|---|---|
| DCGAN (250 épocas, 14 min CPU) | Rostros reconocibles; sin memorización (distancia al real más cercano 19,1 vs. 11,5 entre reales) ni colapso de modo |
| Difusión base (2 000 pasos, 29 min CPU) | Genera prendas de las 9 clases vistas; no genera bolsos |
| LoRA r = 8 (7,6 % de los parámetros, 100 imágenes) | Muestras clasificadas como bolso: 16,8 % → 63,3 % (igual que el ajuste completo) |

## Estructura

```
├── src/          dcgan.py · datos.py
├── notebooks/    01 · 02 · 03
├── datos/        att_faces/ · fashion/
├── figuras/ · resultados/ · modelos/ (generador DCGAN, U-Net base, pesos LoRA)
└── requirements.txt
```
