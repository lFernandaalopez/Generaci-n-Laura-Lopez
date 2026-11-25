"""
datos.py — Carga de los conjuntos de imágenes del proyecto.

  * Rostros AT&T / ORL (AT&T Laboratories Cambridge): 40 personas x 10 fotos, escala de grises.
    Se usan para entrenar la DCGAN ("dibujar rostros humanos").
  * Fashion-MNIST (Xiao et al., 2017): 70 000 prendas 28x28 en 10 clases. Se usa para
    preentrenar un modelo de difusión con diffusers y luego hacer fine-tuning (LoRA) sobre una
    clase que el modelo nunca vio.

Autor: Jahir Ávila — Especialización en Inteligencia Artificial.
"""
from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "datos"
CLASES_FASHION = ["camiseta", "pantalón", "suéter", "vestido", "abrigo", "sandalia", "camisa", "tenis", "bolso", "botín"]
_FASHION_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"


def cargar_rostros(tam: int = 64) -> np.ndarray:
    """Rostros AT&T: recorte cuadrado centrado (92x92) -> tam x tam, valores en [-1, 1]. Forma (400, 1, tam, tam)."""
    imgs = []
    for s in range(1, 41):
        for i in range(1, 11):
            im = cv2.imread(str(DATOS / "att_faces" / f"s{s}" / f"{i}.pgm"), cv2.IMREAD_GRAYSCALE)
            im = im[10:102, :]                                   # 112x92 -> 92x92 centrado en el rostro
            im = cv2.resize(im, (tam, tam), interpolation=cv2.INTER_AREA)
            imgs.append(im)
    x = np.stack(imgs).astype(np.float32) / 127.5 - 1.0
    return x[:, None]


def _leer_idx(ruta):
    with gzip.open(ruta, "rb") as f:
        datos = f.read()
    ndim = datos[3]
    dims = [int.from_bytes(datos[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(datos, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def cargar_fashion(particion: str = "train", tam: int = 32):
    """Fashion-MNIST redimensionado a tam x tam (relleno de 2 px: 28 -> 32), valores en [-1, 1]."""
    carpeta = DATOS / "fashion"
    carpeta.mkdir(parents=True, exist_ok=True)
    pref = "train" if particion == "train" else "t10k"
    rutas = []
    for tipo in ["images-idx3-ubyte", "labels-idx1-ubyte"]:
        ruta = carpeta / f"{pref}-{tipo}.gz"
        if not ruta.exists():
            urllib.request.urlretrieve(_FASHION_URL + ruta.name, ruta)
        rutas.append(ruta)
    x, y = _leer_idx(rutas[0]), _leer_idx(rutas[1])
    pad = (tam - 28) // 2
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    return (x.astype(np.float32) / 127.5 - 1.0)[:, None], y.astype(np.int64)


def a_imagen(x):
    """Tensor/array en [-1, 1] -> uint8 [0, 255] para visualizar."""
    x = np.asarray(x)
    return ((np.clip(x, -1, 1) + 1) * 127.5).astype(np.uint8)
