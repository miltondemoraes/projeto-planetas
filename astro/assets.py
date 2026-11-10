"""Gerenciador de texturas: usa texturas REAIS (se baixadas) ou procedurais (cache em PNG).

Texturas reais (Solar System Scope, CC BY 4.0) ficam em assets/textures/ e podem
ser obtidas com `python tools/baixar_texturas.py`. Sem elas, tudo é gerado
proceduralmente em numpy na primeira execução e salvo em assets/textures/generated/.
"""
import os
import time
import numpy as np
from PIL import Image

from . import procedural

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(ROOT, "assets", "textures")
GEN_DIR = os.path.join(REAL_DIR, "generated")
CACHE_VERSION = 3

# (corpo, papel) -> nomes de arquivos reais aceitos (Solar System Scope)
REAL_FILES = {
    ("sun", "albedo"): ["8k_sun.jpg", "2k_sun.jpg"],
    ("mercury", "albedo"): ["8k_mercury.jpg", "2k_mercury.jpg"],
    ("venus", "albedo"): ["4k_venus_atmosphere.jpg", "2k_venus_atmosphere.jpg"],
    ("earth", "albedo"): ["8k_earth_daymap.jpg", "2k_earth_daymap.jpg"],
    ("earth", "night"): ["8k_earth_nightmap.jpg", "2k_earth_nightmap.jpg"],
    ("earth", "clouds"): ["8k_earth_clouds.jpg", "2k_earth_clouds.jpg"],
    ("earth", "normal"): ["8k_earth_normal_map.tif", "2k_earth_normal_map.tif"],
    ("earth", "spec"): ["8k_earth_specular_map.tif", "2k_earth_specular_map.tif"],
    ("moon", "albedo"): ["8k_moon.jpg", "2k_moon.jpg"],
    ("mars", "albedo"): ["8k_mars.jpg", "2k_mars.jpg"],
    ("jupiter", "albedo"): ["8k_jupiter.jpg", "2k_jupiter.jpg"],
    ("saturn", "albedo"): ["8k_saturn.jpg", "2k_saturn.jpg"],
    ("rings", "rings"): ["8k_saturn_ring_alpha.png", "2k_saturn_ring_alpha.png"],
    ("uranus", "albedo"): ["2k_uranus.jpg"],
    ("neptune", "albedo"): ["2k_neptune.jpg"],
    ("sky", "sky"): ["8k_stars_milky_way.jpg", "2k_stars_milky_way.jpg"],
}


def _real_path(body, role):
    for fn in REAL_FILES.get((body, role), []):
        p = os.path.join(REAL_DIR, fn)
        if os.path.exists(p):
            return p
    return None


class TextureStore:
    """Fornece arrays uint8 (orientação de imagem: linha 0 = topo/norte)."""

    def __init__(self, use_real=True, quality=1.0):
        self.use_real = use_real
        self.quality = quality
        self.sources = {}
        os.makedirs(GEN_DIR, exist_ok=True)

    def _gen_path(self, body, role, W):
        return os.path.join(GEN_DIR, f"v{CACHE_VERSION}_{body}_{role}_{W}.png")

    def get(self, body, role):
        """Retorna array (H,W,C) uint8 ou None se o corpo não tem esse mapa."""
        if self.use_real:
            p = _real_path(body, role)
            if p:
                self.sources[(body, role)] = "real"
                img = Image.open(p)
                if role in ("rings",):
                    img = img.convert("RGBA")
                elif role in ("clouds", "spec"):
                    img = img.convert("L")
                else:
                    img = img.convert("RGB")
                arr = np.asarray(img)
                if role == "rings" and arr[..., 3].max() == 255 and arr[..., 3].min() == 255:
                    arr = arr.copy()
                    arr[..., 3] = arr[..., :3].mean(-1).astype(np.uint8)
                return arr
            # Com textura real de albedo mas sem mapa auxiliar, não mistura com procedural.
            if role != "albedo" and _real_path(body, "albedo") and body != "earth":
                return None
        if body not in procedural.GENERATORS:
            return None
        fn, W = procedural.GENERATORS[body]
        W = max(256, int(W * self.quality) // 2 * 2)
        path = self._gen_path(body, role, W)
        if not os.path.exists(path):
            # gera todos os mapas do corpo de uma vez e salva no cache
            existing = [f for f in os.listdir(GEN_DIR) if f.startswith(f"v{CACHE_VERSION}_{body}_") and f.endswith(f"_{W}.png")]
            if existing:
                return None
            t0 = time.time()
            print(f"  gerando texturas procedurais de '{body}' ({W}px)...", end="", flush=True)
            maps = fn(W)
            for r, arr in maps.items():
                Image.fromarray(arr).save(self._gen_path(body, r, W))
            print(f" {time.time() - t0:.1f}s")
            if not os.path.exists(path):
                return None
        self.sources[(body, role)] = "procedural"
        return np.asarray(Image.open(path))


def flip(arr):
    """Imagem (linha 0 = topo) -> convenção OpenGL (linha 0 = v=0 = base)."""
    return np.ascontiguousarray(arr[::-1])


def height_to_normal(arr_rgb, strength=1.0):
    """Para texturas reais sem normal map: usa a luminância como pseudo-altura."""
    lum = arr_rgb[..., :3].astype(np.float32).mean(-1) / 255.0
    return procedural.normal_from_height(lum * 0.004, strength)
