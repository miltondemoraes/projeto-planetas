"""Criação de texturas OpenGL (mipmaps, filtragem anisotrópica, sRGB)."""
import numpy as np
from PIL import Image
from OpenGL.GL import *

GL_TEXTURE_MAX_ANISOTROPY_EXT = 0x84FE
GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT = 0x84FF
_max_aniso = None


def _aniso():
    global _max_aniso
    if _max_aniso is None:
        try:
            _max_aniso = float(glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT))
        except Exception:
            _max_aniso = 0.0
    return _max_aniso


def load_image_array(path):
    """Carrega imagem como array uint8 HxWxC, já invertida (linha 0 = v=0 = sul)."""
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    return np.asarray(img.transpose(Image.FLIP_TOP_BOTTOM))


def texture_from_array(arr, srgb=True, mipmap=True, wrap_s=GL_REPEAT, wrap_t=GL_CLAMP_TO_EDGE):
    """Envia um array uint8 (H,W[,C]) para a GPU.

    srgb=True para texturas de COR (albedo): o hardware converte sRGB->linear
    na amostragem, deixando a iluminação ser calculada em espaço linear.
    Mapas de dados (normal, especular, densidade de nuvem) usam srgb=False.
    """
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w, c = arr.shape
    fmt = {1: GL_RED, 2: GL_RG, 3: GL_RGB, 4: GL_RGBA}[c]
    if srgb and c in (3, 4):
        internal = GL_SRGB8 if c == 3 else GL_SRGB8_ALPHA8
    else:
        internal = {1: GL_R8, 2: GL_RG8, 3: GL_RGB8, 4: GL_RGBA8}[c]
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, GL_UNSIGNED_BYTE, arr)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    if mipmap:
        glGenerateMipmap(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        a = _aniso()
        if a > 0:
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, min(16.0, a))
    else:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    if c == 1:
        # textura de 1 canal lida como (r, r, r, 1) nos shaders
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, [GL_RED, GL_RED, GL_RED, GL_ONE])
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


def solid_texture(rgba, srgb=False):
    return texture_from_array(np.array(rgba, np.uint8).reshape(1, 1, -1), srgb=srgb, mipmap=False)


def bind(unit, tex, target=GL_TEXTURE_2D):
    glActiveTexture(GL_TEXTURE0 + unit)
    glBindTexture(target, tex)
