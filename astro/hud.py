"""HUD: texto 2D desenhado com OpenGL usando um atlas de glifos (rasterizado com Pillow)."""
import ctypes
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from OpenGL.GL import *

from .texture import texture_from_array

CHARS = [chr(c) for c in range(32, 127)] + [chr(c) for c in range(160, 256)] + ["—", "…", "“", "”", "█"]


class TextRenderer:
    def __init__(self, shader, size=16):
        self.shader = shader
        font = None
        for f in ("consola.ttf", "DejaVuSansMono.ttf", "cour.ttf", "LiberationMono-Regular.ttf"):
            for d in ("", "C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/"):
                try:
                    font = ImageFont.truetype(os.path.join(d, f) if d else f, size)
                    break
                except OSError:
                    pass
            if font:
                break
        if font is None:
            font = ImageFont.load_default()
        bbox = font.getbbox("M")
        self.cw = max(int(font.getlength("M")), 6)
        self.ch = int(size * 1.3)
        cols = 16
        rows = (len(CHARS) + cols - 1) // cols
        img = Image.new("L", (cols * self.cw, rows * self.ch), 0)
        dr = ImageDraw.Draw(img)
        self.uv = {}
        W, H = img.size
        for i, c in enumerate(CHARS):
            x, y = (i % cols) * self.cw, (i // cols) * self.ch
            if c == "█":
                dr.rectangle([x, y, x + self.cw - 1, y + self.ch - 1], fill=255)
            else:
                dr.text((x, y + 1), c, font=font, fill=255)
            self.uv[c] = (x / W, y / H, (x + self.cw) / W, (y + self.ch) / H)
        # textura sem inverter: v cresce para baixo, o shader usa coordenadas de tela com y para baixo
        self.tex = texture_from_array(np.asarray(img), srgb=False, mipmap=False)
        self.verts = []
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        for loc, size_, off in ((0, 2, 0), (1, 2, 2), (2, 4, 4)):
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, size_, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(off * 4))
        glBindVertexArray(0)

    def rect(self, x, y, w, h, color):
        u0, v0, u1, v1 = self.uv["█"]
        uc, vc = (u0 + u1) / 2, (v0 + v1) / 2
        self._quad(x, y, x + w, y + h, uc, vc, uc, vc, color)

    def _quad(self, x0, y0, x1, y1, u0, v0, u1, v1, c):
        self.verts += [x0, y0, u0, v0, *c, x1, y0, u1, v0, *c, x1, y1, u1, v1, *c,
                       x0, y0, u0, v0, *c, x1, y1, u1, v1, *c, x0, y1, u0, v1, *c]

    def text(self, s, x, y, color=(1, 1, 1, 1), shadow=True):
        if shadow:
            self.text(s, x + 1, y + 1, (0, 0, 0, color[3] * 0.8), shadow=False)
        cx = x
        for ch in s:
            if ch == "\n":
                cx = x
                y += self.ch
                continue
            uv = self.uv.get(ch, self.uv["?"])
            if ch != " ":
                self._quad(cx, y, cx + self.cw, y + self.ch, *uv, color)
            cx += self.cw
        return y + self.ch

    def width(self, s):
        return max(len(l) for l in s.split("\n")) * self.cw

    def flush(self, w, h):
        if not self.verts:
            return
        data = np.array(self.verts, np.float32)
        self.verts = []
        self.shader.use()
        self.shader.v2("uScreen", (w, h))
        self.shader.i("uAtlas", 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STREAM_DRAW)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glDrawArrays(GL_TRIANGLES, 0, len(data) // 8)
        glEnable(GL_CULL_FACE)
        glDisable(GL_BLEND)
