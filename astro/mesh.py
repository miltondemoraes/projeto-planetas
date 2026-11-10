"""Geração procedural de malhas e upload para a GPU (VAO/VBO/EBO).

Layout de vértice intercalado (11 floats = 44 bytes):
    location 0: posição  (vec3)
    location 1: normal   (vec3)
    location 2: UV       (vec2)
    location 3: tangente (vec3)  -> direção de +U, usada no normal mapping (TBN)
"""
import ctypes
import math
import numpy as np
from OpenGL.GL import *

STRIDE = 11 * 4


class Mesh:
    def __init__(self, verts, idx, mode=GL_TRIANGLES):
        verts = np.ascontiguousarray(verts, dtype=np.float32)
        idx = np.ascontiguousarray(idx, dtype=np.uint32)
        self.count = idx.size
        self.mode = mode
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
        for loc, size, off in ((0, 3, 0), (1, 3, 3), (2, 2, 6), (3, 3, 8)):
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, STRIDE, ctypes.c_void_p(off * 4))
        glBindVertexArray(0)

    def add_instance_buffer(self, data, layout, start_loc=4):
        """Anexa um VBO de atributos por instância (glVertexAttribDivisor = 1)."""
        data = np.ascontiguousarray(data, dtype=np.float32)
        glBindVertexArray(self.vao)
        buf = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, buf)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        stride = sum(layout) * 4
        off = 0
        for i, size in enumerate(layout):
            glEnableVertexAttribArray(start_loc + i)
            glVertexAttribPointer(start_loc + i, size, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(off * 4))
            glVertexAttribDivisor(start_loc + i, 1)
            off += size
        glBindVertexArray(0)
        self.instances = data.shape[0]

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawElements(self.mode, self.count, GL_UNSIGNED_INT, None)

    def draw_instanced(self, n=None):
        glBindVertexArray(self.vao)
        glDrawElementsInstanced(self.mode, self.count, GL_UNSIGNED_INT, None, n or self.instances)


class LineMesh:
    """Malha só de posições, para linhas (órbitas)."""

    def __init__(self, pts, mode=GL_LINE_LOOP):
        pts = np.ascontiguousarray(pts, dtype=np.float32)
        self.count = len(pts)
        self.mode = mode
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, pts.nbytes, pts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glBindVertexArray(0)

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(self.mode, 0, self.count)


def _pack(pos, nrm, uv, tan):
    return np.hstack([pos, nrm, uv, tan]).astype(np.float32)


def _grid_indices(rows, cols):
    """Índices de uma grade (rows+1)x(cols+1) em triângulos CCW (vistos de fora)."""
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    a = r * (cols + 1) + c
    b = a + cols + 1
    tris = np.stack([a, b, a + 1, a + 1, b, b + 1], axis=-1)
    return tris.reshape(-1).astype(np.uint32)


# ----------------------------------------------------------------------------
# Esfera UV (latitude/longitude) -> casa com texturas equiretangulares
# ----------------------------------------------------------------------------
def uv_sphere(slices=128, stacks=64):
    """Esfera unitária. u = longitude (leste +), v = latitude (norte = 1).

    Parametrização: P(φ,λ) = (cosλ·sinφ, sinλ, cosλ·cosφ), φ = 2πu − π, λ = π(v − ½)
    Tangente T = ∂P/∂u normalizada = (cosφ, 0, −sinφ); bitangente B = N × T aponta para o norte.
    """
    v = np.linspace(0.0, 1.0, stacks + 1)
    u = np.linspace(0.0, 1.0, slices + 1)
    # linhas de cima (norte) para baixo para manter winding CCW visto de fora
    vv, uu = np.meshgrid(v[::-1], u, indexing="ij")
    lat = math.pi * (vv - 0.5)
    lon = 2.0 * math.pi * uu - math.pi
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    pos = np.stack([x, y, z], -1).reshape(-1, 3)
    tan = np.stack([np.cos(lon), np.zeros_like(lon), -np.sin(lon)], -1).reshape(-1, 3)
    uv = np.stack([uu, vv], -1).reshape(-1, 2)
    return _pack(pos, pos, uv, tan), _grid_indices(stacks, slices)


def capsule(radius, length, slices=32, stacks=24):
    """Cápsula pendurada ao longo de −Y a partir da origem (articulação no topo)."""
    half = stacks // 2
    lats = np.concatenate([np.linspace(math.pi / 2, 0, half + 1), np.linspace(0, -math.pi / 2, half + 1)])
    offs = np.concatenate([np.zeros(half + 1), np.full(half + 1, -length)])
    u = np.linspace(0, 1, slices + 1)
    L, U = np.meshgrid(lats, u, indexing="ij")
    O = np.repeat(offs[:, None], slices + 1, axis=1)
    lon = 2 * math.pi * U - math.pi
    n = np.stack([np.cos(L) * np.sin(lon), np.sin(L), np.cos(L) * np.cos(lon)], -1)
    p = n * radius
    p[..., 1] += O
    tan = np.stack([np.cos(lon), np.zeros_like(lon), -np.sin(lon)], -1)
    total = length + 2 * radius
    vcoord = (p[..., 1] + length + radius) / total
    uv = np.stack([U, vcoord * (total / (2 * math.pi * radius))], -1)
    rows = lats.size - 1
    return _pack(p.reshape(-1, 3), n.reshape(-1, 3), uv.reshape(-1, 2), tan.reshape(-1, 3)), _grid_indices(rows, slices)


def _spow(x, e):
    return np.sign(x) * np.abs(x) ** e


def superellipsoid(a, b, c, e1=0.3, e2=0.3, slices=40, stacks=24):
    """Superelipsoide (caixa arredondada quando e1,e2 < 1). Normais analíticas."""
    v = np.linspace(math.pi / 2, -math.pi / 2, stacks + 1)
    u = np.linspace(-math.pi, math.pi, slices + 1)
    V, U = np.meshgrid(v, u, indexing="ij")
    cv, sv, cu, su = np.cos(V), np.sin(V), np.cos(U), np.sin(U)
    # parametrização com z = cos e x = sin (mesma orientação da esfera UV)
    x = a * _spow(cv, e1) * _spow(su, e2)
    y = b * _spow(sv, e1)
    z = c * _spow(cv, e1) * _spow(cu, e2)
    nx = _spow(cv, 2 - e1) * _spow(su, 2 - e2) / a
    ny = _spow(sv, 2 - e1) / b
    nz = _spow(cv, 2 - e1) * _spow(cu, 2 - e2) / c
    n = np.stack([nx, ny, nz], -1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    pole = np.abs(cv) < 1e-6
    n[pole] = np.stack([np.zeros(pole.sum()), np.sign(sv[pole]), np.zeros(pole.sum())], -1)
    tan = np.stack([cu, np.zeros_like(cu), -su], -1)
    uv = np.stack([(U + math.pi) / (2 * math.pi), (V + math.pi / 2) / math.pi], -1)
    p = np.stack([x, y, z], -1)
    return _pack(p.reshape(-1, 3), n.reshape(-1, 3), uv.reshape(-1, 2), tan.reshape(-1, 3)), _grid_indices(stacks, slices)


def annulus(inner, outer, segments=256):
    """Anel (disco furado) no plano XZ, normal +Y; u = raio normalizado, v = ângulo."""
    ang = np.linspace(0, 2 * math.pi, segments + 1)
    rad = np.array([outer, inner])
    R, A = np.meshgrid(rad, ang, indexing="ij")
    p = np.stack([R * np.cos(A), np.zeros_like(R), -R * np.sin(A)], -1)
    n = np.zeros_like(p)
    n[..., 1] = 1
    uv = np.stack([(R - inner) / (outer - inner), A / (2 * math.pi)], -1)
    tan = np.stack([np.cos(A), np.zeros_like(A), -np.sin(A)], -1)
    return _pack(p.reshape(-1, 3), n.reshape(-1, 3), uv.reshape(-1, 2), tan.reshape(-1, 3)), _grid_indices(1, segments)


def icosphere(subdiv=3, displace=None):
    """Icosfera (distribuição de vértices uniforme). `displace(dirs)->escala radial` deforma a malha."""
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
             (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
             (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(v, float) / np.linalg.norm(v) for v in verts]
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(i, j):
            k = (min(i, j), max(i, j))
            if k not in cache:
                m = verts[i] + verts[j]
                verts.append(m / np.linalg.norm(m))
                cache[k] = len(verts) - 1
            return cache[k]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    dirs = np.array(verts)
    idx = np.array(faces, dtype=np.uint32)
    pos = dirs * (displace(dirs)[:, None] if displace else 1.0)
    # normais suaves = média das normais das faces adjacentes
    fn = np.cross(pos[idx[:, 1]] - pos[idx[:, 0]], pos[idx[:, 2]] - pos[idx[:, 0]])
    n = np.zeros_like(pos)
    for k in range(3):
        np.add.at(n, idx[:, k], fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    uv = np.stack([np.arctan2(dirs[:, 0], dirs[:, 2]) / (2 * math.pi) + 0.5, np.arcsin(np.clip(dirs[:, 1], -1, 1)) / math.pi + 0.5], 1)
    tan = np.cross([0, 1, 0], n)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-6)
    return _pack(pos, n, uv, tan), idx.reshape(-1)


def circle_points(radius, n=512):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([radius * np.cos(a), np.zeros(n), -radius * np.sin(a)], 1)
