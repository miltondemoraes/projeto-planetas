"""Matemática 3D: matrizes 4x4 (convenção de coluna, v' = M @ v) e quatérnios.

Todas as matrizes são numpy float64 em ordem *row-major* matemática; ao enviar
para o OpenGL usamos transpose=GL_TRUE (ver shader.py).
"""
import math
import numpy as np


# ----------------------------------------------------------------------------
# Matrizes básicas
# ----------------------------------------------------------------------------
def identity():
    return np.eye(4)


def translate(v):
    m = np.eye(4)
    m[:3, 3] = v
    return m


def scale(s):
    if np.isscalar(s):
        s = (s, s, s)
    return np.diag([s[0], s[1], s[2], 1.0])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def mat4_from_mat3(r):
    m = np.eye(4)
    m[:3, :3] = r
    return m


def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def ortho(l, r, b, t, n, f):
    m = np.eye(4)
    m[0, 0] = 2.0 / (r - l)
    m[1, 1] = 2.0 / (t - b)
    m[2, 2] = -2.0 / (f - n)
    m[0, 3] = -(r + l) / (r - l)
    m[1, 3] = -(t + b) / (t - b)
    m[2, 3] = -(f + n) / (f - n)
    return m


def look_at(eye, target, up):
    f = normalize(np.asarray(target, float) - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def normal_matrix(m4):
    """Inversa-transposta da parte 3x3 (corrige normais sob escala não uniforme)."""
    return np.linalg.inv(m4[:3, :3]).T


def normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ----------------------------------------------------------------------------
# Quatérnios [w, x, y, z]
# ----------------------------------------------------------------------------
def q_axis_angle(axis, angle):
    axis = normalize(axis)
    h = angle * 0.5
    return np.array([math.cos(h), *(axis * math.sin(h))])


def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def q_normalize(q):
    return q / np.linalg.norm(q)


def q_to_mat3(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def q_from_mat3(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return q_normalize(np.array(q))


def q_slerp(a, b, t):
    d = float(np.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    if d > 0.9995:
        return q_normalize(a + (b - a) * t)
    th = math.acos(d)
    s = math.sin(th)
    return (a * math.sin((1 - t) * th) + b * math.sin(t * th)) / s


def look_rotation(forward, up=(0.0, 1.0, 0.0)):
    """Quatérnio cuja -Z local aponta para `forward` (convenção de câmera OpenGL)."""
    f = normalize(forward)
    up = np.asarray(up, float)
    if abs(np.dot(f, normalize(up))) > 0.999:
        up = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    r = normalize(np.cross(f, up))
    u = np.cross(r, f)
    m = np.column_stack([r, u, -f])
    return q_from_mat3(m)


def smooth_factor(rate, dt):
    """Fator de suavização exponencial independente de framerate."""
    return 1.0 - math.exp(-rate * dt)
