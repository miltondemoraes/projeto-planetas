"""Geração procedural de texturas (100% numpy, sem arquivos externos).

Todas as texturas de planetas são EQUIRETANGULARES (largura = 2 × altura):
cada pixel (linha, coluna) corresponde a (latitude, longitude). O ruído é
amostrado na direção 3D do ponto da esfera — e não em (u,v) — para não haver
costura na longitude 180° nem distorção/pinçamento nos polos.
"""
import math
import numpy as np

F32 = np.float32


# ----------------------------------------------------------------------------
# Ruído (value noise 3D com interpolação quíntica + fBm)
# ----------------------------------------------------------------------------
def _hash(x, y, z, seed):
    h = (x * np.uint32(0x8DA6B343)) ^ (y * np.uint32(0xD8163841)) ^ (z * np.uint32(0xCB1AB31F)) ^ np.uint32(seed)
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x2C1B3C6D)
    h ^= h >> np.uint32(12)
    h *= np.uint32(0x297A2D39)
    h ^= h >> np.uint32(15)
    return h.astype(F32) * F32(1.0 / 4294967295.0)


def vnoise(p, seed=0):
    """Value noise 3D em [-1, 1]. p: (..., 3)."""
    seed = (seed * 0x9E3779B9 + 0x1234567) & 0xFFFFFFFF
    fl = np.floor(p)
    f = (p - fl).astype(F32)
    i = np.ascontiguousarray(fl.astype(np.int32)).view(np.uint32)
    x0, y0, z0 = i[..., 0], i[..., 1], i[..., 2]
    one = np.uint32(1)
    x1, y1, z1 = x0 + one, y0 + one, z0 + one
    u = f * f * f * (f * (f * F32(6) - F32(15)) + F32(10))
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]

    def lerp(a, b, t):
        return a + (b - a) * t

    c00 = lerp(_hash(x0, y0, z0, seed), _hash(x1, y0, z0, seed), ux)
    c10 = lerp(_hash(x0, y1, z0, seed), _hash(x1, y1, z0, seed), ux)
    c01 = lerp(_hash(x0, y0, z1, seed), _hash(x1, y0, z1, seed), ux)
    c11 = lerp(_hash(x0, y1, z1, seed), _hash(x1, y1, z1, seed), ux)
    return lerp(lerp(c00, c10, uy), lerp(c01, c11, uy), uz) * F32(2) - F32(1)


def fbm(p, octaves=6, lac=2.03, gain=0.5, seed=0):
    """Fractal Brownian Motion: soma de oitavas de ruído (freq ×lac, amp ×gain)."""
    p = np.asarray(p, F32)
    total = np.zeros(p.shape[:-1], F32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        total += F32(amp) * vnoise(p, seed + o * 1013)
        norm += amp
        amp *= gain
        p = p * F32(lac) + F32(17.17 * (o + 1))
    return total / F32(norm)


def ridged(p, octaves=5, seed=0):
    return 1.0 - np.abs(fbm(p, octaves, seed=seed))


def normalize01(x):
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1).astype(F32)


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return (t * t * (3 - 2 * t)).astype(F32)


def mix(a, b, t):
    a = np.asarray(a, F32)
    b = np.asarray(b, F32)
    t = np.asarray(t, F32)[..., None]
    return a + (b - a) * t


def to_u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------
# Geometria da textura equiretangular
# ----------------------------------------------------------------------------
def sphere_dirs(W):
    H = W // 2
    u = (np.arange(W) + 0.5) / W
    v = 1.0 - (np.arange(H) + 0.5) / H          # linha 0 = norte
    lon = 2 * math.pi * u - math.pi
    lat = math.pi * (v - 0.5)
    cl = np.cos(lat)[:, None]
    d = np.stack([cl * np.sin(lon)[None, :],
                  np.repeat(np.sin(lat)[:, None], W, 1),
                  cl * np.cos(lon)[None, :]], -1).astype(F32)
    return d, np.degrees(lat)[:, None].astype(F32), np.degrees(lon)[None, :].astype(F32)


def warp_field(d, freq, oct, seed):
    return np.stack([fbm(d * freq + F32(s * 3.7), oct, seed=seed + s) for s in range(3)], -1)


def normal_from_height(h, strength):
    """Mapa de normais (espaço tangente) a partir de um mapa de altura equiretangular.

    Usa diferenças centrais: ∂h/∂leste (com correção 1/cos(lat), pois os
    meridianos convergem) e ∂h/∂norte. n = normalize(−s·∂h/∂x, −s·∂h/∂y, 1).
    """
    H, W = h.shape
    lat = math.pi * (0.5 - (np.arange(H) + 0.5) / H)
    coslat = np.maximum(np.cos(lat), 0.05)[:, None]
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * (2 * math.pi / W) * coslat)
    hp = np.vstack([h[:1], h, h[-1:]])
    dy = (hp[:-2] - hp[2:]) / (2 * (math.pi / H))   # linha anterior = mais ao norte
    n = np.stack([-dx * strength, -dy * strength, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return to_u8(n * 0.5 + 0.5)


def add_craters(d, n, rmin, rmax, seed, depth=1.0, rays=0):
    """Crateras de impacto: tigela parabólica + borda elevada (+ raios de ejeção)."""
    rng = np.random.default_rng(seed)
    flat = d.reshape(-1, 3)
    h = np.zeros(flat.shape[0], F32)
    bright = np.zeros(flat.shape[0], F32)
    c = rng.normal(size=(n, 3))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    radii = rmin * (rmax / rmin) ** (rng.random(n) ** 4.0)        # lei de potência: muitas pequenas
    for k in range(n):
        r = radii[k]
        dots = flat @ c[k].astype(F32)
        reach = 2.2 if k >= rays else 9.0
        m = np.nonzero(dots > math.cos(min(r * reach, math.pi)))[0]
        if m.size == 0:
            continue
        x = np.arccos(np.clip(dots[m], -1, 1)) / r
        bowl = np.where(x < 1.0, (x * x - 1.0), 0.0)
        rim = 0.35 * np.exp(-((x - 1.0) / 0.22) ** 2)
        h[m] += (0.4 * bowl + 0.12 * rim / 0.35) * r * depth          # profundidade/diâmetro ~ 0,2
        if k < rays:   # crateras jovens: raios brilhantes de material ejetado
            ang = np.arctan2(flat[m] @ np.cross(c[k], [0, 1, 0]).astype(F32), flat[m] @ np.cross(c[k], np.cross(c[k], [0, 1, 0])).astype(F32))
            streak = np.clip(np.sin(ang * 23 + k) * np.sin(ang * 37 + 2 * k), 0, 1) ** 3
            bright[m] += np.where(x > 1, streak * np.exp(-(x - 1) / 2.5) * 0.6, 0.25)
    return h.reshape(d.shape[:-1]), bright.reshape(d.shape[:-1])


# ----------------------------------------------------------------------------
# Geradores por corpo celeste
# ----------------------------------------------------------------------------
def gen_earth(W):
    d, lat, lon = sphere_dirs(W)
    alat = np.abs(lat)
    w = warp_field(d, 1.4, 4, 11)
    h = fbm(d * 1.8 + w * 1.1, 8, seed=1)
    sea = np.quantile(h, 0.70)                        # ~30% de terra firme
    land = h > sea
    e = np.clip((h - sea) / (h.max() - sea), 0, 1) ** 0.75
    depth = np.clip((sea - h) / (sea - h.min()), 0, 1)
    n2 = fbm(d * 6, 5, seed=3)
    moist = normalize01(fbm(d * 2.5 + w * 0.5, 5, seed=4))
    moist = moist - 0.45 * np.exp(-((alat - 24) / 9) ** 2) + 0.25 * np.exp(-(alat / 10) ** 2)
    temp = 1 - alat / 78 - e * 0.6 + n2 * 0.25

    col = mix((0.80, 0.68, 0.47), (0.62, 0.56, 0.33), sstep(0.20, 0.40, moist))     # deserto -> savana
    col = mix(col, (0.33, 0.42, 0.20), sstep(0.38, 0.55, moist))                   # campos
    col = mix(col, (0.13, 0.27, 0.10), sstep(0.55, 0.70, moist))                   # floresta
    col = mix(col, (0.08, 0.21, 0.07), sstep(0.65, 0.85, moist) * sstep(25, 5, alat))  # floresta tropical
    col = mix(col, (0.42, 0.40, 0.34), sstep(0.45, 0.25, temp))                    # tundra
    col = mix(col, (0.40, 0.35, 0.29), sstep(0.45, 0.80, e))                       # montanhas
    col = mix(col, (0.95, 0.96, 0.98), np.maximum(sstep(0.16, 0.06, temp), sstep(0.82, 0.95, e)))  # neve
    det = fbm(d * 40, 4, seed=5)
    col *= (0.86 + 0.28 * (det * 0.5 + 0.5))[..., None]
    ocean = mix((0.010, 0.045, 0.13), (0.04, 0.17, 0.28), sstep(0.25, 0.0, depth))
    ocean = mix(ocean, (0.07, 0.27, 0.33), sstep(0.04, 0.0, depth))
    alb = np.where(land[..., None], col, ocean)
    ice = sstep(0, 5, alat - (71 + n2 * 9))
    alb = mix(alb, (0.93, 0.95, 0.98), ice)

    spec = np.where(land, 0.03, 1.0) * (1 - ice * 0.85)
    height = np.where(land, e, 0.0) + ice * 0.05
    normal = normal_from_height(height.astype(F32), 0.9)

    city_area = sstep(0.05, 0.4, fbm(d * 9, 4, seed=8)) * sstep(0.15, 0.45, moist) * sstep(0.15, 0.4, temp)
    city_area *= land * (1 - ice) * (0.5 + 1.0 * sstep(0.12, 0.0, e))     # costas mais povoadas
    speck = sstep(0.25, 0.6, fbm(d * 380, 2, seed=9)) * sstep(0.0, 0.5, fbm(d * 60, 3, seed=7))
    lights = np.clip(city_area * speck * 2.5, 0, 1) ** 1.3
    night = mix(np.zeros(3, F32), (1.0, 0.72, 0.38), lights)

    w2 = warp_field(d, 1.8, 4, 31)
    c = fbm(d * 2.0 + w2 * 0.9, 8, seed=21)
    c = (c - c.mean()) / c.std()
    band = 0.45 * np.exp(-(alat / 8) ** 2) - 0.55 * np.exp(-((alat - 24) / 9) ** 2) + 0.35 * np.exp(-((alat - 55) / 12) ** 2)
    thick = sstep(0.1, 1.6, c + band)
    wisp = fbm(d * np.array([5, 9, 5], F32) + w2 * 1.5, 6, seed=23)
    thin = sstep(0.0, 0.5, wisp) * sstep(-0.8, 0.3, c + band) * 0.45
    clouds = np.clip(np.maximum(thick, thin) * (0.85 + 0.3 * (fbm(d * 25, 3, seed=22) * 0.5 + 0.5)), 0, 1)
    return {"albedo": to_u8(alb), "normal": normal, "spec": to_u8(spec), "night": to_u8(night), "clouds": to_u8(clouds)}


def gen_moon(W):
    d, lat, lon = sphere_dirs(W)
    base = fbm(d * 3, 7, seed=40)
    w = warp_field(d, 2.0, 4, 44)
    maria = sstep(-0.02, 0.22, fbm(d * 1.3 + w * 0.7, 7, seed=41) + 0.22 * d[..., 2] - 0.06)
    cr, br = add_craters(d, 1400, 0.002, 0.07, seed=42, depth=1.0, rays=6)
    alb = mix((0.60, 0.58, 0.55), (0.29, 0.28, 0.27), maria * 0.85)
    alb *= (0.82 + 0.36 * (base * 0.5 + 0.5))[..., None]
    alb *= (1.0 + 0.18 * np.clip(cr / 0.01, -1, 1))[..., None]        # bordas claras, fundos escuros
    alb = mix(alb, (0.82, 0.80, 0.77), np.clip(br, 0, 1) * 0.7)
    height = cr * 6 + base * 0.02 - maria * 0.01
    return {"albedo": to_u8(alb), "normal": normal_from_height(height, 1.6)}


def gen_mercury(W):
    d, lat, lon = sphere_dirs(W)
    base = fbm(d * 3, 6, seed=50)
    cr, br = add_craters(d, 1100, 0.003, 0.12, seed=51, depth=1.0, rays=5)
    alb = mix((0.48, 0.45, 0.42), (0.62, 0.58, 0.54), base * 0.5 + 0.5)
    alb *= (1.0 + 0.15 * np.clip(cr / 0.01, -1, 1))[..., None]
    alb = mix(alb, (0.75, 0.73, 0.70), np.clip(br, 0, 1) * 0.6)
    return {"albedo": to_u8(alb), "normal": normal_from_height(cr * 6 + base * 0.02, 1.6)}


def gen_mars(W):
    d, lat, lon = sphere_dirs(W)
    alat = np.abs(lat)
    w = warp_field(d, 1.5, 4, 61)
    dark = sstep(-0.05, 0.25, fbm(d * 1.6 + w * 0.8, 6, seed=60))
    base = fbm(d * 8, 5, seed=62)
    cr, _ = add_craters(d, 300, 0.002, 0.05, seed=63, depth=0.6)
    canyon = np.exp(-((lat + 8 + 2 * fbm(d * 4, 3, seed=64)) / 1.4) ** 2) * sstep(-115, -100, lon) * sstep(-40, -55, lon)
    alb = mix((0.74, 0.43, 0.25), (0.42, 0.25, 0.16), dark * 0.85)
    alb *= (0.88 + 0.24 * (base * 0.5 + 0.5))[..., None]
    alb = mix(alb, (0.35, 0.20, 0.13), canyon * 0.7)
    cap = np.where(lat > 0, sstep(0, 3, alat - (76 + base * 6)), sstep(0, 3, alat - (80 + base * 5)))
    alb = mix(alb, (0.95, 0.93, 0.90), cap)
    height = cr * 6 + fbm(d * 2, 6, seed=65) * 0.06 - canyon * 0.08 + cap * 0.01
    return {"albedo": to_u8(alb), "normal": normal_from_height(height, 1.3)}


def gen_venus(W):
    d, lat, lon = sphere_dirs(W)
    q = d * np.array([2.0, 7.0, 2.0], F32)
    latw = lat + 10 * fbm(q, 5, seed=70) + 25 * np.abs(np.sin(np.radians(lon) * 0.5)) * np.cos(np.radians(lat))
    v = 0.5 + 0.5 * np.sin(np.radians(latw) * 6 + 2 * fbm(d * 3, 4, seed=71))
    alb = mix((0.93, 0.86, 0.68), (0.78, 0.65, 0.45), v * 0.65)
    alb *= (0.92 + 0.16 * (fbm(d * np.array([6, 18, 6], F32), 4, seed=72) * 0.5 + 0.5))[..., None]
    return {"albedo": to_u8(alb)}


def _bands(lat_w, keys, cols):
    keys = np.asarray(keys, F32)
    cols = np.asarray(cols, F32)
    order = np.argsort(keys)
    keys, cols = keys[order], cols[order]
    return np.stack([np.interp(lat_w, keys, cols[:, k]) for k in range(3)], -1).astype(F32)


def gen_jupiter(W):
    d, lat, lon = sphere_dirs(W)
    wj = warp_field(d * np.array([1, 4, 1], F32), 3.0, 4, 83)
    turb = fbm(d * np.array([3, 16, 3], F32) + wj * 0.8, 6, seed=80)
    fine = fbm(d * np.array([10, 45, 10], F32) + wj * 1.5, 4, seed=81)
    latw = lat + 4.5 * turb + 1.2 * fine
    keys = [90, 55, 48, 40, 31, 25, 18, 8, 6, -6, -8, -19, -21, -27, -30, -36, -45, -55, -90]
    cols = [(0.52, 0.49, 0.45), (0.62, 0.57, 0.50), (0.70, 0.60, 0.50), (0.82, 0.77, 0.68),
            (0.62, 0.47, 0.35), (0.88, 0.84, 0.76), (0.86, 0.80, 0.70), (0.60, 0.40, 0.28),
            (0.88, 0.80, 0.66), (0.92, 0.85, 0.72), (0.66, 0.47, 0.33), (0.62, 0.44, 0.31),
            (0.88, 0.84, 0.76), (0.86, 0.82, 0.74), (0.64, 0.51, 0.40), (0.80, 0.74, 0.64),
            (0.70, 0.63, 0.54), (0.60, 0.56, 0.50), (0.50, 0.47, 0.44)]
    alb = _bands(latw, keys, cols)
    alb *= (0.93 + 0.14 * (fine * 0.5 + 0.5))[..., None]
    # Grande Mancha Vermelha (elipse com halo claro)
    dl = (lon - 40.0) * np.cos(np.radians(lat)) / 12.0
    dy = (lat + 22.0) / 6.0
    r = np.sqrt(dl * dl + dy * dy) + 0.15 * fbm(d * 12, 3, seed=82)
    alb = mix(alb, (0.93, 0.88, 0.80), sstep(1.35, 1.05, r) * 0.7)
    alb = mix(alb, (0.72, 0.36, 0.22), sstep(1.0, 0.55, r))
    return {"albedo": to_u8(alb)}


def gen_saturn(W):
    d, lat, lon = sphere_dirs(W)
    turb = fbm(d * np.array([2, 14, 2], F32), 5, seed=90)
    latw = lat + 1.2 * turb
    keys = [90, 75, 62, 45, 30, 20, 10, 0, -10, -20, -32, -45, -62, -75, -90]
    cols = [(0.50, 0.55, 0.60), (0.62, 0.62, 0.58), (0.72, 0.66, 0.52), (0.80, 0.72, 0.55),
            (0.84, 0.76, 0.57), (0.78, 0.69, 0.50), (0.86, 0.78, 0.58), (0.90, 0.82, 0.62),
            (0.86, 0.78, 0.58), (0.79, 0.70, 0.51), (0.84, 0.76, 0.58), (0.78, 0.70, 0.54),
            (0.72, 0.65, 0.50), (0.65, 0.60, 0.50), (0.58, 0.55, 0.50)]
    alb = _bands(latw, keys, cols)
    alb *= (0.96 + 0.05 * np.sin(np.radians(latw) * 40))[..., None]
    return {"albedo": to_u8(alb)}


def gen_uranus(W):
    d, lat, lon = sphere_dirs(W)
    latw = lat + 1.5 * fbm(d * np.array([2, 10, 2], F32), 4, seed=100)
    alb = np.zeros(lat.shape + (3,), F32) + np.array([0.62, 0.82, 0.86], F32)
    alb = alb * (0.97 + 0.03 * np.sin(np.radians(latw) * 14))[..., None]
    alb = np.broadcast_to(alb, d.shape).copy()
    alb = mix(alb, (0.72, 0.88, 0.90), sstep(55, 80, lat))
    return {"albedo": to_u8(alb)}


def gen_neptune(W):
    d, lat, lon = sphere_dirs(W)
    turb = fbm(d * np.array([3, 14, 3], F32), 5, seed=110)
    latw = lat + 3 * turb
    keys = [90, 60, 40, 25, 10, -10, -20, -30, -45, -60, -90]
    cols = [(0.22, 0.36, 0.70), (0.25, 0.42, 0.80), (0.28, 0.46, 0.84), (0.24, 0.40, 0.78),
            (0.26, 0.44, 0.82), (0.24, 0.42, 0.80), (0.18, 0.32, 0.68), (0.26, 0.44, 0.82),
            (0.22, 0.38, 0.76), (0.25, 0.42, 0.78), (0.20, 0.34, 0.68)]
    alb = _bands(latw, keys, cols)
    dl = (lon + 30) * np.cos(np.radians(lat)) / 9.0
    dy = (lat + 20) / 5.0
    r = np.sqrt(dl * dl + dy * dy)
    alb = mix(alb, (0.10, 0.17, 0.42), sstep(1.0, 0.6, r))
    streak = sstep(0.55, 0.8, fbm(d * np.array([4, 30, 4], F32), 4, seed=111)) * (np.exp(-((lat + 26) / 4) ** 2) + np.exp(-((lat - 30) / 5) ** 2))
    alb = mix(alb, (0.90, 0.93, 0.98), np.clip(streak, 0, 1) * 0.8)
    return {"albedo": to_u8(alb)}


def gen_io(W):
    d, lat, lon = sphere_dirs(W)
    n = fbm(d * 4, 6, seed=120)
    alb = mix((0.86, 0.80, 0.42), (0.80, 0.52, 0.22), sstep(0.0, 0.3, n))
    alb = mix(alb, (0.93, 0.92, 0.82), sstep(0.15, 0.4, fbm(d * 3, 5, seed=121)))
    rng = np.random.default_rng(122)
    flat = d.reshape(-1, 3)
    spots = np.zeros(flat.shape[0], F32)
    ring = np.zeros(flat.shape[0], F32)
    for _ in range(40):
        c = rng.normal(size=3)
        c /= np.linalg.norm(c)
        r = rng.uniform(0.01, 0.05)
        x = np.arccos(np.clip(flat @ c.astype(F32), -1, 1)) / r
        spots = np.maximum(spots, sstep(1.0, 0.5, x))
        ring = np.maximum(ring, sstep(3.5, 1.2, x) * 0.6)
    alb = mix(alb, (0.70, 0.30, 0.12), ring.reshape(lat.shape[0], -1))
    alb = mix(alb, (0.10, 0.07, 0.05), spots.reshape(lat.shape[0], -1))
    return {"albedo": to_u8(alb)}


def gen_europa(W):
    d, lat, lon = sphere_dirs(W)
    alb = mix((0.88, 0.85, 0.78), (0.70, 0.60, 0.48), sstep(0.0, 0.4, fbm(d * 2.5, 5, seed=130)))
    lines = np.zeros(lat.shape[:1] + (W,), F32)
    for s in range(4):
        r = ridged(d * (2.5 + s) + F32(s), 5, seed=131 + s)
        lines = np.maximum(lines, sstep(0.94, 0.99, r))
    alb = mix(alb, (0.52, 0.32, 0.20), lines * 0.85)
    return {"albedo": to_u8(alb), "normal": normal_from_height(lines * 0.01, 1.0)}


def gen_ganymede(W):
    d, lat, lon = sphere_dirs(W)
    light = sstep(-0.05, 0.15, fbm(d * 2.2, 6, seed=140))
    grooves = 0.5 + 0.5 * np.sin(fbm(d * 3, 3, seed=141) * 60)
    cr, br = add_craters(d, 250, 0.004, 0.08, seed=142, depth=0.8, rays=6)
    alb = mix((0.38, 0.34, 0.30), (0.66, 0.63, 0.58), light)
    alb *= (1 - 0.12 * grooves * light)[..., None]
    alb = mix(alb, (0.85, 0.85, 0.83), np.clip(br, 0, 1) * 0.8)
    return {"albedo": to_u8(alb), "normal": normal_from_height(cr * 5 + grooves * light * 0.002, 1.2)}


def gen_callisto(W):
    d, lat, lon = sphere_dirs(W)
    base = fbm(d * 3, 6, seed=150)
    cr, br = add_craters(d, 500, 0.003, 0.09, seed=151, depth=0.8, rays=20)
    alb = mix((0.26, 0.23, 0.20), (0.38, 0.34, 0.30), base * 0.5 + 0.5)
    alb = mix(alb, (0.80, 0.80, 0.78), np.clip(br, 0, 1))
    return {"albedo": to_u8(alb), "normal": normal_from_height(cr * 5, 1.2)}


def gen_titan(W):
    d, lat, lon = sphere_dirs(W)
    n = fbm(d * 3, 5, seed=160)
    alb = mix((0.85, 0.62, 0.30), (0.70, 0.48, 0.22), sstep(0.0, 0.4, n) * 0.5)
    return {"albedo": to_u8(alb)}


def gen_rings(W=2048):
    """Perfil radial dos anéis de Saturno. u: 0 = raio interno (1.11 Rs) -> 1 = externo (2.27 Rs)."""
    r = 1.11 + (np.arange(W) + 0.5) / W * (2.27 - 1.11)
    rng = np.random.default_rng(170)
    fine = np.zeros(W)
    for _ in range(60):
        fine += np.sin(r * rng.uniform(50, 900) + rng.uniform(0, 6.3)) / 60
    fine = fine / np.abs(fine).max()
    a = np.full(W, 0.03)
    a = np.where((r >= 1.24) & (r < 1.53), 0.12 + 0.06 * fine, a)                               # anel C
    a = np.where((r >= 1.53) & (r < 1.95), 0.80 + 0.12 * fine + 0.08 * sstep(1.53, 1.75, r), a)   # anel B
    a = np.where((r >= 1.95) & (r < 2.03), 0.06, a)                                              # divisão de Cassini
    a = np.where((r >= 2.03) & (r < 2.27), 0.55 + 0.1 * fine, a)                                 # anel A
    a = np.where((r > 2.205) & (r < 2.215), 0.02, a)                                             # divisão de Encke
    a *= sstep(1.11, 1.13, r) * sstep(2.27, 2.26, r)
    col = np.where((r < 1.53)[:, None], [0.55, 0.52, 0.48], np.where((r < 2.03)[:, None], [0.86, 0.79, 0.66], [0.78, 0.72, 0.62]))
    col = col * (0.9 + 0.1 * fine)[:, None]
    rgba = np.concatenate([col, np.clip(a, 0, 1)[:, None]], 1)
    return {"rings": to_u8(np.repeat(rgba[None], 4, 0))}


def gen_sky(W=4096):
    """Via Láctea: faixa difusa no plano galáctico (inclinado ~60° em relação à eclíptica)."""
    d, lat, lon = sphere_dirs(W)
    gn = np.array([0.0, math.cos(math.radians(60)), math.sin(math.radians(60))], F32)
    gc = np.array([1.0, 0.0, 0.0], F32)
    e2 = np.cross(gn, gc).astype(F32)
    b = np.arcsin(np.clip(d @ gn, -1, 1))
    l = np.arctan2(d @ e2, d @ gc)
    band = np.exp(-(b / 0.13) ** 2) * (0.45 + 0.55 * np.exp(-(l / 0.9) ** 2))
    bulge = np.exp(-((l / 0.30) ** 2 + (b / 0.16) ** 2))
    mott = sstep(-0.4, 0.6, fbm(d * 10, 6, seed=181))
    dust = sstep(0.0, 0.45, fbm(d * 5, 6, seed=182)) * np.exp(-(b / 0.06) ** 2)
    v = (band * (0.5 + 0.8 * mott) + bulge * 0.9) * (1 - 0.8 * dust)
    v += 0.04 * sstep(-0.2, 0.8, fbm(d * 3, 4, seed=183))
    col = mix((0.70, 0.76, 1.0), (1.0, 0.85, 0.66), np.clip(bulge * 1.5 + 0.2, 0, 1))
    return {"sky": to_u8(col * (v * 0.5)[..., None])}


def gen_suit(W=512):
    y, x = np.mgrid[0:W, 0:W].astype(F32) / W
    p = np.stack([x * 8, y * 8, np.zeros_like(x)], -1)
    weave = 0.025 * np.sin(x * 2 * math.pi * 96) * np.sin(y * 2 * math.pi * 96)
    n = fbm(p, 4, seed=190) * 0.04
    v = 0.90 + weave + n
    seams = np.zeros_like(x)
    for s in (0.25, 0.5, 0.75):
        seams = np.maximum(seams, np.exp(-((y - s) / 0.004) ** 2))
    seams = np.maximum(seams, np.exp(-((x - 0.5) / 0.003) ** 2) * 0.6)
    v = v * (1 - 0.25 * seams)
    col = np.stack([v, v * 0.995, v * 0.98], -1)
    return {"suit": to_u8(col)}


GENERATORS = {
    "earth": (gen_earth, 2048), "moon": (gen_moon, 2048), "mercury": (gen_mercury, 2048),
    "venus": (gen_venus, 1024), "mars": (gen_mars, 1024), "jupiter": (gen_jupiter, 2048),
    "saturn": (gen_saturn, 1024), "uranus": (gen_uranus, 512), "neptune": (gen_neptune, 1024),
    "io": (gen_io, 1024), "europa": (gen_europa, 1024), "ganymede": (gen_ganymede, 1024),
    "callisto": (gen_callisto, 1024), "titan": (gen_titan, 512),
    "rings": (gen_rings, 2048), "sky": (gen_sky, 4096), "suit": (gen_suit, 512),
}
