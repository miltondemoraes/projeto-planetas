"""Dados físicos e mecânica orbital (simplificada) do Sistema Solar.

ESCALA (ver DOCUMENTACAO.md): 1 unidade = 1 raio terrestre (6.371 km).
 - Raios dos planetas e luas: proporções REAIS entre si.
 - Sol: reduzido (20 em vez de 109) para não engolir as órbitas internas.
 - Distâncias orbitais: comprimidas (senão os planetas seriam pontos invisíveis).
 - Períodos orbitais/rotação, inclinações axiais e orbitais: valores REAIS.
"""
import math
from dataclasses import dataclass, field
import numpy as np

from . import glmath as gm

EARTH_ORBIT = 320.0   # raio orbital da Terra na cena (referência para a irradiância)


@dataclass
class BodyDef:
    key: str
    name: str
    radius: float
    parent: str = None
    orbit: float = 0.0          # raio orbital (cena)
    period: float = 1.0         # período orbital (dias)
    rot: float = 1.0            # período de rotação sideral (dias)
    tilt: float = 0.0           # obliquidade (graus)
    incl: float = 0.0           # inclinação orbital (graus)
    node: float = 0.0           # longitude do nodo ascendente (graus)
    phase: float = 0.0          # ângulo orbital inicial (graus)
    locked: bool = False        # rotação sincronizada (mesma face para o planeta)
    equatorial: bool = False    # órbita no plano equatorial do pai
    roughness: float = 0.0      # Oren-Nayar sigma (0 = Lambert)
    spec: float = 0.0
    shininess: float = 32.0
    normal_strength: float = 1.0
    atmo: dict = None
    clouds: bool = False
    rings: bool = False
    night: bool = False
    emissive: bool = False
    color: tuple = (0.5, 0.5, 0.5)
    info: list = field(default_factory=list)


def _atmo(ra, tauR, tauM, hr, hm, g=0.76, k=1.0):
    return {"ra": ra, "betaR": np.array(tauR) / hr, "betaM": np.array(tauM) / hm, "hr": hr, "hm": hm, "g": g, "k": k}


BODIES = [
    BodyDef("sun", "Sol", 20.0, rot=25.38, tilt=7.25, emissive=True, color=(1, 0.9, 0.7),
            info=["Estrela anã amarela (G2V)", "Raio real: 696.340 km (109 Terras)",
                  "Temperatura da fotosfera: ~5.500 °C", "Contém 99,86% da massa do Sistema Solar"]),
    BodyDef("mercury", "Mercúrio", 0.383, "sun", 150, 87.97, 58.65, 0.03, 7.0, 48.3, 40, roughness=0.6, normal_strength=1.0,
            info=["Raio: 2.440 km   Distância: 0,39 UA", "Ano: 88 dias   Dia solar: 176 dias",
                  "Sem atmosfera: -180 °C a 430 °C"]),
    BodyDef("venus", "Vênus", 0.949, "sun", 230, 224.7, 243.0, 177.36, 3.39, 76.7, 200,
            atmo=_atmo(1.06, (0.30, 0.24, 0.15), (0.12, 0.10, 0.08), 0.010, 0.004, 0.7, 1.0),
            info=["Raio: 6.052 km   Distância: 0,72 UA", "Rotação retrógrada de 243 dias",
                  "Efeito estufa extremo: 465 °C", "Nuvens de ácido sulfúrico"]),
    BodyDef("earth", "Terra", 1.0, "sun", EARTH_ORBIT, 365.256, 0.99727, 23.44, 0.0, 0, 110,
            spec=1.0, shininess=220, normal_strength=1.2, clouds=True, night=True,
            atmo=_atmo(1.035, (0.046, 0.108, 0.265), (0.021, 0.021, 0.021), 0.0045, 0.0012, 0.76, 1.0),
            info=["Raio: 6.371 km   Distância: 1 UA (150 milhões km)", "Dia: 23h56m   Ano: 365,25 dias",
                  "Inclinação do eixo: 23,4° (estações do ano)", "71% da superfície coberta por oceanos"]),
    BodyDef("moon", "Lua", 0.273, "earth", 12.0, 27.32, 27.32, 6.68, 5.14, 125, 60, locked=True, roughness=0.75,
            normal_strength=0.7,
            info=["Raio: 1.737 km   A 384.400 km da Terra", "Rotação sincronizada: sempre a mesma face",
                  "Mares de basalto e crateras de impacto"]),
    BodyDef("mars", "Marte", 0.532, "sun", 450, 686.98, 1.026, 25.19, 1.85, 49.6, 300, roughness=0.5, normal_strength=1.0,
            atmo=_atmo(1.025, (0.010, 0.015, 0.025), (0.06, 0.04, 0.025), 0.003, 0.003, 0.7, 1.0),
            info=["Raio: 3.390 km   Distância: 1,52 UA", "Dia: 24h37m   Ano: 687 dias",
                  "Óxido de ferro dá a cor vermelha", "Valles Marineris: cânion de 4.000 km"]),
    BodyDef("jupiter", "Júpiter", 11.21, "sun", 900, 4332.6, 0.4135, 3.13, 1.30, 100.5, 160,
            atmo=_atmo(1.012, (0.02, 0.04, 0.09), (0.02, 0.02, 0.02), 0.002, 0.001, 0.7, 1.0),
            info=["Raio: 69.911 km (11 Terras)   Distância: 5,2 UA", "Dia: 9h56m   Ano: 11,9 anos",
                  "Grande Mancha Vermelha: tempestade maior que a Terra", "95 luas conhecidas"]),
    BodyDef("io", "Io", 0.286, "jupiter", 22.0, 1.769, 1.769, locked=True, equatorial=True, phase=10, roughness=0.3,
            info=["Raio: 1.822 km", "O corpo mais vulcânico do Sistema Solar"]),
    BodyDef("europa", "Europa", 0.245, "jupiter", 30.0, 3.551, 3.551, locked=True, equatorial=True, phase=130, roughness=0.2,
            info=["Raio: 1.561 km", "Crosta de gelo sobre um oceano líquido"]),
    BodyDef("ganymede", "Ganimedes", 0.413, "jupiter", 40.0, 7.155, 7.155, locked=True, equatorial=True, phase=250, roughness=0.4,
            info=["Raio: 2.634 km", "Maior lua do Sistema Solar (maior que Mercúrio)"]),
    BodyDef("callisto", "Calisto", 0.378, "jupiter", 58.0, 16.69, 16.69, locked=True, equatorial=True, phase=320, roughness=0.6,
            info=["Raio: 2.410 km", "Superfície mais craterada conhecida"]),
    BodyDef("saturn", "Saturno", 9.45, "sun", 1300, 10759.2, 0.444, 26.73, 2.49, 113.7, 230, rings=True,
            atmo=_atmo(1.012, (0.02, 0.035, 0.06), (0.02, 0.02, 0.015), 0.002, 0.001, 0.7, 1.0),
            info=["Raio: 58.232 km   Distância: 9,5 UA", "Dia: 10h33m   Ano: 29,4 anos",
                  "Anéis: 280.000 km de largura, ~10 m de espessura", "Densidade menor que a da água"]),
    BodyDef("titan", "Titã", 0.404, "saturn", 50.0, 15.95, 15.95, locked=True, equatorial=True, phase=80, roughness=0.2,
            atmo=_atmo(1.10, (0.01, 0.015, 0.02), (0.45, 0.28, 0.12), 0.02, 0.015, 0.6, 1.0),
            info=["Raio: 2.575 km", "Atmosfera densa de nitrogênio", "Lagos de metano líquido"]),
    BodyDef("uranus", "Urano", 4.01, "sun", 1800, 30688.5, 0.718, 97.77, 0.77, 74.0, 20,
            atmo=_atmo(1.02, (0.05, 0.12, 0.18), (0.01, 0.01, 0.01), 0.003, 0.0015, 0.7, 1.0),
            info=["Raio: 25.362 km   Distância: 19,2 UA", "Eixo inclinado 97,8°: gira \"deitado\"",
                  "Dia: 17h14m   Ano: 84 anos"]),
    BodyDef("neptune", "Netuno", 3.88, "sun", 2300, 60182.0, 0.671, 28.32, 1.77, 131.8, 280,
            atmo=_atmo(1.02, (0.04, 0.10, 0.25), (0.01, 0.01, 0.01), 0.003, 0.0015, 0.7, 1.0),
            info=["Raio: 24.622 km   Distância: 30,1 UA", "Ventos de até 2.100 km/h",
                  "Dia: 16h06m   Ano: 165 anos"]),
]

HOTKEYS = ["sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune", "moon"]

RING_INNER, RING_OUTER = 1.11, 2.27   # em raios de Saturno


class Body:
    def __init__(self, d: BodyDef):
        self.d = d
        self.key = d.key
        self.radius = d.radius
        self.parent = None
        self.children = []
        self.pos = np.zeros(3)
        self.prev_pos = np.zeros(3)
        self.rot = np.eye(3)
        self.spin = 0.0
        self.prev_spin = 0.0
        self.frame = np.eye(3)       # orientação do plano orbital
        self.avg_color = np.array(d.color, float)

    def orbit_frame(self):
        d = self.d
        if d.equatorial and self.parent is not None:
            return gm.rot_x(math.radians(self.parent.d.tilt))[:3, :3]
        return (gm.rot_y(math.radians(d.node)) @ gm.rot_x(math.radians(d.incl)))[:3, :3]

    def soi(self):
        """'Esfera de influência' de jogo: dentro dela o astronauta é carregado junto."""
        r = self.radius * 8 + 2
        if self.children:
            r = max(r, max(c.d.orbit for c in self.children) * 1.3)
        return r


class SolarSystem:
    def __init__(self):
        self.bodies = [Body(d) for d in BODIES]
        self.by_key = {b.key: b for b in self.bodies}
        for b in self.bodies:
            if b.d.parent:
                b.parent = self.by_key[b.d.parent]
                b.parent.children.append(b)
        for b in self.bodies:
            b.frame = b.orbit_frame()
        self.sun = self.by_key["sun"]
        self.time = 0.0
        self.update(0.0, first=True)

    def orbit_angle(self, b, t):
        return math.radians(b.d.phase) + 2 * math.pi * t / b.d.period

    def update(self, t_days, first=False):
        self.time = t_days
        for b in self.bodies:   # lista em ordem: pais antes dos filhos
            b.prev_pos = b.pos.copy()
            b.prev_spin = b.spin
            d = b.d
            if b.parent is None:
                b.pos = np.zeros(3)
            else:
                th = self.orbit_angle(b, t_days)
                local = np.array([d.orbit * math.cos(th), 0.0, -d.orbit * math.sin(th)])
                b.pos = b.parent.pos + b.frame @ local
            if d.locked:
                th = self.orbit_angle(b, t_days)
                b.spin = th - math.pi / 2
                b.rot = b.frame @ gm.rot_y(b.spin)[:3, :3]
            else:
                b.spin = 2 * math.pi * t_days / d.rot
                b.rot = gm.rot_x(math.radians(d.tilt))[:3, :3] @ gm.rot_y(b.spin)[:3, :3]
            if first:
                b.prev_pos = b.pos.copy()
                b.prev_spin = b.spin

    def irradiance(self, p):
        """Irradiância solar relativa à da Terra (lei do inverso do quadrado da distância)."""
        d = max(np.linalg.norm(p - self.sun.pos), self.sun.radius)
        return (EARTH_ORBIT / d) ** 2

    def nearest(self, p, exclude=None, include_sun=True):
        best, bd = None, 1e30
        for b in self.bodies:
            if b is exclude or (not include_sun and b is self.sun):
                continue
            dd = np.linalg.norm(p - b.pos) - b.radius
            if dd < bd:
                best, bd = b, dd
        return best, bd

    def occluders_for(self, p, exclude=None, n=6):
        cand = [b for b in self.bodies if b is not self.sun and b is not exclude]
        cand.sort(key=lambda b: np.linalg.norm(b.pos - p) / b.radius)
        return cand[:n]

    def frame_body(self, p):
        """Corpo cujo referencial carrega o astronauta (a menor SOI que o contém)."""
        best = None
        for b in self.bodies:
            if b is self.sun:
                continue
            if np.linalg.norm(p - b.pos) < b.soi():
                if best is None or b.soi() < best.soi():
                    best = b
        return best
