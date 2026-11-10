"""Astronauta: modelagem HIERÁRQUICA (grafo de cena) + animação procedural + controle de voo.

Cada parte do corpo tem uma transformação LOCAL relativa ao pai (articulação).
A matriz final de cada parte é o produto das matrizes da raiz até ela:
    M_antebraço = M_raiz · M_tronco · M_ombro · M_cotovelo
Assim, girar o ombro move automaticamente cotovelo, antebraço e luva.
Unidades do modelo: metros (altura ~1,9 m); a escala no mundo é ASTRO_SCALE.
"""
import math
import numpy as np

from . import glmath as gm
from . import mesh as M

ASTRO_SCALE = 0.0032          # 1,9 m de modelo -> ~0,006 unidades (raios terrestres)
# Velocidade ADAPTATIVA: v = multiplicador × K × altitude (distância até a superfície mais próxima).
# Perto de um planeta o voo é lento e preciso; no vazio entre planetas acelera sozinho
# (atravessar o Sistema Solar leva segundos). A roda do mouse escolhe o multiplicador.
SPEED_MULTS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
SPEED_K = 1.6            # unidades/s por unidade de altitude (com multiplicador 1×)
SPEED_MIN, SPEED_MAX = 0.004, 4000.0

MATERIALS = {
    #          cor base             textura  F0                    brilho  espec  espelho
    "suit":   ((0.95, 0.95, 0.93), 1, (0.03, 0.03, 0.03), 14.0, 0.35, 0.0),
    "helmet": ((0.96, 0.96, 0.96), 0, (0.05, 0.05, 0.05), 120.0, 1.0, 0.0),
    "visor":  ((0.02, 0.015, 0.01), 0, (1.0, 0.77, 0.34), 1500.0, 1.0, 1.0),
    "gray":   ((0.50, 0.51, 0.53), 0, (0.04, 0.04, 0.04), 40.0, 0.5, 0.0),
    "dark":   ((0.10, 0.10, 0.11), 0, (0.30, 0.30, 0.32), 80.0, 0.8, 0.0),
    "metal":  ((0.30, 0.30, 0.32), 0, (0.90, 0.90, 0.92), 200.0, 1.0, 0.0),
    "glove":  ((0.80, 0.80, 0.78), 1, (0.03, 0.03, 0.03), 10.0, 0.3, 0.0),
    "red":    ((0.55, 0.05, 0.04), 0, (0.04, 0.04, 0.04), 30.0, 0.4, 0.0),
    "lamp":   ((0.9, 0.9, 0.8), 0, (0.04, 0.04, 0.04), 60.0, 0.5, 0.0),
}


def build_meshes():
    return {
        "sphere": M.Mesh(*M.uv_sphere(48, 24)),
        "pelvis": M.Mesh(*M.superellipsoid(0.17, 0.11, 0.13, 0.5, 0.5)),
        "torso": M.Mesh(*M.superellipsoid(0.22, 0.29, 0.15, 0.55, 0.5)),
        "pack": M.Mesh(*M.superellipsoid(0.20, 0.30, 0.10, 0.25, 0.25)),
        "chest": M.Mesh(*M.superellipsoid(0.10, 0.06, 0.045, 0.25, 0.25)),
        "neck": M.Mesh(*M.superellipsoid(0.135, 0.035, 0.135, 0.4, 1.0)),
        "upper": M.Mesh(*M.capsule(0.068, 0.26)),
        "fore": M.Mesh(*M.capsule(0.060, 0.24)),
        "thigh": M.Mesh(*M.capsule(0.085, 0.38)),
        "shin": M.Mesh(*M.capsule(0.075, 0.36)),
        "glove": M.Mesh(*M.superellipsoid(0.05, 0.075, 0.035, 0.6, 0.6)),
        "boot": M.Mesh(*M.superellipsoid(0.075, 0.07, 0.135, 0.4, 0.4)),
        "ring": M.Mesh(*M.superellipsoid(0.07, 0.022, 0.07, 0.5, 1.0)),
    }


IDLE = dict(pitch=0.0, waist=0.0, head=0.0, r_raise=0.15, r_abd=0.25, r_elb=0.35, l_raise=0.15, l_abd=0.25,
            l_elb=0.35, r_hip=0.10, l_hip=-0.05, r_knee=-0.30, l_knee=-0.45, spread=0.07)
FLY = dict(pitch=-1.45, waist=0.05, head=1.15, r_raise=2.95, r_abd=0.12, r_elb=0.05, l_raise=0.25, l_abd=0.18,
           l_elb=0.35, r_hip=0.02, l_hip=0.08, r_knee=-0.12, l_knee=-0.55, spread=0.04)


def pose_blend(fly, t):
    p = {k: IDLE[k] + (FLY[k] - IDLE[k]) * fly for k in IDLE}
    br = math.sin(t * 1.3)
    w = 1.0 - fly
    p["r_raise"] += 0.06 * br * w
    p["l_raise"] -= 0.06 * br * w
    p["r_hip"] += 0.05 * math.sin(t * 0.9) * w
    p["l_hip"] -= 0.05 * math.sin(t * 0.9) * w
    p["l_knee"] += 0.08 * math.sin(t * 2.2) * fly      # pernas "tremulando" no voo
    p["head"] += 0.04 * math.sin(t * 0.7)
    return p


def build_parts(pose):
    """Percorre o grafo de cena e devolve [(malha, matriz no espaço do modelo, material)]."""
    T, S, Rx, Rz = gm.translate, gm.scale, gm.rot_x, gm.rot_z
    out = []
    root = Rx(pose["pitch"])
    out.append(("pelvis", root, "suit"))
    torso = root @ T((0, 0.08, 0)) @ Rx(pose["waist"])
    out.append(("torso", torso @ T((0, 0.29, 0)), "suit"))
    out.append(("pack", torso @ T((0, 0.33, 0.24)), "suit"))
    out.append(("chest", torso @ T((0, 0.38, -0.155)), "gray"))
    out.append(("sphere", torso @ T((0.05, 0.40, -0.205)) @ S(0.012), "red"))
    out.append(("sphere", torso @ T((-0.03, 0.40, -0.205)) @ S(0.010), "lamp"))
    neck = torso @ T((0, 0.60, 0)) @ Rx(pose["head"])
    out.append(("neck", neck, "metal"))
    out.append(("sphere", neck @ T((0, 0.15, 0)) @ S(0.175), "helmet"))
    out.append(("sphere", neck @ T((0, 0.155, -0.058)) @ S((0.150, 0.120, 0.138)), "visor"))
    for s in (-1, 1):
        out.append(("sphere", neck @ T((s * 0.115, 0.26, -0.095)) @ S(0.022), "lamp"))
    for s, side in ((1, "r"), (-1, "l")):
        sh = torso @ T((s * 0.27, 0.50, 0)) @ Rz(s * pose[side + "_abd"]) @ Rx(pose[side + "_raise"])
        out.append(("sphere", sh @ S(0.088), "suit"))
        out.append(("upper", sh, "suit"))
        out.append(("sphere", sh @ T((s * 0.069, -0.07, 0)) @ S((0.004, 0.03, 0.03)), "red"))
        el = sh @ T((0, -0.28, 0)) @ Rx(pose[side + "_elb"])
        out.append(("fore", el, "suit"))
        wr = el @ T((0, -0.26, 0))
        out.append(("ring", wr, "dark"))
        out.append(("glove", wr @ T((0, -0.075, 0)), "glove"))
        hip = root @ T((s * 0.10, -0.04, 0)) @ Rz(s * pose["spread"]) @ Rx(pose[side + "_hip"])
        out.append(("thigh", hip, "suit"))
        kn = hip @ T((0, -0.40, 0)) @ Rx(pose[side + "_knee"])
        out.append(("shin", kn, "suit"))
        an = kn @ T((0, -0.38, 0)) @ Rx(-pose[side + "_knee"] * 0.3)
        out.append(("ring", an @ T((0, 0.02, 0)) @ S((1.15, 1, 1.15)), "dark"))
        out.append(("boot", an @ T((0, -0.06, -0.035)), "gray"))
    return out


def head_offset(pose):
    """Posição do olho (centro do capacete) no espaço do modelo."""
    torso = gm.rot_x(pose["pitch"]) @ gm.translate((0, 0.08, 0)) @ gm.rot_x(pose["waist"])
    neck = torso @ gm.translate((0, 0.60, 0)) @ gm.rot_x(pose["head"])
    return (neck @ np.array([0, 0.15, -0.02, 1.0]))[:3]


class Astronaut:
    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.q = np.array([1.0, 0, 0, 0])
        self.fly = 0.0
        self.speed_idx = 3
        self.altitude = 1.0
        self.lantern = False
        self.anim_t = 0.0
        self.autopilot = None
        self.pose = pose_blend(0.0, 0.0)

    @property
    def rot(self):
        return gm.q_to_mat3(self.q)

    def height(self):
        return 1.9 * ASTRO_SCALE

    def mult(self):
        return SPEED_MULTS[self.speed_idx]

    def speed(self):
        """Velocidade de cruzeiro atual (unidades/s), proporcional à altitude."""
        return float(np.clip(self.mult() * SPEED_K * max(self.altitude, 0.0), SPEED_MIN, SPEED_MAX))

    def rotate(self, yaw, pitch, roll):
        if yaw:
            self.q = gm.q_mul(self.q, gm.q_axis_angle((0, 1, 0), yaw))
        if pitch:
            self.q = gm.q_mul(self.q, gm.q_axis_angle((1, 0, 0), pitch))
        if roll:
            self.q = gm.q_mul(self.q, gm.q_axis_angle((0, 0, 1), roll))
        self.q = gm.q_normalize(self.q)

    def update(self, dt, thrust_local, boost, altitude):
        self.altitude = altitude
        target = self.rot @ (np.asarray(thrust_local, float) * self.speed() * (6.0 if boost else 1.0))
        self.vel += (target - self.vel) * gm.smooth_factor(4.0, dt)
        self.pos += self.vel * dt
        fwd = float(-(self.rot.T @ self.vel)[2])
        fly_target = float(np.clip(fwd / (self.speed() * 0.6), 0.0, 1.0))
        if self.autopilot:
            fly_target = 1.0
        self.fly += (fly_target - self.fly) * gm.smooth_factor(2.0, dt)
        self.anim_t += dt
        self.pose = pose_blend(self.fly, self.anim_t)

    def model_matrix(self, cam_pos):
        """Matriz raiz: posição relativa à câmera · orientação · escala."""
        return gm.translate(self.pos - cam_pos) @ gm.mat4_from_mat3(self.rot) @ gm.scale(ASTRO_SCALE)
