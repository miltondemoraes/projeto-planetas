"""Aplicação principal: janela, entrada, simulação e pipeline de renderização.

Pipeline por quadro
  1. Simulação: órbitas/rotações, astronauta, câmera.
  2. Shadow pass: profundidade do astronauta vista do Sol -> shadow map 2048².
  3. Passo HDR (FBO RGBA16F com MSAA 4x):
       céu (Via Láctea) + estrelas -> Sol -> planetas/luas (opacos) -> asteroides
       (instancing) -> astronauta -> linhas de órbita -> transparentes ordenados
       de trás p/ frente (nuvens, atmosferas, anéis) -> halo do Sol.
  4. Resolve MSAA -> bloom (cadeia de mips) -> composição: exposição + ACES + gama.
  5. HUD (texto) por cima, em LDR.
"""
import math
import os
import time

import glfw
import numpy as np
from OpenGL.GL import *

from . import glmath as gm
from . import mesh as M
from . import procedural
from .assets import TextureStore, flip, height_to_normal
from .astronaut import Astronaut, build_meshes, build_parts, head_offset, pose_blend, MATERIALS, ASTRO_SCALE, SPEED_MULTS
from .hud import TextRenderer
from .shader import Shader
from .solar_system import SolarSystem, HOTKEYS, RING_INNER, RING_OUTER
from .texture import texture_from_array, solid_texture, bind
from .audio import AudioEngine, body_voice_params
from .speech import Speech

NEAR, FAR = 1e-5, 1e5
SUN_COLOR = np.array([1.0, 0.96, 0.91])
SUN_EMISSION = 15.0
ATMO_K = 3.0
SHADOW_RES = 2048
TIME_SCALES = [0.0, 0.0002, 0.001, 0.01, 0.05, 0.25, 1.0, 5.0, 25.0, 100.0]
VOICE_KIND = {"mercury": "rocky", "venus": "lava", "earth": "earth", "moon": "rocky", "mars": "rocky",
              "jupiter": "gas", "io": "lava", "europa": "ice", "ganymede": "rocky", "callisto": "rocky",
              "saturn": "gas", "titan": "gas", "uranus": "ice", "neptune": "ice"}
SHADING_NAMES = ["Phong (por fragmento)", "Gouraud (por vértice)", "Flat (por face)"]


class Options:
    def __init__(self):
        self.shading = 0
        self.normal_map = True
        self.shadows = True
        self.textures = True
        self.atmosphere = True
        self.bloom = True
        self.wireframe = False
        self.tonemap = True
        self.orbits = False
        self.labels = True
        self.hud = True
        self.help = False


class App:
    def __init__(self, args):
        self.args = args
        self.opt = Options()
        if args.nohud:
            self.opt.hud = False
        self.opt.shading = args.shading
        self.opt.orbits = args.orbits
        self.opt.wireframe = args.wire
        print("Preparando texturas (a 1ª execução gera e salva no cache)...")
        self.store = TextureStore(use_real=not args.procedural, quality=args.quality)
        self._preload_arrays()
        self._init_window()
        self._init_gl()
        self.system = SolarSystem()
        self.time_idx = 3
        self.sim_time = args.time
        self.system.update(self.sim_time, first=True)
        self.astro = Astronaut()
        self.astro.lantern = args.lantern
        self.audio = AudioEngine(enabled=not (args.shot or args.mute),
                                 bodies=[body_voice_params(b.key, b.radius, VOICE_KIND[b.key])
                                         for b in self.system.bodies if b.key in VOICE_KIND])
        self.touching = False
        self.speech = Speech(self.audio, enabled=not args.shot or getattr(args, "speak", False))
        if args.fly:
            self.astro.fly = 1.0
            self.astro.pose = pose_blend(1.0, 0.0)
        self.cam_q = self.astro.q.copy()
        self.cam_pos = np.zeros(3)
        self.cam_mode = 1 if args.fp else 0     # 0 = terceira pessoa, 1 = primeira pessoa
        self.cam_orbit = np.array([math.radians(args.camyaw), 0.0])   # yaw/pitch extra (botão direito)
        self.cam_zoom = 1.0
        self.exposure = 1.0
        self.target_idx = 3
        self.keys = set()
        self.mouse_dx = self.mouse_dy = 0.0
        self.last_mouse = None
        self.rmb = False
        self.fps = 60.0
        self.message, self.message_t = "", 0.0
        self.place_near(args.goto, args.dist, args.angle, args.elev)
        self.cam_q = self.astro.q.copy()
        self.exposure = self._exposure_target()
        self._update_camera(1.0, snap=True)

    # ------------------------------------------------------------------ setup
    def _preload_arrays(self):
        self.arrays = {}
        keys = ["sun", "mercury", "venus", "earth", "moon", "mars", "jupiter", "io", "europa", "ganymede",
                "callisto", "saturn", "titan", "uranus", "neptune"]
        for k in keys:
            for role in ("albedo", "normal", "spec", "night", "clouds"):
                a = self.store.get(k, role)
                if a is not None:
                    self.arrays[(k, role)] = a
            if self.store.sources.get((k, "albedo")) == "real" and k in ("mercury", "moon", "mars") \
                    and (k, "normal") not in self.arrays:
                self.arrays[(k, "normal")] = height_to_normal(self.arrays[(k, "albedo")], 1.0)
        for k in ("rings", "sky", "suit"):
            self.arrays[(k, k)] = self.store.get(k, k)
        n_real = sum(1 for v in self.store.sources.values() if v == "real")
        print(f"Texturas: {n_real} reais, {len(self.store.sources) - n_real} procedurais.")

    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("Falha ao inicializar GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        if self.args.shot:
            glfw.window_hint(glfw.VISIBLE, False)
        mon = glfw.get_primary_monitor() if self.args.fullscreen else None
        w, h = self.args.width, self.args.height
        if mon:
            mode = glfw.get_video_mode(mon)
            w, h = mode.size.width, mode.size.height
        self.win = glfw.create_window(w, h, "Astronauta Cósmico — OpenGL", mon, None)
        if not self.win:
            raise RuntimeError("Não foi possível criar janela OpenGL 3.3")
        glfw.make_context_current(self.win)
        glfw.swap_interval(0 if self.args.shot else 1)
        glfw.set_key_callback(self.win, self._on_key)
        glfw.set_cursor_pos_callback(self.win, self._on_mouse)
        glfw.set_scroll_callback(self.win, self._on_scroll)
        glfw.set_mouse_button_callback(self.win, self._on_button)
        glfw.set_framebuffer_size_callback(self.win, lambda w_, a, b: self._create_fbos(a, b))
        self.captured = False
        if not self.args.shot:
            self._capture_mouse(True)
        print("OpenGL:", glGetString(GL_VERSION).decode(), "|", glGetString(GL_RENDERER).decode())

    def _capture_mouse(self, on):
        self.captured = on
        glfw.set_input_mode(self.win, glfw.CURSOR, glfw.CURSOR_DISABLED if on else glfw.CURSOR_NORMAL)
        if on and glfw.raw_mouse_motion_supported():
            glfw.set_input_mode(self.win, glfw.RAW_MOUSE_MOTION, True)
        self.last_mouse = None

    def _init_gl(self):
        S = Shader
        self.sh = {
            "planet": S("planet.vert", "planet.frag"),
            "astro": S("planet.vert", "astronaut.frag"),
            "ring": S("simple.vert", "ring.frag"),
            "atmo": S("simple.vert", "atmosphere.frag"),
            "sun": S("simple.vert", "sun.frag"),
            "glow": S("glow.vert", "glow.frag"),
            "sky": S("fullscreen.vert", "sky.frag"),
            "stars": S("stars.vert", "stars.frag"),
            "asteroid": S("asteroid.vert", "asteroid.frag"),
            "depth": S("depth.vert", "depth.frag"),
            "line": S("line.vert", "line.frag"),
            "down": S("fullscreen.vert", "bloom_down.frag"),
            "up": S("fullscreen.vert", "bloom_up.frag"),
            "comp": S("fullscreen.vert", "composite.frag"),
            "text": S("text.vert", "text.frag"),
        }
        self.empty_vao = glGenVertexArrays(1)
        self.sphere_hi = M.Mesh(*M.uv_sphere(256, 128))
        self.sphere_mid = M.Mesh(*M.uv_sphere(96, 48))
        self.ring_mesh = M.Mesh(*M.annulus(RING_INNER, RING_OUTER, 256))
        self.astro_meshes = build_meshes()
        self._make_textures()
        self._make_stars()
        self._make_asteroids()
        self.orbit_meshes = {}
        self.text = TextRenderer(self.sh["text"], 16)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_CULL_FACE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        self.samples = min(4, int(glGetIntegerv(GL_MAX_SAMPLES)))
        self._create_shadow_fbo()
        self.fbo_size = (0, 0)
        w, h = glfw.get_framebuffer_size(self.win)
        self._create_fbos(w, h)

    def _make_textures(self):
        self.tex = {}
        for (k, role), arr in self.arrays.items():
            srgb = role in ("albedo", "night", "rings", "sky", "suit")
            wrap_t = GL_CLAMP_TO_EDGE
            self.tex[(k, role)] = texture_from_array(flip(arr), srgb=srgb, wrap_s=GL_REPEAT if role != "rings" else GL_CLAMP_TO_EDGE,
                                                     wrap_t=wrap_t)
            if role == "albedo":
                lin = (arr[..., :3].astype(np.float32) / 255.0) ** 2.2
                self.avg_colors = getattr(self, "avg_colors", {})
                self.avg_colors[k] = lin.reshape(-1, 3).mean(0)
        self.white = solid_texture((255, 255, 255, 255))
        self.flat_normal = solid_texture((128, 128, 255, 255))

    def _make_stars(self):
        rng = np.random.default_rng(7)
        n = 9000
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        # metade concentrada perto do plano galáctico
        gn = np.array([0, math.cos(math.radians(60)), math.sin(math.radians(60))])
        gc = np.array([1.0, 0, 0])
        e2 = np.cross(gn, gc)
        m = n // 2
        b = rng.normal(0, 0.18, m)
        l = rng.uniform(-math.pi, math.pi, m)
        d[:m] = (np.cos(b)[:, None] * (np.cos(l)[:, None] * gc + np.sin(l)[:, None] * e2) + np.sin(b)[:, None] * gn)
        mag = np.clip(6.5 - rng.exponential(1.0, n), -1.4, 6.5)          # magnitude aparente
        flux = 0.015 * 10 ** (-0.4 * (mag - 6.5))                           # escala de Pogson
        temp = np.exp(rng.uniform(math.log(3000), math.log(25000), n))
        col = np.array([_blackbody(t) for t in temp])
        data = np.hstack([d, col * flux[:, None]]).astype(np.float32)
        self.star_count = n
        self.star_vao = glGenVertexArrays(1)
        glBindVertexArray(self.star_vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        import ctypes
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glBindVertexArray(0)

    def _make_asteroids(self):
        rng = np.random.default_rng(11)
        self.asteroids = []
        per = 1000
        for v in range(3):
            def disp(dirs, v=v):
                return 1.0 + 0.45 * procedural.fbm(dirs.astype(np.float32) * 1.7 + v * 5.0, 4, seed=300 + v)
            verts, idx = M.icosphere(3, disp)
            verts[:, 0:3] *= np.array([1.0, 0.7 + 0.1 * v, 0.85], np.float32)
            mesh = M.Mesh(verts, idx)
            r = rng.uniform(600, 720, per)
            a_real = 2.2 + (r - 600) / 120 * 1.1                            # UA reais (2,2–3,3)
            omega = 2 * math.pi / (365.25 * a_real ** 1.5)                   # 3ª lei de Kepler
            size = np.clip(0.03 * (rng.pareto(2.2, per) + 1), 0.02, 0.6)
            axis = rng.normal(size=(per, 3))
            inst = np.column_stack([r, rng.uniform(0, 2 * math.pi, per), rng.normal(0, 6, per), size,
                                    axis, rng.uniform(-25, 25, per), omega, rng.uniform(0, 1, per)])
            mesh.add_instance_buffer(inst, [4, 4, 2])
            self.asteroids.append(mesh)

    def _create_shadow_fbo(self):
        self.shadow_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.shadow_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, SHADOW_RES, SHADOW_RES, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1.0, 1.0, 1.0, 1.0])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)
        self.shadow_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.shadow_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.shadow_tex, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _color_tex(self, w, h, internal=GL_RGBA16F, fmt=GL_FLOAT):
        t = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, t)
        glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, GL_RGBA, fmt, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        return t

    def _fbo_with(self, tex):
        f = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, f)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        return f

    def _create_fbos(self, w, h):
        if w == 0 or h == 0 or (w, h) == self.fbo_size:
            return
        if self.fbo_size != (0, 0):
            glDeleteFramebuffers(len(self._fbos), self._fbos)
            glDeleteTextures(self._texs)
            glDeleteRenderbuffers(2, [self.msaa_color, self.msaa_depth])
        self.fbo_size = (w, h)
        self._fbos, self._texs = [], []
        # FBO HDR multiamostrado (MSAA)
        self.msaa_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.msaa_fbo)
        self.msaa_color = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.msaa_color)
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, self.samples, GL_RGBA16F, w, h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self.msaa_color)
        self.msaa_depth = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.msaa_depth)
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, self.samples, GL_DEPTH_COMPONENT32F, w, h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.msaa_depth)
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "FBO MSAA incompleto"
        self._fbos.append(self.msaa_fbo)
        # HDR resolvido
        self.hdr_tex = self._color_tex(w, h)
        self.hdr_fbo = self._fbo_with(self.hdr_tex)
        # cadeia de bloom
        self.bloom = []
        bw, bh = w, h
        for i in range(6):
            bw, bh = max(1, bw // 2), max(1, bh // 2)
            t = self._color_tex(bw, bh, GL_R11F_G11F_B10F)
            self.bloom.append((t, self._fbo_with(t), bw, bh))
            self._texs.append(t)
            self._fbos.append(self.bloom[-1][1])
        # LDR final (lido em screenshots)
        self.ldr_tex = self._color_tex(w, h, GL_RGBA8, GL_UNSIGNED_BYTE)
        self.ldr_fbo = self._fbo_with(self.ldr_tex)
        self._texs += [self.hdr_tex, self.ldr_tex]
        self._fbos += [self.hdr_fbo, self.ldr_fbo]
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    # ------------------------------------------------------------------ input
    def _on_key(self, win, key, sc, action, mods):
        if action == glfw.PRESS:
            self.keys.add(key)
            self._on_press(key, mods)
        elif action == glfw.RELEASE:
            self.keys.discard(key)

    def _say(self, msg):
        self.message, self.message_t = msg, 3.0

    def _on_press(self, key, mods):
        o = self.opt
        if key in (glfw.KEY_F1, glfw.KEY_F2, glfw.KEY_F3, glfw.KEY_F4, glfw.KEY_F5, glfw.KEY_F6, glfw.KEY_F7,
                   glfw.KEY_F8, glfw.KEY_F9, glfw.KEY_C, glfw.KEY_L, glfw.KEY_R, glfw.KEY_O, glfw.KEY_H, glfw.KEY_TAB):
            self.audio.play("click")
        if key == glfw.KEY_M:
            self._say("Som " + ("desligado" if self.audio.toggle_mute() else "ligado") if self.audio.ok else "Áudio indisponível")
        if key == glfw.KEY_ESCAPE:
            if self.captured:
                self._capture_mouse(False)
            else:
                glfw.set_window_should_close(self.win, True)
        elif glfw.KEY_0 <= key <= glfw.KEY_9:
            self.start_autopilot(self.system.by_key[HOTKEYS[key - glfw.KEY_0]])
        elif key == glfw.KEY_TAB:
            self.target_idx = (self.target_idx + (-1 if mods & glfw.MOD_SHIFT else 1)) % len(self.system.bodies)
            self._say("Alvo: " + self.system.bodies[self.target_idx].d.name + "  (Enter para viajar)")
        elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            self.start_autopilot(self.system.bodies[self.target_idx])
        elif key == glfw.KEY_C:
            self.cam_mode = 1 - self.cam_mode
            self._say("Câmera: " + ("primeira pessoa" if self.cam_mode else "terceira pessoa"))
        elif key == glfw.KEY_V:
            self.cam_orbit[:] = 0
            self.cam_zoom = 1.0
        elif key == glfw.KEY_L:
            self.astro.lantern = not self.astro.lantern
            self._say("Lanterna " + ("ligada" if self.astro.lantern else "desligada"))
        elif key == glfw.KEY_R:
            o.labels = not o.labels
        elif key == glfw.KEY_O:
            o.orbits = not o.orbits
        elif key == glfw.KEY_H:
            o.help = not o.help
        elif key == glfw.KEY_P:
            self.time_idx = 3 if self.time_idx == 0 else 0
            self._say("Tempo " + ("pausado" if self.time_idx == 0 else "retomado"))
        elif key == glfw.KEY_RIGHT_BRACKET or key == glfw.KEY_KP_ADD:
            self.time_idx = min(self.time_idx + 1, len(TIME_SCALES) - 1)
        elif key == glfw.KEY_LEFT_BRACKET or key == glfw.KEY_KP_SUBTRACT:
            self.time_idx = max(self.time_idx - 1, 0)
        elif key == glfw.KEY_F1:
            o.hud = not o.hud
        elif key == glfw.KEY_F2:
            o.shading = (o.shading + 1) % 3
            self._say("Sombreamento: " + SHADING_NAMES[o.shading])
        elif key == glfw.KEY_F3:
            o.normal_map = not o.normal_map
            self._say("Normal mapping " + ("ON" if o.normal_map else "OFF"))
        elif key == glfw.KEY_F4:
            o.shadows = not o.shadows
            self._say("Sombras " + ("ON" if o.shadows else "OFF"))
        elif key == glfw.KEY_F5:
            o.textures = not o.textures
            self._say("Texturas " + ("ON" if o.textures else "OFF"))
        elif key == glfw.KEY_F6:
            o.atmosphere = not o.atmosphere
            self._say("Atmosferas " + ("ON" if o.atmosphere else "OFF"))
        elif key == glfw.KEY_F7:
            o.bloom = not o.bloom
            self._say("Bloom " + ("ON" if o.bloom else "OFF"))
        elif key == glfw.KEY_F8:
            o.wireframe = not o.wireframe
        elif key == glfw.KEY_F9:
            o.tonemap = not o.tonemap
            self._say("Tone mapping " + ("ACES" if o.tonemap else "OFF (clamp)"))
        elif key == glfw.KEY_F12:
            self.audio.play("click")
            self.screenshot()

    def _on_mouse(self, win, x, y):
        if self.last_mouse is not None:
            self.mouse_dx += x - self.last_mouse[0]
            self.mouse_dy += y - self.last_mouse[1]
        self.last_mouse = (x, y)

    def _on_scroll(self, win, dx, dy):
        a = self.astro
        a.speed_idx = int(np.clip(a.speed_idx + (1 if dy > 0 else -1), 0, len(SPEED_MULTS) - 1))
        self.audio.play("up" if dy > 0 else "down")
        self._say(f"Multiplicador de velocidade: {a.mult():g}x  (cruzeiro agora: {_fmt_speed(a.speed())})")

    def _on_button(self, win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and not self.captured:
            self._capture_mouse(True)
        if button == glfw.MOUSE_BUTTON_RIGHT:
            self.rmb = action == glfw.PRESS

    # ------------------------------------------------------------- simulação
    def place_near(self, key, dist, angle, elev):
        b = self.system.by_key.get(key, self.system.by_key["earth"])
        sun_dir = gm.normalize(self.system.sun.pos - b.pos) if b is not self.system.sun else np.array([1.0, 0, 0])
        d = gm.rot_y(math.radians(angle))[:3, :3] @ sun_dir
        d = gm.normalize(d + np.array([0, math.tan(math.radians(elev)), 0]))
        self.astro.pos = b.pos + d * b.radius * dist
        self.astro.q = gm.look_rotation(b.pos - self.astro.pos)
        self.astro.vel[:] = 0

    def start_autopilot(self, body):
        sysm = self.system
        a = self.astro
        rel0 = a.pos - body.pos
        if body is sysm.sun:
            d = gm.normalize(rel0)
            dist = body.radius * 3.5
        else:
            sun_dir = gm.normalize(sysm.sun.pos - body.pos)
            d = gm.normalize(sun_dir * 0.7 + gm.normalize(rel0) * 0.3 + np.array([0, 0.25, 0]))
            dist = body.radius * (5.5 if body.d.rings else 3.3) + (body.radius * 0.1 if body.d.atmo else 0)
        rel1 = d * dist
        mid = (rel0 + rel1) / 2
        sun_rel = sysm.sun.pos - body.pos
        mid = _push_out(mid, sun_rel, sysm.sun.radius * 5, rel1 - rel0)
        mid = _push_out(mid, np.zeros(3), body.radius * 3, rel1 - rel0)
        span = np.linalg.norm(rel1 - rel0)
        dur = float(np.clip(1.5 + 0.9 * math.log10(1 + span / max(body.radius, 0.3)), 2.0, 5.0))
        a.autopilot = dict(body=body, p0=rel0, p1=mid, p2=rel1, t=0.0, dur=dur)
        self.audio.play("warp")
        self._say(f"Piloto automático: {body.d.name}")

    def _update_autopilot(self, dt):
        a = self.astro
        ap = a.autopilot
        ap["t"] += dt / ap["dur"]
        s = min(ap["t"], 1.0)
        s = s * s * s * (s * (s * 6 - 15) + 10)       # smootherstep (acelera e freia suave)
        b = ap["body"]
        u = 1 - s
        rel = u * u * ap["p0"] + 2 * u * s * ap["p1"] + s * s * ap["p2"]   # Bézier quadrática
        new = b.pos + rel
        a.vel = (new - a.pos) / max(dt, 1e-4)
        a.pos = new
        qt = gm.look_rotation(b.pos - a.pos, a.rot[:, 1])
        a.q = gm.q_normalize(gm.q_slerp(a.q, qt, gm.smooth_factor(3.0, dt)))
        if ap["t"] >= 1.0:
            a.autopilot = None
            a.vel[:] = 0
            self._say(f"Chegamos: {b.d.name}")
            self.audio.play("arrive")

    def update(self, dt):
        sysm, a, k = self.system, self.astro, self.keys
        old_t = self.sim_time
        self.sim_time += TIME_SCALES[self.time_idx] * dt
        sysm.update(self.sim_time)
        # referencial: o astronauta é carregado pelo corpo em cuja SOI ele está
        fb = sysm.frame_body(a.pos)
        if fb is not None:
            a.pos += fb.pos - fb.prev_pos
            dist = np.linalg.norm(a.pos - fb.pos)
            dspin = fb.spin - fb.prev_spin
            if dist < fb.radius * 1.3 and abs(dspin) < 0.5:   # co-rotação perto da superfície
                axis = fb.rot[:, 1]
                R = gm.mat4_from_mat3(gm.q_to_mat3(gm.q_axis_angle(axis, dspin)))[:3, :3]
                a.pos = fb.pos + R @ (a.pos - fb.pos)
                a.q = gm.q_normalize(gm.q_mul(gm.q_axis_angle(axis, dspin), a.q))
        # entrada
        sens = 0.0022
        if self.captured and not self.rmb:
            a.rotate(-self.mouse_dx * sens, -self.mouse_dy * sens, 0)
        elif self.rmb:
            self.cam_orbit[0] -= self.mouse_dx * 0.005
            self.cam_orbit[1] = float(np.clip(self.cam_orbit[1] - self.mouse_dy * 0.005, -1.4, 1.4))
        self.mouse_dx = self.mouse_dy = 0.0
        roll = (glfw.KEY_Q in k) - (glfw.KEY_E in k)
        if roll:
            a.rotate(0, 0, roll * 1.5 * dt)
        thrust = np.array([(glfw.KEY_D in k) - (glfw.KEY_A in k),
                           (glfw.KEY_SPACE in k) - (glfw.KEY_LEFT_CONTROL in k or glfw.KEY_RIGHT_CONTROL in k),
                           (glfw.KEY_S in k) - (glfw.KEY_W in k)], float)
        if glfw.KEY_Z in k:
            self.cam_zoom = max(0.4, self.cam_zoom * (1 - dt))
        if glfw.KEY_X in k:
            self.cam_zoom = min(8.0, self.cam_zoom * (1 + dt))
        if a.autopilot and np.any(thrust):
            a.autopilot = None
            self._say("Piloto automático cancelado")
        old = a.pos.copy()
        touch = False
        impact = 0.0
        if a.autopilot:
            self._update_autopilot(dt)
            a.fly += (1.0 - a.fly) * gm.smooth_factor(2.0, dt)
            a.anim_t += dt
            a.pose = pose_blend(a.fly, a.anim_t)
        else:
            boost = glfw.KEY_LEFT_SHIFT in k or glfw.KEY_RIGHT_SHIFT in k
            _, alt = sysm.nearest(a.pos)
            a.update(dt, thrust, boost, alt)
        # colisão CONTÍNUA (segmento x esfera): em alta velocidade um teste só da
        # posição final deixaria o astronauta "atravessar" o planeta (tunelamento)
        for b in sysm.bodies:
            lim = b.radius * 1.03 if b is sysm.sun else b.radius + a.height() * 0.55
            seg = a.pos - old
            L = np.linalg.norm(seg)
            if L > 1e-12 and np.linalg.norm(old - b.pos) >= lim:
                dvec = seg / L
                oc = old - b.pos
                bb = np.dot(oc, dvec)
                disc = bb * bb - (np.dot(oc, oc) - lim * lim)
                if disc > 0:
                    t = -bb - math.sqrt(disc)
                    if 0 <= t <= L:
                        a.pos = old + dvec * t
            v = a.pos - b.pos
            d = np.linalg.norm(v)
            if d < lim:
                n = v / d
                a.pos = b.pos + n * lim
                vn = np.dot(a.vel, n)
                touch = touch or b is not sysm.sun
                if vn < 0:
                    impact = max(impact, -vn / max(a.speed(), 1e-6))
                    a.vel -= vn * n
        # exposição automática: o "olho" se adapta à luz solar local
        self.exposure += (self._exposure_target() - self.exposure) * gm.smooth_factor(1.5, dt)
        if touch and not self.touching and impact > 0.2:
            self.audio.play("thump", min(1.0, impact))
        self.touching = touch
        self._update_camera(dt)
        self._update_audio()
        self.speech.update(dt, sysm, a.pos)
        self.message_t = max(0.0, self.message_t - dt)

    def _update_audio(self):
        """Publica para a thread de áudio o volume/pan de cada fonte sonora."""
        au = self.audio
        if not au.ok:
            return
        sysm, a = self.system, self.astro
        right = self.cam_rot[:, 0]
        voices = {}
        for b in sysm.bodies:
            v = b.pos - self.cam_pos
            d = np.linalg.norm(v)
            alt = max(d - b.radius, 0.0)
            g = 1.0 / (1.0 + (alt / (2.5 * b.radius)) ** 2)        # cresce ao se aproximar
            pan = float(np.clip(np.dot(v / d, right), -1, 1)) * 0.8
            if b is sysm.sun:
                sun = (g * 0.7, pan)
            elif g > 0.003:
                voices[b.key] = (g * 0.3, pan)
        nb, alt = sysm.nearest(a.pos)
        spd = np.linalg.norm(a.vel)
        rel = spd / max(alt, 1e-4)                                   # velocidade aparente (altitudes/s)
        speed01 = float(np.clip(math.log10(1 + rel * 4) / 1.3, 0, 1))
        wind = 0.0
        if nb.d.atmo:
            h = nb.radius * (nb.d.atmo["ra"] - 1)
            dens = math.exp(-alt / (h * 0.6))
            wind = dens * (0.25 + 0.75 * min(1.0, spd / (nb.radius * 0.01)))
        sat = sysm.by_key["saturn"]
        rv = a.pos - sat.pos
        nrm = gm.rot_x(math.radians(sat.d.tilt))[:3, :3] @ np.array([0, 1.0, 0])
        hgt = abs(np.dot(rv, nrm))
        rr = np.linalg.norm(rv - nrm * np.dot(rv, nrm)) / sat.radius
        rings = math.exp(-hgt / (sat.radius * 0.25)) if RING_INNER * 0.95 < rr < RING_OUTER * 1.05 else 0.0
        sp = a.pos - sysm.sun.pos
        rxz = math.hypot(sp[0], sp[2])
        belt = math.exp(-max(0.0, abs(rxz - 660) - 60) / 15) * math.exp(-max(0.0, abs(sp[1]) - 15) / 15)
        au.set_state(voices=voices, sun=sun, speed=speed01, wind=wind, rings=rings, belt=belt)

    def _exposure_target(self):
        """Adaptação do olho: exposição ~ E^-0.7 (compensa parcialmente o inverso do quadrado)."""
        E = self.system.irradiance(self.astro.pos)
        return float(np.clip(1.6 * E ** -0.7, 0.05, 40.0))

    def _update_camera(self, dt, snap=False):
        a = self.astro
        k = 1.0 if snap else gm.smooth_factor(7.0, dt)
        self.cam_q = gm.q_normalize(gm.q_slerp(self.cam_q, a.q, k))
        orbit = gm.q_mul(gm.q_axis_angle((0, 1, 0), self.cam_orbit[0]), gm.q_axis_angle((1, 0, 0), self.cam_orbit[1]))
        if self.cam_mode == 1:
            R = gm.q_to_mat3(a.q)
            self.cam_rot = R @ gm.q_to_mat3(orbit)
            self.cam_pos = a.pos + R @ (head_offset(a.pose) * ASTRO_SCALE)
        else:
            qc = gm.q_mul(self.cam_q, orbit)
            self.cam_rot = gm.q_to_mat3(qc)
            off = np.array([0.0, 1.1, 4.6]) * self.cam_zoom * ASTRO_SCALE
            self.cam_pos = a.pos + self.cam_rot @ off
            # câmera nunca entra num planeta
            for b in self.system.bodies:
                v = self.cam_pos - b.pos
                d = np.linalg.norm(v)
                lim = b.radius + ASTRO_SCALE * 0.3
                if d < lim:
                    self.cam_pos = b.pos + v / d * lim

    # ---------------------------------------------------------- renderização
    def _rel(self, p):
        return p - self.cam_pos

    def _set_common(self, s):
        s.use()
        s.f("uFcoef", self.fcoef)
        s.m4("uViewProj", self.vp)
        s.v3("uSunPos", self._rel(self.system.sun.pos))
        s.f("uSunRadius", self.system.sun.radius)
        s.v3("uSunColor", SUN_COLOR)
        s.i("uShadowsOn", int(self.opt.shadows))
        s.i("uRingTex", 5)
        s.i("uShadowMap", 6)
        s.m4("uLightVP", self.light_vp)
        s.i("uShadowMapOn", int(self.shadow_active))
        s.f("uShadowTexel", 1.0 / SHADOW_RES)
        s.f("uShadowNormalOffset", self.shadow_extent / SHADOW_RES * 3.0)
        s.f("uShadowBias", self.shadow_extent * 2.0 / SHADOW_RES / 2.0 * 1.5)
        s.i("uRingOn", 0)
        s.i("uShadingMode", self.opt.shading)
        s.i("uUseTextures", int(self.opt.textures))
        s.i("uNormalMapOn", int(self.opt.normal_map))
        a = self.astro
        s.i("uSpotOn", int(a.lantern))
        if a.lantern:
            R = a.rot
            s.v3("uSpotPos", self._rel(a.pos) + R @ (head_offset(a.pose) * ASTRO_SCALE))
            fwd = R @ (gm.rot_x(a.pose["pitch"] + a.pose["head"] + a.pose["waist"])[:3, :3] @ np.array([0, 0, -1.0]))
            s.v3("uSpotDir", gm.normalize(fwd))
            s.v2("uSpotCos", (math.cos(math.radians(14)), math.cos(math.radians(24))))
            s.v3("uSpotColor", np.array([1.0, 0.95, 0.85]) * 0.0015)
            s.f("uSpotRange", 0.25)

    def _set_occluders(self, s, p, exclude=None):
        occ = self.system.occluders_for(p, exclude=exclude, n=6)
        arr = [list(self._rel(b.pos)) + [b.radius] for b in occ]
        s.i("uOccCount", len(arr))
        if arr:
            s.v4a("uOcc", arr)

    def _set_ring(self, s, on):
        sat = self.system.by_key["saturn"]
        s.i("uRingOn", int(on))
        if on:
            s.v3("uRingCenter", self._rel(sat.pos))
            s.v3("uRingNormal", gm.rot_x(math.radians(sat.d.tilt))[:3, :3] @ np.array([0, 1.0, 0]))
            s.v2("uRingRadii", (RING_INNER * sat.radius, RING_OUTER * sat.radius))

    def _shine(self, body, p):
        """Luz refletida pelo corpo `body` chegando no ponto p (earthshine/planetshine)."""
        if body is None or body is self.system.sun:
            return np.array([0, 1.0, 0]), np.zeros(3)
        to_b = body.pos - p
        d = np.linalg.norm(to_b)
        sun_dir = gm.normalize(self.system.sun.pos - body.pos)
        phase = 0.5 * (1 + np.dot(sun_dir, -to_b / d))          # fração iluminada vista de p
        col = self.avg_colors.get(body.key, np.array(body.d.color)) * self.system.irradiance(body.pos)
        return to_b / d, col * phase * (body.radius / d) ** 2 * 1.5

    def _visible(self, center_rel, radius):
        """Frustum culling de esfera contra os 6 planos extraídos da matriz VP."""
        p = np.append(center_rel, 1.0)
        for pl in self.frustum:
            if pl[:3] @ p[:3] + pl[3] < -radius * np.linalg.norm(pl[:3]):
                return False
        return True

    def render(self):
        w, h = self.fbo_size
        sysm, a, o = self.system, self.astro, self.opt
        self.fcoef = 2.0 / math.log2(FAR + 1.0)
        aspect = w / max(h, 1)
        self.proj = gm.perspective(self.args.fov, aspect, NEAR, FAR)
        self.view = gm.mat4_from_mat3(self.cam_rot.T)                 # só rotação (câmera na origem)
        self.vp = self.proj @ self.view
        vp = self.vp
        self.frustum = [vp[3] + vp[0], vp[3] - vp[0], vp[3] + vp[1], vp[3] - vp[1], vp[3] + vp[2], vp[3] - vp[2]]
        show_astro = self.cam_mode == 0
        self.sky_comp = (1.6 / self.exposure) ** 0.6     # céu não "acende" tanto com a adaptação do olho
        parts = build_parts(a.pose)
        root = a.model_matrix(self.cam_pos)

        # ---------------- 1) SHADOW PASS ----------------
        self.shadow_extent = a.height() * 0.75
        self.shadow_active = o.shadows
        Ld = gm.normalize(a.pos - sysm.sun.pos)
        center = self._rel(a.pos)
        up = np.array([0, 1.0, 0]) if abs(Ld[1]) < 0.9 else np.array([1.0, 0, 0])
        depth_range = 2.0
        lv = gm.look_at(center - Ld * self.shadow_extent * 4, center, up)
        lp = gm.ortho(-self.shadow_extent, self.shadow_extent, -self.shadow_extent, self.shadow_extent, 0.0, depth_range)
        self.light_vp = lp @ lv
        if self.shadow_active:
            glBindFramebuffer(GL_FRAMEBUFFER, self.shadow_fbo)
            glViewport(0, 0, SHADOW_RES, SHADOW_RES)
            glClear(GL_DEPTH_BUFFER_BIT)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(1.5, 2.0)
            glDisable(GL_CULL_FACE)
            s = self.sh["depth"].use()
            s.m4("uLightVP", self.light_vp)
            for mk, mm, mat in parts:
                s.m4("uModel", root @ mm)
                self.astro_meshes[mk].draw()
            glDisable(GL_POLYGON_OFFSET_FILL)
            glEnable(GL_CULL_FACE)
        bind(6, self.shadow_tex)
        bind(5, self.tex[("rings", "rings")])

        # ---------------- 2) PASSO HDR ----------------
        glBindFramebuffer(GL_FRAMEBUFFER, self.msaa_fbo)
        glViewport(0, 0, w, h)
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # céu de fundo + estrelas (sem profundidade)
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        s = self.sh["sky"].use()
        s.m4("uInvViewProj", np.linalg.inv(self.vp))
        s.i("uSky", 0)
        s.f("uSkyIntensity", 0.12 * self.sky_comp)
        bind(0, self.tex[("sky", "sky")])
        glBindVertexArray(self.empty_vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)
        s = self.sh["stars"].use()
        s.m4("uViewProj", self.vp)
        s.f("uPointScale", max(1.0, h / 1080.0))
        s.f("uStarIntensity", self.sky_comp)
        glBindVertexArray(self.star_vao)
        glDrawArrays(GL_POINTS, 0, self.star_count)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)

        if o.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Sol
        sun = sysm.sun
        s = self.sh["sun"]
        self._set_common(s)
        Msun = gm.translate(self._rel(sun.pos)) @ gm.mat4_from_mat3(sun.rot) @ gm.scale(sun.radius)
        s.m4("uModel", Msun)
        s.m3("uNormalMat", sun.rot)
        s.f("uTime", glfw.get_time())
        s.f("uEmission", SUN_EMISSION)
        has_sun_tex = ("sun", "albedo") in self.tex
        s.i("uHasTex", int(has_sun_tex and o.textures))
        s.i("uAlbedo", 0)
        if has_sun_tex:
            bind(0, self.tex[("sun", "albedo")])
        if self._visible(self._rel(sun.pos), sun.radius):
            self.sphere_hi.draw()

        # planetas e luas (opacos)
        s = self.sh["planet"]
        self._set_common(s)
        for i, nm in enumerate(("uAlbedo", "uNormalTex", "uSpecTex", "uNightTex", "uCloudTex")):
            s.i(nm, i)
        cloud_drift = self.sim_time * 0.6
        for b in sysm.bodies:
            if b is sun or not self._visible(self._rel(b.pos), b.radius * 1.1):
                continue
            k = b.key
            Mb = gm.translate(self._rel(b.pos)) @ gm.mat4_from_mat3(b.rot) @ gm.scale(b.radius)
            s.m4("uModel", Mb)
            s.m3("uNormalMat", b.rot)
            bind(0, self.tex.get((k, "albedo"), self.white))
            has_n = (k, "normal") in self.tex
            bind(1, self.tex.get((k, "normal"), self.flat_normal))
            has_s = (k, "spec") in self.tex
            bind(2, self.tex.get((k, "spec"), self.white))
            has_night = (k, "night") in self.tex
            bind(3, self.tex.get((k, "night"), self.white))
            has_c = (k, "clouds") in self.tex
            bind(4, self.tex.get((k, "clouds"), self.white))
            s.i("uHasNormal", int(has_n))
            s.i("uHasSpec", int(has_s))
            s.i("uHasNight", int(has_night))
            s.i("uHasCloudShadow", int(has_c and o.textures))
            s.f("uCloudOffset", -cloud_drift / (2 * math.pi))
            s.i("uIsCloud", 0)
            s.f("uNormalStrength", b.d.normal_strength)
            s.f("uRoughness", b.d.roughness)
            s.f("uSpecStrength", b.d.spec)
            s.f("uShininess", b.d.shininess)
            s.f("uIrradiance", sysm.irradiance(b.pos))
            s.f("uAmbient", 0.0008)
            s.f("uNightIntensity", 0.9)
            s.v3("uFlatColor", (0.6, 0.6, 0.6))
            det = 1.0 if b.d.roughness > 0 else (0.7 if k == "earth" else 0.0)
            texw = self.arrays[(k, "albedo")].shape[1] if (k, "albedo") in self.arrays else 1024
            s.f("uDetail", det)
            s.f("uDetailFreq", b.radius / 0.02)
            s.f("uDetailFade", 2 * math.pi * b.radius / texw * 600)
            sd, sc = self._shine(b.parent if b.parent is not sun else None, b.pos)
            s.v3("uShineDir", sd)
            s.v3("uShineColor", sc)
            self._set_occluders(s, b.pos, exclude=b)
            self._set_ring(s, b.d.rings and o.textures)
            self.sphere_hi.draw()

        # cinturão de asteroides (instancing)
        s = self.sh["asteroid"]
        self._set_common(s)
        s.v3("uCenter", self._rel(sun.pos))
        s.f("uTime", self.sim_time)
        s.f("uIrradiance", (320 / 660.0) ** 2)
        s.f("uAmbient", 0.0008)
        for m in self.asteroids:
            m.draw_instanced()

        # astronauta (grafo de cena hierárquico)
        if show_astro:
            s = self.sh["astro"]
            self._set_common(s)
            s.i("uAlbedo", 0)
            s.i("uSky", 7)
            bind(0, self.tex[("suit", "suit")])
            bind(7, self.tex[("sky", "sky")])
            s.f("uSkyIntensity", 0.12 * self.sky_comp)
            s.f("uIrradiance", sysm.irradiance(a.pos))
            s.f("uAmbient", 0.001)
            nb, _ = sysm.nearest(a.pos, include_sun=False)
            sd, sc = self._shine(nb, a.pos)
            s.v3("uShineDir", sd)
            s.v3("uShineColor", sc)
            self._set_occluders(s, a.pos)
            self._set_ring(s, nb is not None and nb.d.rings)
            glDisable(GL_CULL_FACE)
            for mk, mm, mat in parts:
                base, use_tex, f0, shin, spec, mirror = MATERIALS[mat]
                Mw = root @ mm
                s.m4("uModel", Mw)
                s.m3("uNormalMat", gm.normal_matrix(Mw))
                s.v3("uBaseColor", base)
                s.i("uUseTex", use_tex)
                s.v3("uF0", f0)
                s.f("uShininess", shin)
                s.f("uSpecStrength", spec)
                s.f("uMirror", mirror)
                emis = (0, 0, 0)
                if mat == "lamp":
                    emis = (6.0, 5.6, 4.6) if a.lantern else (0.0, 0.0, 0.0)
                if mat == "red":
                    emis = (0.6, 0.02, 0.0) if int(glfw.get_time() * 1.5) % 2 == 0 else (0.0, 0.0, 0.0)
                s.v3("uEmissive", np.array(emis) / max(self.exposure, 1e-3) * 1.6)
                self.astro_meshes[mk].draw()
            glEnable(GL_CULL_FACE)

        if o.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # linhas das órbitas
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        if o.orbits:
            s = self.sh["line"]
            self._set_common(s)
            for b in sysm.bodies:
                if b.parent is None:
                    continue
                if b.key not in self.orbit_meshes:
                    self.orbit_meshes[b.key] = M.LineMesh(M.circle_points(b.d.orbit, 720))
                s.m4("uModel", gm.translate(self._rel(b.parent.pos)) @ gm.mat4_from_mat3(b.frame))
                c = (0.35, 0.55, 1.0) if b.parent is sun else (0.7, 0.7, 0.7)
                s.v4("uColor", (*(np.array(c) * 0.25 / self.exposure), 0.6))
                self.orbit_meshes[b.key].draw()

        # transparentes: de trás para frente
        trans = sorted([b for b in sysm.bodies if b is not sun], key=lambda b: -np.linalg.norm(self._rel(b.pos)))
        for b in trans:
            rel = self._rel(b.pos)
            if not self._visible(rel, b.radius * 2.5):
                continue
            if b.d.clouds and o.textures:
                s = self.sh["planet"]
                self._set_common(s)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                rot = gm.rot_x(math.radians(b.d.tilt)) @ gm.rot_y(b.spin + cloud_drift)
                Mc = gm.translate(rel) @ rot @ gm.scale(b.radius * 1.006)
                s.m4("uModel", Mc)
                s.m3("uNormalMat", rot[:3, :3])
                bind(0, self.tex[(b.key, "clouds")])
                s.i("uIsCloud", 1)
                for nm in ("uHasCloudShadow", "uHasNormal", "uHasSpec", "uHasNight"):
                    s.i(nm, 0)
                s.f("uRoughness", 0.0)
                s.f("uSpecStrength", 0.0)
                s.f("uDetail", 0.0)
                s.f("uIrradiance", sysm.irradiance(b.pos))
                s.f("uAmbient", 0.0006)
                self._set_occluders(s, b.pos, exclude=b)
                self.sphere_hi.draw()
                s.i("uIsCloud", 0)
            if b.d.atmo and o.atmosphere:
                at = b.d.atmo
                s = self.sh["atmo"]
                self._set_common(s)
                glBlendFunc(GL_ONE, GL_ONE)
                rr = b.radius * at["ra"] * 1.01
                inside = np.linalg.norm(rel) < rr
                glCullFace(GL_FRONT if inside else GL_BACK)
                s.m4("uModel", gm.translate(rel) @ gm.scale(rr))
                s.m3("uNormalMat", np.eye(3))
                s.v3("uCenter", rel)
                s.f("uRadius", b.radius)
                s.f("uAtmoRadius", at["ra"])
                s.v3("uBetaR", at["betaR"])
                s.v3("uBetaM", at["betaM"])
                s.f("uHR", at["hr"])
                s.f("uHM", at["hm"])
                s.f("uG", at["g"])
                s.f("uSunIntensity", sysm.irradiance(b.pos) * ATMO_K * at["k"])
                self.sphere_mid.draw()
                glCullFace(GL_BACK)
            if b.d.rings:
                s = self.sh["ring"]
                self._set_common(s)
                glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)          # alpha pré-multiplicado
                glDisable(GL_CULL_FACE)
                rot = gm.rot_x(math.radians(b.d.tilt))
                s.m4("uModel", gm.translate(rel) @ rot @ gm.scale(b.radius))
                s.m3("uNormalMat", rot[:3, :3])
                s.f("uIrradiance", sysm.irradiance(b.pos))
                s.f("uAmbient", 0.0005)
                s.i("uOccCount", 1)
                s.v4a("uOcc", [list(rel) + [b.radius]])
                self.ring_mesh.draw()
                glEnable(GL_CULL_FACE)

        # halo do Sol (aditivo)
        srel = self._rel(sun.pos)
        dsun = np.linalg.norm(srel)
        if self._visible(srel, sun.radius * 8):
            glBlendFunc(GL_ONE, GL_ONE)
            s = self.sh["glow"]
            self._set_common(s)
            size = sun.radius * 7
            s.v3("uCenter", srel - srel / dsun * sun.radius)
            s.v3("uRight", self.cam_rot[:, 0])
            s.v3("uUp", self.cam_rot[:, 1])
            s.f("uSize", size)
            s.f("uInner", sun.radius / size)
            s.f("uIntensity", 2.5)
            s.f("uTime", glfw.get_time())
            glDisable(GL_CULL_FACE)
            glBindVertexArray(self.empty_vao)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            glEnable(GL_CULL_FACE)
        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)

        # ---------------- 3) RESOLVE + BLOOM + TONE MAPPING ----------------
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.msaa_fbo)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.hdr_fbo)
        glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST)
        glDisable(GL_DEPTH_TEST)
        glBindVertexArray(self.empty_vao)
        if o.bloom:
            s = self.sh["down"].use()
            s.i("uSrc", 0)
            src, sw, shh = self.hdr_tex, w, h
            for i, (t, f, bw, bh) in enumerate(self.bloom):
                glBindFramebuffer(GL_FRAMEBUFFER, f)
                glViewport(0, 0, bw, bh)
                s.v2("uTexel", (1.0 / sw, 1.0 / shh))
                s.i("uFirst", int(i == 0))
                bind(0, src)
                glDrawArrays(GL_TRIANGLES, 0, 3)
                src, sw, shh = t, bw, bh
            s = self.sh["up"].use()
            s.i("uSrc", 0)
            s.f("uRadius", 1.0)
            glEnable(GL_BLEND)
            glBlendFunc(GL_ONE, GL_ONE)
            for i in range(len(self.bloom) - 1, 0, -1):
                t, f, bw, bh = self.bloom[i]
                _, f2, bw2, bh2 = self.bloom[i - 1]
                glBindFramebuffer(GL_FRAMEBUFFER, f2)
                glViewport(0, 0, bw2, bh2)
                s.v2("uTexel", (1.0 / bw, 1.0 / bh))
                bind(0, t)
                glDrawArrays(GL_TRIANGLES, 0, 3)
            glDisable(GL_BLEND)
        glBindFramebuffer(GL_FRAMEBUFFER, self.ldr_fbo)
        glViewport(0, 0, w, h)
        s = self.sh["comp"].use()
        s.i("uHDR", 0)
        s.i("uBloom", 1)
        bind(0, self.hdr_tex)
        bind(1, self.bloom[0][0])
        s.f("uExposure", self.exposure)
        s.f("uBloomStrength", 0.035)
        s.i("uBloomOn", int(o.bloom))
        s.i("uTonemapOn", int(o.tonemap))
        glDrawArrays(GL_TRIANGLES, 0, 3)

        # ---------------- 4) HUD ----------------
        self.draw_hud(w, h)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.ldr_fbo)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glEnable(GL_DEPTH_TEST)

    # ------------------------------------------------------------------- HUD
    def draw_hud(self, w, h):
        t, o, a, sysm = self.text, self.opt, self.astro, self.system
        if o.labels and o.hud:
            for b in sysm.bodies:
                rel = self._rel(b.pos)
                c = self.vp @ np.append(rel, 1.0)
                if c[3] <= 0:
                    continue
                ndc = c[:3] / c[3]
                if abs(ndc[0]) > 1 or abs(ndc[1]) > 1:
                    continue
                dist = np.linalg.norm(rel)
                ang = b.radius / dist
                px_r = ang / math.tan(math.radians(self.args.fov) / 2) * h / 2
                if px_r > h * 0.25:
                    continue
                x = (ndc[0] * 0.5 + 0.5) * w
                y = (1 - (ndc[1] * 0.5 + 0.5)) * h
                col = (1.0, 0.85, 0.5, 0.85) if b.parent is None or b.parent is sysm.sun else (0.75, 0.8, 0.9, 0.7)
                label = f"{b.d.name}  {_fmt_dist(dist - b.radius)}"
                t.text(label, x + px_r + 6, y - t.ch / 2, col)
        if not o.hud:
            t.flush(w, h)
            return
        nb, nd = sysm.nearest(a.pos)
        spd = np.linalg.norm(a.vel)
        lines = [f"ASTRONAUTA CÓSMICO   {self.fps:5.1f} FPS",
                 f"Velocidade: {_fmt_speed(spd)}  (cruzeiro {_fmt_speed(a.speed())}, {a.mult():g}x)",
                 f"Tempo: {_fmt_time(TIME_SCALES[self.time_idx])}   Dia {self.sim_time:9.2f}",
                 f"Sombreamento: {SHADING_NAMES[o.shading]}",
                 "H = ajuda"]
        t.rect(8, 8, t.width("\n".join(lines)) + 16, len(lines) * t.ch + 10, (0, 0, 0, 0.35))
        y = 13
        for i, l in enumerate(lines):
            y = t.text(l, 16, y, (1, 1, 1, 0.95) if i == 0 else (0.85, 0.9, 1.0, 0.9))
        # informações do corpo mais próximo
        info = [f"{nb.d.name.upper()}  —  altitude {_fmt_dist(max(nd, 0))}"] + nb.d.info
        bw = t.width("\n".join(info)) + 16
        bh = len(info) * t.ch + 10
        t.rect(8, h - bh - 8, bw, bh, (0, 0, 0, 0.35))
        y = h - bh - 3
        for i, l in enumerate(info):
            y = t.text(l, 16, y, (1.0, 0.85, 0.5, 1) if i == 0 else (0.9, 0.9, 0.9, 0.9))
        # estado dos efeitos
        flags = [("F2", SHADING_NAMES[o.shading].split()[0]), ("F3 Normal map", o.normal_map), ("F4 Sombras", o.shadows),
                 ("F5 Texturas", o.textures), ("F6 Atmosfera", o.atmosphere), ("F7 Bloom", o.bloom),
                 ("F8 Wireframe", o.wireframe), ("F9 Tone map", o.tonemap), ("L Lanterna", a.lantern), ("O Órbitas", o.orbits),
                 ("M Som", self.audio.ok and not self.audio.muted)]
        y = 13
        for name, v in flags:
            s = f"{name}: {v}" if isinstance(v, str) else f"{name}: {'ON' if v else 'off'}"
            col = (0.6, 1.0, 0.6, 0.9) if v is True or isinstance(v, str) else (0.7, 0.7, 0.7, 0.7)
            y = t.text(s, w - t.width(s) - 16, y, col)
        sp = self.speech
        if sp.sub_t > 0 and sp.subtitle:
            line = f"Astronauta: “{sp.subtitle}”"
            alpha = min(1.0, sp.sub_t * 2)
            tw = t.width(line)
            t.rect((w - tw) / 2 - 14, h * 0.80 - 8, tw + 28, t.ch + 16, (0, 0, 0, 0.55 * alpha))
            t.text(line, (w - tw) / 2, h * 0.80, (1.0, 0.95, 0.75, alpha))
        if a.autopilot:
            s = f">>> Piloto automático para {a.autopilot['body'].d.name} <<<"
            t.text(s, (w - t.width(s)) / 2, h * 0.18, (0.6, 0.9, 1.0, 1))
        if self.message_t > 0:
            t.text(self.message, (w - t.width(self.message)) / 2, h * 0.12, (1, 1, 1, min(1.0, self.message_t)))
        if not self.captured and not self.args.shot:
            s = "Clique na janela para controlar o astronauta (ESC solta o mouse)"
            t.text(s, (w - t.width(s)) / 2, h / 2, (1, 1, 0.6, 1))
        if o.help:
            self._draw_help(w, h)
        t.flush(w, h)

    def _draw_help(self, w, h):
        t = self.text
        lines = [
            "CONTROLES",
            "",
            "Mouse ............ olhar / guiar o voo     Botão direito + arrastar: orbitar a câmera",
            "W/S A/D .......... frente/trás, laterais   Espaço/Ctrl: subir/descer   Q/E: girar (roll)",
            "Shift ............ turbo (x6)              Roda do mouse: multiplicador de velocidade",
            "                   (a velocidade se ajusta sozinha à altitude: rápida no vazio, lenta perto)",
            "0..9 ............. viajar: 0 Sol 1 Mercúrio 2 Vênus 3 Terra 4 Marte 5 Júpiter 6 Saturno",
            "                   7 Urano 8 Netuno 9 Lua  |  Tab/Shift+Tab: escolher alvo, Enter: viajar",
            "C ................ 1ª / 3ª pessoa          Z/X: aproximar/afastar câmera   V: resetar",
            "L ................ lanterna do capacete (luz spot)",
            "[ ] / P .......... acelerar/desacelerar o tempo / pausar",
            "R / O ............ rótulos / linhas das órbitas      M: som liga/desliga",
            "",
            "F1 HUD   F2 Phong/Gouraud/Flat   F3 normal map   F4 sombras   F5 texturas",
            "F6 atmosfera   F7 bloom   F8 wireframe   F9 tone mapping   F12 captura de tela",
            "ESC .............. soltar o mouse / sair",
        ]
        bw = t.width("\n".join(lines)) + 40
        bh = len(lines) * t.ch + 30
        x0, y0 = (w - bw) / 2, (h - bh) / 2
        t.rect(x0, y0, bw, bh, (0.02, 0.03, 0.06, 0.82))
        y = y0 + 15
        for i, l in enumerate(lines):
            y = t.text(l, x0 + 20, y, (1.0, 0.85, 0.5, 1) if i == 0 else (0.9, 0.92, 1.0, 1))

    def screenshot(self, path=None):
        w, h = self.fbo_size
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.ldr_fbo)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        from PIL import Image
        img = Image.frombuffer("RGB", (w, h), data, "raw", "RGB", 0, -1)
        if path is None:
            os.makedirs("screenshots", exist_ok=True)
            path = os.path.join("screenshots", time.strftime("captura_%Y%m%d_%H%M%S.png"))
        img.save(path)
        self._say(f"Captura salva: {path}")
        print("Captura salva:", path)

    # ------------------------------------------------------------------ loop
    def run(self):
        last = glfw.get_time()
        frames = 0
        while not glfw.window_should_close(self.win):
            now = glfw.get_time()
            dt = min(now - last, 0.1)
            if self.args.shot:
                dt = 1 / 60
            last = now
            glfw.poll_events()
            self.update(dt)
            self.render()
            glfw.swap_buffers(self.win)
            self.fps += (1.0 / max(dt, 1e-4) - self.fps) * 0.05
            frames += 1
            if self.args.shot and frames == 3:
                t_start = glfw.get_time()
            if self.args.shot and frames >= self.args.frames:
                if frames > 10:
                    glFinish()
                    print(f"Tempo médio por quadro: {(glfw.get_time() - t_start) / (frames - 3) * 1000:.2f} ms")
                self.screenshot(self.args.shot)
                break
        self.audio.close()
        glfw.terminate()


# --------------------------------------------------------------------- utils
def _push_out(p, c, r, seg_dir):
    v = p - c
    d = np.linalg.norm(v)
    if d >= r:
        return p
    if d < 1e-6:
        v = np.cross(seg_dir, [0, 1, 0])
        if np.linalg.norm(v) < 1e-6:
            v = np.array([1.0, 0, 0])
        d = np.linalg.norm(v)
    return c + v / d * r * 2.0


def _blackbody(T):
    """Cor aproximada de um corpo negro (algoritmo de Tanner Helland), linear 0..1."""
    t = T / 100.0
    r = 255 if t <= 66 else 329.7 * (t - 60) ** -0.1332
    g = 99.47 * math.log(t) - 161.1 if t <= 66 else 288.1 * (t - 60) ** -0.0755
    b = 255 if t >= 66 else (0 if t <= 19 else 138.5 * math.log(t - 10) - 305.0)
    c = np.clip(np.array([r, g, b]) / 255.0, 0, 1) ** 2.2
    return c / max(c.max(), 1e-3)


def _fmt_dist(d):
    km = d * 6371.0
    if km < 1:
        return f"{km * 1000:.0f} m"
    if km < 1e6:
        return f"{km:,.0f} km".replace(",", ".")
    return f"{km / 1e6:,.1f} mi km".replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_speed(v):
    kms = v * 6371.0
    if kms < 1:
        return f"{kms * 1000:.0f} m/s"
    return f"{kms:,.0f} km/s".replace(",", ".")


def _fmt_time(days_per_s):
    if days_per_s == 0:
        return "PAUSADO"
    s = days_per_s * 86400
    if s < 3600:
        return f"{s / 60:.1f} min/s"
    if s < 86400:
        return f"{s / 3600:.1f} h/s"
    return f"{days_per_s:g} dias/s"
