"""Áudio PROCEDURAL em tempo real (sem arquivos de som): tudo é sintetizado com numpy.

Arquitetura
  * `sounddevice` abre um stream de saída; a função `_callback` roda numa thread de
    áudio e precisa gerar 1024 amostras estéreo a cada ~23 ms.
  * O jogo (thread principal) só publica PARÂMETROS a cada quadro (`set_state`) e
    dispara EVENTOS (`play`). O callback lê esses valores e sintetiza.
  * Ganhos mudam por rampa linear dentro de cada bloco -> sem estalos ("clicks").

Camadas sonoras
  1. Ambiente do espaço: drone grave + ruído marrom muito baixo (sempre).
  2. "Voz" de cada corpo celeste: acorde de senoides com tremolo lento; a altura
     depende do tamanho (planetas grandes = graves) e o timbre do tipo (rochoso,
     gasoso, gelado, Terra). Volume ~ 1/(1 + (altitude / 2,5R)²) e pan estéreo pela
     direção do corpo em relação à câmera -> o som cresce ao se aproximar.
  3. Sol: rugido de plasma (ruído grave) + estalos.
  4. Vento de velocidade: ruído filtrado cuja "cor" sobe com a velocidade aparente.
  5. Vento atmosférico: ao entrar numa atmosfera, proporcional à densidade × velocidade.
  6. Anéis e cinturão de asteroides: cintilação aguda / estalidos de rochas.
  7. Eventos: início/chegada do piloto automático, toque na superfície, cliques de interface.
"""
import math
import threading

import numpy as np

try:
    import sounddevice as sd
except Exception:  # noqa: BLE001
    sd = None

SR = 44100
BLOCK = 1024
LOOP_SEC = 6


def _shaped_noise(rng, n, lo, hi, tilt=0.0):
    """Ruído com espectro moldado por FFT (filtro passa-banda + inclinação 1/f^tilt).

    O filtro no domínio da frequência é uma convolução CIRCULAR -> o buffer
    faz loop perfeito, sem emenda audível.
    """
    spec = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n, 1 / SR)
    mask = (1 / (1 + (lo / np.maximum(f, 1e-3)) ** 4)) * (1 / (1 + (f / hi) ** 4))
    mask *= 1 / np.maximum(f, 20) ** (tilt / 2)
    x = np.fft.irfft(spec * mask, n).astype(np.float32)
    return x / (np.abs(x).max() + 1e-9)


def _env(n, attack, decay):
    t = np.arange(n) / SR
    return (1 - np.exp(-t / max(attack, 1e-4))) * np.exp(-t / decay)


def _make_events():
    rng = np.random.default_rng(5)
    ev = {}
    # início de viagem: varredura ascendente com harmônicos
    n = int(SR * 1.6)
    t = np.arange(n) / SR
    f = 120 * (6 ** (t / 1.6))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = (np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)) * np.sin(np.pi * t / 1.6) ** 2
    ev["warp"] = (s * 0.25).astype(np.float32)
    # chegada: sino (parciais inarmônicas de um sino)
    n = int(SR * 3.0)
    t = np.arange(n) / SR
    s = sum(a * np.sin(2 * np.pi * 523.25 * r * t) * np.exp(-t / d) for a, r, d in
            [(1, 1, 1.6), (0.5, 2.76, 0.9), (0.3, 5.40, 0.5), (0.25, 1.5, 1.2)])
    ev["arrive"] = (s * _env(n, 0.003, 10) * 0.18).astype(np.float32)
    # toque na superfície: baque grave + ruído
    n = int(SR * 0.5)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * (70 - 30 * t) * t) * np.exp(-t / 0.12) + _shaped_noise(rng, n, 40, 600) * np.exp(-t / 0.05)
    ev["thump"] = (s * 0.5).astype(np.float32)
    # cliques de interface
    n = int(SR * 0.05)
    t = np.arange(n) / SR
    ev["click"] = (np.sin(2 * np.pi * 1800 * t) * np.exp(-t / 0.008) * 0.15).astype(np.float32)
    ev["up"] = (np.sin(2 * np.pi * 880 * t) * np.exp(-t / 0.02) * 0.15).astype(np.float32)
    ev["down"] = (np.sin(2 * np.pi * 440 * t) * np.exp(-t / 0.02) * 0.15).astype(np.float32)
    return ev


# timbres: lista de (razão de frequência, amplitude)
TIMBRES = {
    "rocky": [(1.0, 1.0), (2.01, 0.35), (3.03, 0.15), (0.5, 0.4)],
    "gas": [(1.0, 1.0), (1.003, 0.8), (1.5, 0.45), (2.0, 0.3), (0.5, 0.6)],
    "ice": [(1.0, 0.7), (2.0, 0.5), (3.0, 0.35), (4.02, 0.25), (6.0, 0.15)],
    "earth": [(1.0, 1.0), (1.25, 0.55), (1.5, 0.5), (2.0, 0.35), (0.5, 0.4)],
    "lava": [(1.0, 1.0), (1.06, 0.6), (2.12, 0.3), (0.5, 0.5)],
}


class Voice:
    def __init__(self, f0, timbre, lfo):
        self.partials = np.array([(f0 * r, a) for r, a in TIMBRES[timbre]], np.float64)
        self.partials[:, 1] /= self.partials[:, 1].sum()
        self.phase = np.zeros(len(self.partials))
        self.lfo_rate = lfo
        self.lfo_phase = 0.0
        self.gain = 0.0
        self.pan = 0.0


class AudioEngine:
    def __init__(self, enabled=True, bodies=()):
        self.ok = False
        self.muted = False
        self.master = 0.8
        self.lock = threading.Lock()
        self.state = {"voices": {}, "sun": (0.0, 0.0), "speed": 0.0, "wind": 0.0, "rings": 0.0, "belt": 0.0}
        self.events = []
        if not enabled or sd is None:
            if enabled:
                print("Áudio desativado (instale 'sounddevice').")
            return
        rng = np.random.default_rng(1)
        n = SR * LOOP_SEC
        self.loops = {
            "brown": _shaped_noise(rng, n, 15, 250, tilt=1.0),
            "low": _shaped_noise(rng, n, 60, 500),
            "mid": _shaped_noise(rng, n, 300, 2500),
            "high": _shaped_noise(rng, n, 2000, 9000),
            "roar": _shaped_noise(rng, n, 25, 180, tilt=0.5),
        }
        crack = np.zeros(n, np.float32)
        idx = rng.integers(0, n, 900)
        crack[idx] = rng.uniform(-1, 1, 900)
        k = np.exp(-np.arange(300) / 40.0)
        self.loops["crackle"] = np.convolve(np.concatenate([crack, crack[:299]]), k, "valid")[:n].astype(np.float32)
        self.loops["crackle"] /= np.abs(self.loops["crackle"]).max()
        self.pos = 0
        self.ev = _make_events()
        self.playing = []
        self.voices = {}
        for key, f0, timbre in bodies:
            self.voices[key] = Voice(f0, timbre, 0.07 + 0.2 * rng.random())
        self.g = {k: 0.0 for k in ("sun", "speed", "wind", "rings", "belt", "amb")}
        self.speed_mix = 0.0
        self.drone_phase = np.zeros(3)
        try:
            self.stream = sd.OutputStream(samplerate=SR, blocksize=BLOCK, channels=2, dtype="float32",
                                          callback=self._callback, latency="low")
            self.stream.start()
            self.ok = True
        except Exception as e:  # noqa: BLE001
            print("Áudio indisponível:", e)

    # ---------------------------------------------------------------- API do jogo
    def set_state(self, **kw):
        with self.lock:
            self.state.update(kw)

    def play(self, name, gain=1.0):
        if self.ok and name in self.ev:
            self.events.append((name, gain))

    def toggle_mute(self):
        self.muted = not self.muted
        return self.muted

    def close(self):
        if self.ok:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass

    # ---------------------------------------------------------------- síntese
    def _loop(self, name, n, offset=0):
        buf = self.loops[name]
        i = (self.pos + offset) % len(buf)
        if i + n <= len(buf):
            return buf[i:i + n]
        return np.concatenate([buf[i:], buf[:n - (len(buf) - i)]])

    def _callback(self, out, frames, time_info, status):
        n = frames
        t = np.arange(n) / SR
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        L = np.zeros(n, np.float32)
        R = np.zeros(n, np.float32)
        with self.lock:
            st = dict(self.state)

        def glide(key, target):
            g0 = self.g[key]
            self.g[key] = target
            return g0 + (target - g0) * ramp

        # 1) ambiente: drone grave (55 / 82,5 / 110 Hz) + ruído marrom
        amb = glide("amb", 0.05)
        for j, (f, a) in enumerate(((55.0, 1.0), (82.5, 0.5), (110.3, 0.3))):
            ph = self.drone_phase[j] + 2 * np.pi * f * t
            self.drone_phase[j] = (ph[-1] + 2 * np.pi * f / SR) % (2 * np.pi)
            s = (a * np.sin(ph) * amb * 0.5).astype(np.float32)
            L += s
            R += s
        L += self._loop("brown", n) * amb * 0.6
        R += self._loop("brown", n, SR * 3) * amb * 0.6

        # 2) vozes dos corpos celestes
        vs = st["voices"]
        for key, v in self.voices.items():
            tg, pan = vs.get(key, (0.0, v.pan))
            if tg < 1e-4 and v.gain < 1e-4:
                v.gain = 0.0
                continue
            g = v.gain + (tg - v.gain) * ramp
            p = v.pan + (pan - v.pan) * ramp
            v.gain, v.pan = tg, pan
            lfo_ph = v.lfo_phase + 2 * np.pi * v.lfo_rate * t
            v.lfo_phase = (lfo_ph[-1] + 2 * np.pi * v.lfo_rate / SR) % (2 * np.pi)
            trem = 0.75 + 0.25 * np.sin(lfo_ph)
            s = np.zeros(n)
            for k, (f, a) in enumerate(v.partials):
                ph = v.phase[k] + 2 * np.pi * f * t
                v.phase[k] = (ph[-1] + 2 * np.pi * f / SR) % (2 * np.pi)
                s += a * np.sin(ph)
            s = (s * trem * g).astype(np.float32)
            ang = (p + 1) * np.pi / 4                       # pan de potência constante
            L += s * np.cos(ang).astype(np.float32)
            R += s * np.sin(ang).astype(np.float32)

        # 3) Sol: rugido + estalos
        sg, span = st["sun"]
        gs = glide("sun", sg)
        ang = (span + 1) * np.pi / 4
        roar = self._loop("roar", n) * 0.9 + self._loop("crackle", n) * 0.25
        L += roar * gs * math.cos(ang)
        R += roar * gs * math.sin(ang)

        # 4) vento de velocidade: mistura low -> mid -> high conforme a velocidade
        spd = float(np.clip(st["speed"], 0, 1))
        gv = glide("speed", spd * 0.22)
        m = spd * 2
        wl = max(0, 1 - m)
        wm = 1 - abs(m - 1)
        wh = max(0, m - 1)
        L += (self._loop("low", n) * wl + self._loop("mid", n) * wm + self._loop("high", n) * wh) * gv
        R += (self._loop("low", n, SR) * wl + self._loop("mid", n, SR) * wm + self._loop("high", n, SR) * wh) * gv

        # 5) vento atmosférico
        gw = glide("wind", st["wind"] * 0.35)
        L += (self._loop("low", n, SR * 2) * 0.7 + self._loop("mid", n, SR * 2) * 0.3) * gw
        R += (self._loop("low", n, SR * 4) * 0.7 + self._loop("mid", n, SR * 4) * 0.3) * gw

        # 6) anéis (cintilação de gelo) e cinturão (estalidos)
        gr = glide("rings", st["rings"] * 0.22)
        sh = self._loop("high", n, SR // 2) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * (t + self.pos / SR)))
        L += sh * gr
        R += self._loop("high", n, SR * 5) * gr
        gb = glide("belt", st["belt"] * 0.55)
        L += self._loop("crackle", n, SR) * gb
        R += self._loop("crackle", n, SR * 2) * gb

        # 7) eventos (one-shots)
        while self.events:
            name, gain = self.events.pop(0)
            self.playing.append([self.ev[name], 0, gain])
        keep = []
        for item in self.playing:
            buf, i, gain = item
            chunk = buf[i:i + n]
            L[:len(chunk)] += chunk * gain
            R[:len(chunk)] += chunk * gain
            item[1] += n
            if item[1] < len(buf):
                keep.append(item)
        self.playing = keep

        self.pos = (self.pos + n) % (SR * LOOP_SEC)
        vol = 0.0 if self.muted else self.master
        out[:, 0] = np.tanh(L * vol)                      # saturação suave (evita clipping)
        out[:, 1] = np.tanh(R * vol)


def body_voice_params(key, radius, kind):
    """Frequência fundamental pelo tamanho (maior = mais grave) e timbre pelo tipo."""
    f0 = float(np.clip(98.0 * radius ** -0.3, 45.0, 260.0))
    return key, f0, kind
