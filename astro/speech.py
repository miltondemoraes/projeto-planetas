"""Falas do astronauta: síntese de voz (TTS do Windows) + efeito de rádio do capacete.

* As frases são geradas UMA vez com a voz pt-BR do Windows (tools/tts_windows.ps1:
  "Microsoft Daniel" via WinRT, ou SAPI como alternativa), numa thread em segundo
  plano, e ficam em cache em assets/speech/.
* Cada WAV recebe o efeito de "rádio": filtro passa-banda 300–3400 Hz (banda telefônica),
  leve saturação e os bipes de Quindar (2525 Hz no início, 2475 Hz no fim) usados nas
  comunicações das missões Apollo.
* Sem Windows/TTS o jogo continua funcionando: mostra só a legenda.
"""
import hashlib
import os
import subprocess
import threading
import wave

import numpy as np

from .audio import SR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "assets", "speech")
PS1 = os.path.join(ROOT, "tools", "tts_windows.ps1")

PLANET_NAMES = {"mercury": "Mercúrio", "venus": "Vênus", "mars": "Marte", "jupiter": "Júpiter",
                "saturn": "Saturno", "uranus": "Urano", "neptune": "Netuno"}

PHRASES = {k: f"Eu me autoproclamo presidente de {n}." for k, n in PLANET_NAMES.items()}
PHRASES["earth"] = "Eu me autoproclamo presidente do Brasil."
PHRASES["sun"] = "Meu Deus, cheguei no Ceará, foi?"


def _cache_path(key, text):
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return os.path.join(CACHE, f"{key}_{h}.wav")


def _read_wav(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype={1: np.uint8, 2: np.int16}[sw])
    x = data.astype(np.float32)
    x = (x - 128) / 128 if sw == 1 else x / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    if sr != SR:                                   # reamostragem linear para 44,1 kHz
        t_new = np.arange(int(len(x) * SR / sr)) / SR
        x = np.interp(t_new, np.arange(len(x)) / sr, x).astype(np.float32)
    return x


def radio_fx(x):
    """Efeito de rádio do capacete: passa-banda por FFT + saturação + bipes de Quindar."""
    n = len(x)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    spec *= (1 / (1 + (300 / np.maximum(f, 1)) ** 4)) * (1 / (1 + (f / 3400) ** 4))
    y = np.fft.irfft(spec, n)
    y = np.tanh(y / (np.abs(y).max() + 1e-9) * 1.8) * 0.55
    hiss = np.random.default_rng(3).normal(0, 0.01, n)

    def tone(freq, dur=0.25):
        t = np.arange(int(SR * dur)) / SR
        return (np.sin(2 * np.pi * freq * t) * np.minimum(1, np.minimum(t, dur - t) * 200) * 0.12)

    gap = np.zeros(int(SR * 0.08))
    return np.concatenate([tone(2525), gap, y + hiss, gap, tone(2475)]).astype(np.float32)


class Speech:
    def __init__(self, audio, enabled=True):
        self.audio = audio
        self.clips = {}                  # chave -> duração (s)
        self.spoken = set()              # corpos onde já falou (até se afastar)
        self.subtitle = ""
        self.sub_t = 0.0
        self.ready = not enabled         # sem TTS: só legendas, pronto imediatamente
        if enabled:
            threading.Thread(target=self._prepare, daemon=True).start()

    def _prepare(self):
        os.makedirs(CACHE, exist_ok=True)
        for key, text in PHRASES.items():
            path = _cache_path(key, text)
            try:
                if not os.path.exists(path) and os.name == "nt":
                    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", PS1,
                                    "-Out", path, "-Text", text], capture_output=True, timeout=60,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if os.path.exists(path):
                    clip = radio_fx(_read_wav(path))
                    if self.audio.ok:
                        self.audio.ev["fala_" + key] = clip
                    self.clips[key] = len(clip) / SR
            except Exception as e:  # noqa: BLE001
                print(f"Fala '{key}' indisponível: {e}")
        self.ready = True

    def update(self, dt, system, pos):
        """Dispara a fala ao chegar perto (altitude < 3R; Saturno < 6R; Sol < 4R) e rearma ao se afastar (> 10R)."""
        self.sub_t = max(0.0, self.sub_t - dt)
        if not self.ready:               # espera as falas ficarem prontas (1ª execução)
            return
        for b in system.bodies:
            if b.key not in PHRASES:
                continue
            alt = np.linalg.norm(pos - b.pos) - b.radius
            near = b.radius * (4.0 if b is system.sun else (6.0 if b.d.rings else 3.0))
            if alt > b.radius * 10:
                self.spoken.discard(b.key)
            elif alt < near and b.key not in self.spoken and self.sub_t <= 0:
                self.say(b.key)

    def say(self, key):
        self.spoken.add(key)
        self.subtitle = PHRASES[key]
        self.sub_t = self.clips.get(key, 3.0) + 0.8
        self.audio.play("fala_" + key, 1.6)
