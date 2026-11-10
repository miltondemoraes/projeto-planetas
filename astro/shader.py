"""Compilação/link de programas GLSL com suporte a #include e cache de uniforms."""
import os
import re
import numpy as np
from OpenGL.GL import *

SHADER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shaders")
_INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.M)


def _load_source(name, seen=None):
    seen = seen or set()
    path = os.path.join(SHADER_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    def repl(m):
        inc = m.group(1)
        if inc in seen:
            return ""
        seen.add(inc)
        return _load_source(inc, seen)

    return _INCLUDE.sub(repl, src)


def _compile(src, kind, name):
    sh = glCreateShader(kind)
    glShaderSource(sh, src)
    glCompileShader(sh)
    if not glGetShaderiv(sh, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(sh).decode(errors="replace")
        numbered = "\n".join(f"{i + 1:4d}: {l}" for i, l in enumerate(src.splitlines()))
        raise RuntimeError(f"Erro compilando {name}:\n{log}\n{numbered}")
    return sh


class Shader:
    def __init__(self, vert, frag):
        self.name = f"{vert}+{frag}"
        vs = _compile(_load_source(vert), GL_VERTEX_SHADER, vert)
        fs = _compile(_load_source(frag), GL_FRAGMENT_SHADER, frag)
        self.prog = glCreateProgram()
        glAttachShader(self.prog, vs)
        glAttachShader(self.prog, fs)
        glLinkProgram(self.prog)
        if not glGetProgramiv(self.prog, GL_LINK_STATUS):
            raise RuntimeError(f"Erro de link {self.name}:\n{glGetProgramInfoLog(self.prog).decode()}")
        glDeleteShader(vs)
        glDeleteShader(fs)
        self._locs = {}

    def use(self):
        glUseProgram(self.prog)
        return self

    def loc(self, name):
        l = self._locs.get(name)
        if l is None:
            l = glGetUniformLocation(self.prog, name)
            self._locs[name] = l
        return l

    # --- setters (ignoram uniforms otimizados/ausentes: loc == -1) ---
    def i(self, name, v):
        l = self.loc(name)
        if l >= 0:
            glUniform1i(l, int(v))

    def f(self, name, v):
        l = self.loc(name)
        if l >= 0:
            glUniform1f(l, float(v))

    def v2(self, name, v):
        l = self.loc(name)
        if l >= 0:
            glUniform2f(l, float(v[0]), float(v[1]))

    def v3(self, name, v):
        l = self.loc(name)
        if l >= 0:
            glUniform3f(l, float(v[0]), float(v[1]), float(v[2]))

    def v4(self, name, v):
        l = self.loc(name)
        if l >= 0:
            glUniform4f(l, float(v[0]), float(v[1]), float(v[2]), float(v[3]))

    def v4a(self, name, arr):
        l = self.loc(name + "[0]")
        if l >= 0 and len(arr):
            a = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1, 4)
            glUniform4fv(l, len(a), a)

    def m3(self, name, m):
        l = self.loc(name)
        if l >= 0:
            glUniformMatrix3fv(l, 1, GL_TRUE, np.ascontiguousarray(m, dtype=np.float32))

    def m4(self, name, m):
        l = self.loc(name)
        if l >= 0:
            glUniformMatrix4fv(l, 1, GL_TRUE, np.ascontiguousarray(m, dtype=np.float32))
