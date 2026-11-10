"""Astronauta Cósmico — voe livremente pelo Sistema Solar (OpenGL 3.3 core + Python).

Uso:
    python main.py                       # jogar
    python main.py --goto saturn         # começar perto de Saturno
    python main.py --shot saida.png --goto earth --frames 30   # captura sem janela
"""
import argparse
import os
import sys

import OpenGL
OpenGL.ERROR_CHECKING = os.environ.get("GL_DEBUG") == "1"   # desligado = muito mais rápido

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astro.app import App  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Astronauta Cósmico (OpenGL)")
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--goto", default="earth", help="corpo inicial (earth, moon, mars, jupiter, saturn, ...)")
    p.add_argument("--dist", type=float, default=3.0, help="distância inicial em raios do corpo")
    p.add_argument("--angle", type=float, default=55.0, help="ângulo (graus) a partir da direção do Sol")
    p.add_argument("--elev", type=float, default=12.0, help="elevação (graus) acima do plano da órbita")
    p.add_argument("--time", type=float, default=0.0, help="dia inicial da simulação")
    p.add_argument("--fp", action="store_true", help="começar em primeira pessoa")
    p.add_argument("--procedural", action="store_true", help="ignorar texturas reais baixadas")
    p.add_argument("--quality", type=float, default=1.0, help="escala da resolução das texturas procedurais")
    p.add_argument("--nohud", action="store_true")
    p.add_argument("--shading", type=int, default=0, help="0 Phong, 1 Gouraud, 2 Flat")
    p.add_argument("--lantern", action="store_true")
    p.add_argument("--mute", action="store_true", help="sem som")
    p.add_argument("--speak", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--orbits", action="store_true")
    p.add_argument("--wire", action="store_true")
    p.add_argument("--camyaw", type=float, default=0.0, help="giro inicial da câmera em torno do astronauta (graus)")
    p.add_argument("--fly", action="store_true", help="pose de voo")
    p.add_argument("--shot", default=None, help="renderiza sem janela visível e salva PNG")
    p.add_argument("--frames", type=int, default=20)
    args = p.parse_args()
    App(args).run()


if __name__ == "__main__":
    main()
