"""(Opcional) Baixa texturas REAIS dos planetas do Solar System Scope.

Licença: Creative Commons Attribution 4.0 (CC BY 4.0) — https://www.solarsystemscope.com/textures/
Crédito obrigatório: "Texturas: Solar System Scope (INOVE), CC BY 4.0".

Sem estas texturas o jogo funciona normalmente com texturas PROCEDURAIS geradas
em numpy. Com elas, o carregador (astro/assets.py) as usa automaticamente.
Para voltar às procedurais: python main.py --procedural

Uso:  python tools/baixar_texturas.py            (resolução 2k, ~20 MB)
"""
import os
import sys
import urllib.request

BASE = "https://www.solarsystemscope.com/textures/download/"
FILES = [
    "2k_sun.jpg", "2k_mercury.jpg", "2k_venus_atmosphere.jpg", "2k_earth_daymap.jpg", "2k_earth_nightmap.jpg",
    "2k_earth_clouds.jpg", "2k_earth_normal_map.tif", "2k_earth_specular_map.tif", "2k_moon.jpg", "2k_mars.jpg",
    "2k_jupiter.jpg", "2k_saturn.jpg", "2k_saturn_ring_alpha.png", "2k_uranus.jpg", "2k_neptune.jpg",
    "2k_stars_milky_way.jpg",
]

DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "textures")


def main():
    os.makedirs(DEST, exist_ok=True)
    ok = 0
    for fn in FILES:
        path = os.path.join(DEST, fn)
        if os.path.exists(path):
            print(f"  já existe: {fn}")
            ok += 1
            continue
        try:
            req = urllib.request.Request(BASE + fn, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            with open(path, "wb") as f:
                f.write(data)
            print(f"  ok: {fn} ({len(data) / 1e6:.1f} MB)")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FALHOU: {fn}: {e}", file=sys.stderr)
    print(f"{ok}/{len(FILES)} texturas em {DEST}")


if __name__ == "__main__":
    main()
