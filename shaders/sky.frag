#version 330 core
#include "common.glsl"
// Céu de fundo (Via Láctea) amostrado por DIREÇÃO: reconstrói o raio de visão
// de cada pixel com a inversa de (Projeção × Rotação da câmera).
in vec2 vUV;
uniform mat4 uInvViewProj;
uniform sampler2D uSky;
uniform float uSkyIntensity;
out vec4 fragColor;
void main() {
    vec4 p = uInvViewProj * vec4(vUV * 2.0 - 1.0, 1.0, 1.0);
    vec3 d = normalize(p.xyz / p.w);
    vec3 c = textureLod(uSky, dirToEquirect(d), 0.0).rgb;
    fragColor = vec4(c * uSkyIntensity, 1.0);
}
