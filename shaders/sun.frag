#version 330 core
#include "common.glsl"
// O Sol é EMISSIVO (não recebe luz): granulação procedural animada + escurecimento de limbo
in vec3 vPos;
in vec3 vN;
in vec3 vObjN;
in vec2 vUV;
in float vFlogz;

uniform float uTime;
uniform float uEmission;
uniform int   uHasTex;
uniform sampler2D uAlbedo;
out vec4 fragColor;

void main() {
    vec3 N = normalize(vN);
    vec3 V = normalize(-vPos);
    float mu = clamp(dot(N, V), 0.0, 1.0);
    vec3 p = normalize(vObjN);
    float t = uTime * 0.6;
    // granulação (células de convecção) + manchas solares
    float gran = fbm3(p * 55.0 + vec3(t, -t * 0.7, t * 0.4), 4);
    float big = fbm3(p * 7.0 + vec3(0.0, t * 0.05, 0.0), 4);
    float spots = smoothstep(0.66, 0.72, fbm3(p * 4.0 + 3.0, 4)) * smoothstep(0.45, 0.2, abs(p.y));
    float g = 0.75 + 0.5 * gran + 0.2 * big;
    // escurecimento de limbo (lei quadrática): I(mu) = 1 - a(1-mu) - b(1-mu)^2
    float limb = 1.0 - 0.55 * (1.0 - mu) - 0.25 * (1.0 - mu) * (1.0 - mu);
    vec3 hot = vec3(1.0, 0.92, 0.80);
    vec3 cool = vec3(1.0, 0.55, 0.20);
    vec3 col = mix(cool, hot, pow(limb, 0.8)) * limb * g * (1.0 - 0.75 * spots);
    // textura real: usamos só a LUMINÂNCIA como detalhe (a cor vem da temperatura/limbo)
    if (uHasTex == 1) col *= 0.35 + 1.3 * dot(texture(uAlbedo, vUV).rgb, vec3(0.2126, 0.7152, 0.0722));
    fragColor = vec4(col * uEmission, 1.0);
    gl_FragDepth = logDepth(vFlogz);
}
