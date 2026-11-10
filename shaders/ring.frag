#version 330 core
#include "common.glsl"
#include "lighting.glsl"
// Anéis de Saturno: camada fina de partículas de gelo.
// A textura guarda a OPACIDADE α = 1 - e^(-τ)  (τ = profundidade óptica normal).
// Usamos alpha PRÉ-MULTIPLICADO (blend ONE, ONE_MINUS_SRC_ALPHA):
//   cor_final = S + (1 - α)·fundo      S = radiância espalhada pela camada
in vec3 vPos;
in vec3 vN;
in vec2 vUV;
in float vFlogz;

uniform float uIrradiance;
uniform float uAmbient;
out vec4 fragColor;

void main() {
    vec4 t = texture(uRingTex, vec2(vUV.x, 0.5));
    float alpha = t.a;
    if (alpha < 0.003) discard;
    vec3 P = vPos;
    vec3 N = normalize(vN);
    vec3 L = normalize(uSunPos - P);
    vec3 V = normalize(-P);
    float NL = dot(N, L), NV = dot(N, V);
    float tau = -log(max(1.0 - alpha, 1e-3));
    float g = 0.55, mu = dot(V, -L);                       // mu = 1 olhando contra o Sol
    float hg = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * mu, 1.5);   // Henyey-Greenstein (x4π)
    vec3 S;
    if (NL * NV > 0.0) {
        // face iluminada: reflexão difusa (pouco dependente do ângulo, multipla dispersão)
        S = t.rgb * alpha * mix(abs(NL), 1.0, 0.55) * (0.8 + 0.2 * hg);
    } else {
        // face escura: luz que ATRAVESSA a camada (máxima para τ ~ 1; espalhamento frontal)
        S = t.rgb * tau * exp(-tau) * (0.5 + 1.2 * hg) * sqrt(abs(NL));
    }
    float shadow = eclipseFactor(P);                       // sombra do próprio Saturno sobre os anéis
    vec3 col = S * uSunColor * uIrradiance * shadow + t.rgb * alpha * uAmbient;
    fragColor = vec4(col, alpha);
    gl_FragDepth = logDepth(vFlogz);
}
