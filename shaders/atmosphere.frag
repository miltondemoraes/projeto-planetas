#version 330 core
#include "common.glsl"
// ---------------------------------------------------------------------------
// Espalhamento atmosférico (Rayleigh + Mie), espalhamento SIMPLES por ray marching.
// Para cada pixel: marchamos o raio de visão dentro da casca atmosférica; em cada
// amostra, marchamos outro raio até o Sol para obter a profundidade óptica.
//   L_in = Σ  E_sol · T(sol->p) · T(p->olho) · [β_R ρ_R(h) P_R(θ) + β_M ρ_M(h) P_M(θ)] ds
// Densidade exponencial ρ(h) = exp(-h/H). Rayleigh ∝ λ^-4 -> céu azul / pôr do sol vermelho.
// Tudo em unidades de "raios do planeta" (planeta = esfera de raio 1).
// ---------------------------------------------------------------------------
in vec3 vPos;
in float vFlogz;

uniform vec3  uCenter;        // centro do planeta (relativo à câmera)
uniform float uRadius;        // raio do planeta (mundo)
uniform float uAtmoRadius;    // raio da atmosfera (em raios do planeta, ex.: 1.03)
uniform vec3  uBetaR;         // coef. de espalhamento Rayleigh (por raio planetário)
uniform vec3  uBetaM;         // coef. de espalhamento Mie
uniform float uHR, uHM;       // alturas de escala
uniform float uG;             // anisotropia Mie (Henyey-Greenstein)
uniform float uSunIntensity;
uniform vec3  uSunPos;
uniform vec3  uSunColor;

out vec4 fragColor;

vec2 raySphere(vec3 o, vec3 d, float r) {
    float b = dot(o, d);
    float c = dot(o, o) - r * r;
    float h = b * b - c;
    if (h < 0.0) return vec2(1e9, -1e9);
    h = sqrt(h);
    return vec2(-b - h, -b + h);
}

void main() {
    vec3 D = normalize(vPos);
    vec3 O = -uCenter / uRadius;
    vec2 ta = raySphere(O, D, uAtmoRadius);
    if (ta.x > ta.y || ta.y < 0.0) discard;
    float t0 = max(ta.x, 0.0), t1 = ta.y;
    vec2 tp = raySphere(O, D, 1.0);
    if (tp.x < tp.y && tp.x > 0.0) t1 = min(t1, tp.x);
    vec3 L = normalize(uSunPos - uCenter);

    const int NS = 16, NL = 6;
    float ds = (t1 - t0) / float(NS);
    vec3 sumR = vec3(0.0), sumM = vec3(0.0);
    float odR = 0.0, odM = 0.0;
    for (int i = 0; i < NS; i++) {
        vec3 p = O + D * (t0 + ds * (float(i) + 0.5));
        float h = length(p) - 1.0;
        float hr = exp(-h / uHR) * ds;
        float hm = exp(-h / uHM) * ds;
        odR += hr; odM += hm;
        // sombra do planeta sobre a atmosfera
        vec2 tpl = raySphere(p, L, 1.0);
        if (tpl.x < tpl.y && tpl.x > 0.0) continue;
        vec2 tl = raySphere(p, L, uAtmoRadius);
        float dsl = tl.y / float(NL);
        float lR = 0.0, lM = 0.0;
        for (int j = 0; j < NL; j++) {
            vec3 pl = p + L * (dsl * (float(j) + 0.5));
            float hl = max(length(pl) - 1.0, 0.0);
            lR += exp(-hl / uHR) * dsl;
            lM += exp(-hl / uHM) * dsl;
        }
        vec3 tau = uBetaR * (odR + lR) + uBetaM * 1.1 * (odM + lM);
        vec3 att = exp(-tau);
        sumR += att * hr;
        sumM += att * hm;
    }
    float mu = dot(D, L);
    float pR = 3.0 / (16.0 * PI) * (1.0 + mu * mu);
    float g = uG;
    float pM = 3.0 / (8.0 * PI) * ((1.0 - g * g) * (1.0 + mu * mu)) / ((2.0 + g * g) * pow(1.0 + g * g - 2.0 * g * mu, 1.5));
    vec3 col = uSunIntensity * uSunColor * (sumR * uBetaR * pR + sumM * uBetaM * pM);
    fragColor = vec4(col, 1.0);     // mistura ADITIVA (ONE, ONE)
    gl_FragDepth = logDepth(vFlogz);
}
