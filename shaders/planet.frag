#version 330 core
#include "common.glsl"
#include "lighting.glsl"

in vec3 vPos;
in vec3 vN;
in vec3 vT;
in vec2 vUV;
in float vFlogz;
in vec2 vGouraud;
in vec3 vObj;

uniform sampler2D uAlbedo;     // unidade 0 (sRGB)
uniform sampler2D uNormalTex;  // unidade 1 (linear, espaço tangente)
uniform sampler2D uSpecTex;    // unidade 2 (máscara especular: oceanos)
uniform sampler2D uNightTex;   // unidade 3 (luzes das cidades)
uniform sampler2D uCloudTex;   // unidade 4 (densidade das nuvens -> sombra das nuvens)

uniform int   uHasNormal, uHasSpec, uHasNight, uHasCloudShadow;
uniform int   uIsCloud;        // 1 = desenhando a camada de nuvens
uniform int   uUseTextures, uNormalMapOn;
uniform int   uShadingMode;    // 0 = Phong (por fragmento), 1 = Gouraud, 2 = Flat
uniform float uNormalStrength;
uniform float uRoughness;      // sigma do Oren-Nayar
uniform float uSpecStrength, uShininess;
uniform float uIrradiance;     // irradiância solar relativa (lei do inverso do quadrado)
uniform float uAmbient;
uniform vec3  uShineDir, uShineColor;   // luz refletida pelo planeta-pai (earthshine)
uniform float uNightIntensity;
uniform float uCloudOffset;
uniform vec3  uFlatColor;
uniform mat3  uNormalMat;
uniform float uDetail;         // intensidade do detalhe procedural (0 = desligado)
uniform float uDetailFreq;     // frequência base (no espaço do objeto, esfera unitária)
uniform float uDetailFade;     // distância (mundo) a partir da qual o detalhe desaparece

out vec4 fragColor;

void main() {
    vec3 P = vPos;
    vec3 V = normalize(-P);
    vec3 L = normalize(uSunPos - P);
    vec3 Ng = normalize(vN);                 // normal geométrica (interpolada)
    vec3 N = Ng;

    if (uShadingMode == 2) {
        // FLAT: normal constante por triângulo, a partir das derivadas de tela
        N = normalize(cross(dFdx(P), dFdy(P)));
        Ng = N;
    } else if (uShadingMode == 0 && uHasNormal == 1 && uNormalMapOn == 1) {
        // NORMAL MAPPING: base TBN ortonormalizada (Gram-Schmidt)
        vec3 T = normalize(vT - Ng * dot(Ng, vT));
        vec3 B = cross(Ng, T);
        vec3 nt = texture(uNormalTex, vUV).xyz * 2.0 - 1.0;
        nt.xy *= uNormalStrength;
        N = normalize(mat3(T, B, Ng) * nt);
    }

    vec4 tex = texture(uAlbedo, vUV);
    vec3 albedo = (uUseTextures == 1) ? tex.rgb : uFlatColor;

    // DETAIL TEXTURING: de muito perto a textura é magnificada (1 texel ~ dezenas
    // de metros). Somamos ruído 3D procedural (textura sólida) à cor e à normal,
    // com peso que cresce conforme a câmera se aproxima.
    float wDet = uDetail * (1.0 - smoothstep(uDetailFade * 0.25, uDetailFade, length(P)));
    if (uHasSpec == 1) wDet *= 1.0 - texture(uSpecTex, vUV).r;     // sem detalhe sobre o oceano
    if (wDet > 0.002 && uIsCloud == 0 && uShadingMode == 0 && uUseTextures == 1) {
        vec3 q = normalize(vObj) * uDetailFreq;
        float n0 = fbm3(q, 6);
        float e = 0.08;
        vec3 g = vec3(fbm3(q + vec3(e, 0, 0), 6) - n0, fbm3(q + vec3(0, e, 0), 6) - n0, fbm3(q + vec3(0, 0, e), 6) - n0) / e;
        g = uNormalMat * g;
        g -= Ng * dot(g, Ng);                        // só a componente tangente
        N = normalize(N - g * 0.2 * wDet);
        albedo *= mix(1.0, 0.55 + 0.9 * n0, wDet);
    }

    // ---------------- Sombras ----------------
    float shadow = eclipseFactor(P) * ringShadow(P, L) * shadowMapFactor(P, Ng);
    float NdotLg = dot(Ng, L);
    if (uHasCloudShadow == 1 && uShadowsOn == 1) {
        float c = texture(uCloudTex, vUV + vec2(uCloudOffset, 0.0)).r;
        shadow *= 1.0 - 0.55 * c * smoothstep(0.0, 0.2, NdotLg);
    }
    vec3 E = uSunColor * uIrradiance * shadow;

    // ---------------- Camada de nuvens ----------------
    if (uIsCloud == 1) {
        float a = tex.r;
        float diff = max(dot(N, L) * 0.85 + 0.15, 0.0) * smoothstep(-0.15, 0.1, NdotLg);
        vec3 col = vec3(0.95) * diff * E + vec3(uAmbient);
        fragColor = vec4(col, a * 0.95);
        gl_FragDepth = logDepth(vFlogz);
        return;
    }

    // ---------------- Iluminação direta do Sol ----------------
    float specMask = (uHasSpec == 1 && uUseTextures == 1) ? texture(uSpecTex, vUV).r : 1.0;
    vec3 color;
    if (uShadingMode == 1) {
        // GOURAUD: usa (difuso, especular) interpolados do vertex shader
        color = (albedo * vGouraud.x + vec3(vGouraud.y * specMask)) * E;
    } else {
        float diff = orenNayar(N, L, V, uRoughness);
        // evita que o normal map "acenda" o lado noturno: o terminador é geométrico
        diff *= smoothstep(-0.05, 0.08, NdotLg);
        float NdotL = max(dot(N, L), 0.0);
        vec3 F = fresnelSchlick(vec3(0.02), max(dot(normalize(L + V), V), 0.0));
        vec3 spec = uSpecStrength * specMask * blinnPhong(N, L, V, uShininess) * NdotL * F * 3.0;
        spec *= step(0.0, NdotLg);
        color = (albedo * diff + spec) * E;
    }

    // ---------------- Termos indiretos ----------------
    color += albedo * uAmbient;
    color += albedo * uShineColor * max(dot(Ng, uShineDir), 0.0);                  // earthshine
    color += spotLight(P, N, V, albedo, 16.0, 0.2);                                 // lanterna

    // luzes das cidades: só no lado noturno (transição suave no crepúsculo)
    if (uHasNight == 1 && uUseTextures == 1) {
        float night = smoothstep(0.05, -0.15, NdotLg);
        float cloudCover = (uHasCloudShadow == 1) ? texture(uCloudTex, vUV + vec2(uCloudOffset, 0.0)).r : 0.0;
        color += texture(uNightTex, vUV).rgb * uNightIntensity * night * (1.0 - 0.7 * cloudCover);
    }

    fragColor = vec4(color, 1.0);
    gl_FragDepth = logDepth(vFlogz);
}
