#version 330 core
#include "common.glsl"
#include "lighting.glsl"
// Materiais do traje: tecido (difuso), capacete (verniz), viseira dourada (espelho)
in vec3 vPos;
in vec3 vN;
in vec3 vT;
in vec2 vUV;
in float vFlogz;
in vec2 vGouraud;

uniform sampler2D uAlbedo;
uniform sampler2D uSky;
uniform int   uUseTex, uUseTextures, uShadingMode;
uniform vec3  uBaseColor;
uniform vec3  uF0;            // reflectância especular na incidência normal
uniform float uShininess, uSpecStrength;
uniform float uMirror;        // 0 = sem reflexo, 1 = espelho (viseira)
uniform vec3  uEmissive;
uniform float uIrradiance, uAmbient, uSkyIntensity;
uniform vec3  uShineDir, uShineColor;
out vec4 fragColor;

void main() {
    vec3 P = vPos;
    vec3 V = normalize(-P);
    vec3 L = normalize(uSunPos - P);
    vec3 Ng = normalize(vN);
    if (uShadingMode == 2) Ng = normalize(cross(dFdx(P), dFdy(P)));
    vec3 N = Ng;
    if (!gl_FrontFacing) { N = -N; Ng = -Ng; }

    vec3 albedo = uBaseColor;
    if (uUseTex == 1 && uUseTextures == 1) albedo *= texture(uAlbedo, vUV).rgb;

    float shadow = eclipseFactor(P) * ringShadow(P, L) * shadowMapFactor(P, Ng);
    vec3 E = uSunColor * uIrradiance * shadow;

    float NdotL = max(dot(N, L), 0.0);
    vec3 F = fresnelSchlick(uF0, max(dot(N, V), 0.0));
    vec3 color;
    if (uShadingMode == 1) {
        color = (albedo * vGouraud.x + uF0 * vGouraud.y) * E;
    } else {
        vec3 spec = uSpecStrength * F * blinnPhong(N, L, V, uShininess) * NdotL;
        color = (albedo * (1.0 - uMirror) * NdotL + spec) * E;
    }
    // reflexo do ambiente (Via Láctea) na viseira — environment mapping equiretangular
    if (uMirror > 0.0) {
        vec3 R = reflect(-V, N);
        vec3 env = textureLod(uSky, dirToEquirect(R), 0.0).rgb * uSkyIntensity;
        color += env * F * uMirror * 4.0;
    }
    // luz refletida pelo planeta mais próximo (planetshine) + ambiente mínimo
    color += albedo * (1.0 - uMirror) * (uShineColor * max(dot(N, uShineDir) * 0.7 + 0.3, 0.0) + vec3(uAmbient));
    color += uEmissive;
    fragColor = vec4(color, 1.0);
    gl_FragDepth = logDepth(vFlogz);
}
