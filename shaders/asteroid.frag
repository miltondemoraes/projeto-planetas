#version 330 core
#include "common.glsl"
#include "lighting.glsl"
in vec3 vPos;
in vec3 vN;
in vec3 vObj;
in float vSeed;
in float vFlogz;
uniform float uIrradiance, uAmbient;
uniform int uShadingMode;
out vec4 fragColor;
void main() {
    vec3 N = (uShadingMode == 2) ? normalize(cross(dFdx(vPos), dFdy(vPos))) : normalize(vN);
    vec3 L = normalize(uSunPos - vPos);
    vec3 V = normalize(-vPos);
    // textura sólida procedural (3D): não precisa de UV
    float n = fbm3(vObj * 6.0 + vSeed * 37.0, 5);
    vec3 a = mix(vec3(0.20, 0.18, 0.16), vec3(0.42, 0.38, 0.33), vSeed);
    vec3 albedo = a * (0.6 + 0.8 * n);
    float diff = orenNayar(N, L, V, 0.6);
    vec3 col = albedo * diff * uSunColor * uIrradiance + albedo * uAmbient;
    col += spotLight(vPos, N, V, albedo, 8.0, 0.0);
    fragColor = vec4(col, 1.0);
    gl_FragDepth = logDepth(vFlogz);
}
