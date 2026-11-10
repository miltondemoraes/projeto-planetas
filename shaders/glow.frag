#version 330 core
#include "common.glsl"
// Coroa / halo do Sol (mistura aditiva). Raios radiais suaves + decaimento exponencial.
in vec2 vC;
in float vFlogz;
uniform float uIntensity;
uniform float uInner;     // raio do disco solar em unidades do quad
uniform float uTime;
out vec4 fragColor;
void main() {
    float r = length(vC);
    if (r > 1.0) discard;
    float x = max(r - uInner, 0.0) / (1.0 - uInner);
    float ang = atan(vC.y, vC.x);
    float rays = 0.75 + 0.25 * vnoise3(vec3(cos(ang) * 6.0, sin(ang) * 6.0, uTime * 0.05));
    float g = (exp(-x * 9.0) * 0.9 * rays + exp(-x * 3.0) * 0.12) * (1.0 - smoothstep(0.85, 1.0, r));
    vec3 col = vec3(1.0, 0.78, 0.5) * g * uIntensity;
    fragColor = vec4(col, 1.0);
    gl_FragDepth = logDepth(vFlogz);
}
