#version 330 core
#include "common.glsl"
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec3 aTangent;

uniform mat4 uModel;       // objeto -> mundo (relativo à câmera)
uniform mat3 uNormalMat;   // inversa-transposta de uModel (3x3)
uniform mat4 uViewProj;
uniform vec3 uSunPos;
uniform float uShininess;
uniform float uSpecStrength;

out vec3 vPos;
out vec3 vN;
out vec3 vT;
out vec2 vUV;
out float vFlogz;
out vec2 vGouraud;         // (difuso, especular) calculados POR VÉRTICE (modo Gouraud)
out vec3 vObj;             // posição no espaço do objeto (textura sólida de detalhe)

void main() {
    vObj = aPos;
    vec4 wp = uModel * vec4(aPos, 1.0);
    vPos = wp.xyz;
    vN = uNormalMat * aNormal;
    vT = mat3(uModel) * aTangent;
    vUV = aUV;
    gl_Position = uViewProj * wp;
    vFlogz = 1.0 + gl_Position.w;

    // Gouraud: iluminação avaliada nos vértices e interpolada pelo rasterizador
    vec3 N = normalize(vN);
    vec3 L = normalize(uSunPos - wp.xyz);
    vec3 V = normalize(-wp.xyz);
    float d = max(dot(N, L), 0.0);
    float s = d > 0.0 ? uSpecStrength * blinnPhong(N, L, V, uShininess) * d : 0.0;
    vGouraud = vec2(d, s);
}
