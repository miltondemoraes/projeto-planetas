#version 330 core
// INSTANCED RENDERING: 1 draw call para milhares de asteroides.
// Cada instância tem atributos próprios (glVertexAttribDivisor = 1); a órbita
// kepleriana e a rotação são calculadas aqui na GPU a partir do tempo.
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 4) in vec4 iOrbit;    // raio, ângulo inicial, altura, escala
layout(location = 5) in vec4 iRot;      // eixo de rotação (xyz), velocidade de rotação
layout(location = 6) in vec2 iMotion;   // velocidade angular orbital, semente de cor

uniform mat4 uViewProj;
uniform vec3 uCenter;     // posição do Sol relativa à câmera
uniform float uTime;      // dias

out vec3 vPos;
out vec3 vN;
out vec3 vObj;
out float vSeed;
out float vFlogz;

mat3 rodrigues(vec3 k, float a) {
    float c = cos(a), s = sin(a);
    mat3 K = mat3(0, k.z, -k.y, -k.z, 0, k.x, k.y, -k.x, 0);
    return mat3(1.0) + s * K + (1.0 - c) * K * K;
}

void main() {
    float ang = iOrbit.y + iMotion.x * uTime;
    vec3 c = uCenter + vec3(iOrbit.x * cos(ang), iOrbit.z, -iOrbit.x * sin(ang));
    mat3 R = rodrigues(normalize(iRot.xyz), iRot.w * uTime);
    vec3 wp = c + R * (aPos * iOrbit.w);
    vPos = wp;
    vN = R * aNormal;
    vObj = aPos;
    vSeed = iMotion.y;
    gl_Position = uViewProj * vec4(wp, 1.0);
    vFlogz = 1.0 + gl_Position.w;
}
