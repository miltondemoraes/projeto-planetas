#version 330 core
// Vertex shader genérico (anéis, atmosfera, Sol): passa posição/normal/UV
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;

uniform mat4 uModel;
uniform mat3 uNormalMat;
uniform mat4 uViewProj;

out vec3 vPos;
out vec3 vN;
out vec3 vObjN;
out vec2 vUV;
out float vFlogz;

void main() {
    vec4 wp = uModel * vec4(aPos, 1.0);
    vPos = wp.xyz;
    vN = uNormalMat * aNormal;
    vObjN = aNormal;
    vUV = aUV;
    gl_Position = uViewProj * wp;
    vFlogz = 1.0 + gl_Position.w;
}
