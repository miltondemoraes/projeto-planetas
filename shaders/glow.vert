#version 330 core
// Billboard (quad sempre virado para a câmera) gerado sem VBO via gl_VertexID
uniform vec3 uCenter;
uniform vec3 uRight;
uniform vec3 uUp;
uniform float uSize;
uniform mat4 uViewProj;
out vec2 vC;
out float vFlogz;
void main() {
    vec2 c = vec2((gl_VertexID & 1) == 0 ? -1.0 : 1.0, (gl_VertexID & 2) == 0 ? -1.0 : 1.0);
    vC = c;
    vec3 p = uCenter + (uRight * c.x + uUp * c.y) * uSize;
    gl_Position = uViewProj * vec4(p, 1.0);
    vFlogz = 1.0 + gl_Position.w;
}
