#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uModel;
uniform mat4 uViewProj;
out float vFlogz;
void main() {
    gl_Position = uViewProj * uModel * vec4(aPos, 1.0);
    vFlogz = 1.0 + gl_Position.w;
}
