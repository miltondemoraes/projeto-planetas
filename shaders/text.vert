#version 330 core
layout(location = 0) in vec2 aPos;    // pixels
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aColor;
uniform vec2 uScreen;
out vec2 vUV;
out vec4 vColor;
void main() {
    vUV = aUV;
    vColor = aColor;
    gl_Position = vec4(aPos / uScreen * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 0.0, 1.0);
}
