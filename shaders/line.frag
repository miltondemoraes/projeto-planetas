#version 330 core
#include "common.glsl"
in float vFlogz;
uniform vec4 uColor;
out vec4 fragColor;
void main() {
    fragColor = uColor;
    gl_FragDepth = logDepth(vFlogz);
}
