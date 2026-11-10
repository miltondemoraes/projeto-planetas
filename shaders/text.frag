#version 330 core
in vec2 vUV;
in vec4 vColor;
uniform sampler2D uAtlas;
out vec4 fragColor;
void main() {
    float a = texture(uAtlas, vUV).r;
    fragColor = vec4(vColor.rgb, vColor.a * a);
}
