#version 330 core
// Bloom — UPSAMPLE com filtro tenda 3x3, somado (blend aditivo) ao mip de cima
in vec2 vUV;
uniform sampler2D uSrc;
uniform vec2 uTexel;
uniform float uRadius;
out vec4 fragColor;
void main() {
    vec2 t = uTexel * uRadius;
    vec3 s = texture(uSrc, vUV).rgb * 4.0;
    s += (texture(uSrc, vUV + vec2(-t.x, 0)).rgb + texture(uSrc, vUV + vec2(t.x, 0)).rgb
        + texture(uSrc, vUV + vec2(0, -t.y)).rgb + texture(uSrc, vUV + vec2(0, t.y)).rgb) * 2.0;
    s += texture(uSrc, vUV + vec2(-t.x, -t.y)).rgb + texture(uSrc, vUV + vec2(t.x, -t.y)).rgb
       + texture(uSrc, vUV + vec2(-t.x, t.y)).rgb + texture(uSrc, vUV + vec2(t.x, t.y)).rgb;
    fragColor = vec4(s / 16.0, 1.0);
}
