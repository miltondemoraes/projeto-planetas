#version 330 core
// Pós-processamento final: HDR + bloom -> exposição -> tone mapping ACES -> gama sRGB
in vec2 vUV;
uniform sampler2D uHDR;
uniform sampler2D uBloom;
uniform float uExposure;
uniform float uBloomStrength;
uniform int uBloomOn;
uniform int uTonemapOn;
out vec4 fragColor;

vec3 aces(vec3 x) {   // aproximação de Narkowicz para o ACES filmic
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
vec3 linearToSrgb(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
float ign(vec2 p) {   // ruído de gradiente intercalado: dithering contra banding
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

void main() {
    vec3 hdr = texture(uHDR, vUV).rgb;
    if (uBloomOn == 1) hdr = mix(hdr, texture(uBloom, vUV).rgb, uBloomStrength);
    vec3 c = hdr * uExposure;
    c = (uTonemapOn == 1) ? aces(c) : clamp(c, 0.0, 1.0);
    vec2 q = vUV - 0.5;
    c *= 1.0 - 0.25 * dot(q, q);                 // vinheta suave (lente)
    c = linearToSrgb(c);
    c += (ign(gl_FragCoord.xy) - 0.5) / 255.0;
    fragColor = vec4(c, 1.0);
}
