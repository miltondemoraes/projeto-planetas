#version 330 core
// Bloom (cadeia de mips) — DOWNSAMPLE com filtro de 13 amostras (Jimenez, CoD:AW 2014)
in vec2 vUV;
uniform sampler2D uSrc;
uniform vec2 uTexel;     // 1 / resolução da ORIGEM
uniform int uFirst;      // no 1º passo: média de Karis (evita "vagalumes" de pixels muito brilhantes)
out vec4 fragColor;

vec3 karis(vec3 a, vec3 b, vec3 c, vec3 d) {
    float wa = 1.0 / (1.0 + max(a.r, max(a.g, a.b)));
    float wb = 1.0 / (1.0 + max(b.r, max(b.g, b.b)));
    float wc = 1.0 / (1.0 + max(c.r, max(c.g, c.b)));
    float wd = 1.0 / (1.0 + max(d.r, max(d.g, d.b)));
    return (a * wa + b * wb + c * wc + d * wd) / (wa + wb + wc + wd);
}

void main() {
    vec2 t = uTexel;
    vec3 a = texture(uSrc, vUV + t * vec2(-2, 2)).rgb;
    vec3 b = texture(uSrc, vUV + t * vec2(0, 2)).rgb;
    vec3 c = texture(uSrc, vUV + t * vec2(2, 2)).rgb;
    vec3 d = texture(uSrc, vUV + t * vec2(-2, 0)).rgb;
    vec3 e = texture(uSrc, vUV).rgb;
    vec3 f = texture(uSrc, vUV + t * vec2(2, 0)).rgb;
    vec3 g = texture(uSrc, vUV + t * vec2(-2, -2)).rgb;
    vec3 h = texture(uSrc, vUV + t * vec2(0, -2)).rgb;
    vec3 i = texture(uSrc, vUV + t * vec2(2, -2)).rgb;
    vec3 j = texture(uSrc, vUV + t * vec2(-1, 1)).rgb;
    vec3 k = texture(uSrc, vUV + t * vec2(1, 1)).rgb;
    vec3 l = texture(uSrc, vUV + t * vec2(-1, -1)).rgb;
    vec3 m = texture(uSrc, vUV + t * vec2(1, -1)).rgb;
    vec3 r;
    if (uFirst == 1) {
        r = karis(j, k, l, m) * 0.5 + karis(a, b, d, e) * 0.125 + karis(b, c, e, f) * 0.125
          + karis(d, e, g, h) * 0.125 + karis(e, f, h, i) * 0.125;
    } else {
        r = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
    }
    fragColor = vec4(max(r, 0.0), 1.0);
}
