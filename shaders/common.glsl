// ---------------------------------------------------------------------------
// common.glsl — constantes e utilidades compartilhadas
// Convenção: TODAS as posições estão em "espaço do mundo relativo à câmera"
// (a câmera está na origem). Isso evita perda de precisão de float32 com as
// distâncias enormes do sistema solar (técnica "floating origin").
// ---------------------------------------------------------------------------
const float PI = 3.14159265359;

// Profundidade logarítmica: z_buffer = log2(1 + w) / log2(1 + far)
// Distribui a precisão do z-buffer de forma ~uniforme em escala logarítmica,
// permitindo near = 1e-5 e far = 1e5 sem z-fighting.
uniform float uFcoef;   // 2 / log2(far + 1)
float logDepth(float flogz) { return log2(flogz) * uFcoef * 0.5; }

vec3 fresnelSchlick(vec3 F0, float cosT) {
    return F0 + (1.0 - F0) * pow(1.0 - clamp(cosT, 0.0, 1.0), 5.0);
}

// Blinn-Phong normalizado: (n+8)/8 conserva energia (lobo fica mais estreito E mais alto)
float blinnPhong(vec3 N, vec3 L, vec3 V, float shininess) {
    vec3 H = normalize(L + V);
    return (shininess + 8.0) / 8.0 * pow(max(dot(N, H), 0.0), shininess);
}

// Oren-Nayar (aprox. qualitativa): difuso para superfícies rugosas/poeirentas
// (Lua, Mercúrio). sigma = 0 -> Lambert puro. Retorna o termo já multiplicado por N·L.
float orenNayar(vec3 N, vec3 L, vec3 V, float sigma) {
    float NdotL = dot(N, L);
    if (NdotL <= 0.0) return 0.0;
    float NdotV = clamp(dot(N, V), 0.0, 1.0);
    if (sigma <= 0.0) return NdotL;
    float s2 = sigma * sigma;
    float A = 1.0 - 0.5 * s2 / (s2 + 0.33);
    float B = 0.45 * s2 / (s2 + 0.09);
    vec3 Lp = L - N * NdotL;
    vec3 Vp = V - N * NdotV;
    float lp = length(Lp), vp = length(Vp);
    float cosPhi = (lp > 1e-4 && vp > 1e-4) ? max(dot(Lp / lp, Vp / vp), 0.0) : 0.0;
    float ti = acos(clamp(NdotL, 0.0, 1.0));
    float tr = acos(NdotV);
    float alpha = max(ti, tr), beta = min(ti, tr);
    return NdotL * (A + B * cosPhi * sin(alpha) * tan(beta));
}

vec2 dirToEquirect(vec3 d) {
    return vec2(atan(d.x, d.z) / (2.0 * PI) + 0.5, asin(clamp(d.y, -1.0, 1.0)) / PI + 0.5);
}

// Ruído 3D (hash) usado em shaders procedurais (Sol, asteroides)
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}
float vnoise3(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash13(i + vec3(0,0,0)), hash13(i + vec3(1,0,0)), u.x),
                   mix(hash13(i + vec3(0,1,0)), hash13(i + vec3(1,1,0)), u.x), u.y),
               mix(mix(hash13(i + vec3(0,0,1)), hash13(i + vec3(1,0,1)), u.x),
                   mix(hash13(i + vec3(0,1,1)), hash13(i + vec3(1,1,1)), u.x), u.y), u.z);
}
float fbm3(vec3 p, int oct) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 8; i++) {
        if (i >= oct) break;
        s += a * vnoise3(p);
        p = p * 2.03 + 11.7;
        a *= 0.5;
    }
    return s;
}
