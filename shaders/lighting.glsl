// ---------------------------------------------------------------------------
// lighting.glsl — fontes de luz e SOMBRAS (usado pelos fragment shaders)
// ---------------------------------------------------------------------------
// Luz principal: o Sol, uma fonte ESFÉRICA (não pontual) -> sombras com penumbra.
uniform vec3  uSunPos;        // centro do Sol (relativo à câmera)
uniform float uSunRadius;
uniform vec3  uSunColor;
uniform int   uShadowsOn;

// --- 1) Sombras analíticas entre corpos esféricos (eclipses) --------------
// Para cada ponto P calculamos qual FRAÇÃO do disco solar está visível,
// considerando oclusores esféricos (planetas/luas). Os discos aparentes do
// Sol (raio angular rs) e do oclusor (rc), separados por ângulo d, se
// sobrepõem; a área da lente de interseção dá umbra/penumbra fisicamente
// corretas (inclusive eclipses anulares).
uniform vec4 uOcc[6];        // xyz = centro, w = raio
uniform int  uOccCount;

float discOverlapFrac(float r1, float r2, float d) {
    if (d >= r1 + r2) return 0.0;
    if (d <= abs(r1 - r2)) return (r2 >= r1) ? 1.0 : (r2 * r2) / (r1 * r1);
    float a = r1 * r1 * acos(clamp((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1), -1.0, 1.0));
    float b = r2 * r2 * acos(clamp((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2), -1.0, 1.0));
    float c = 0.5 * sqrt(max((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2), 0.0));
    return clamp((a + b - c) / (PI * r1 * r1), 0.0, 1.0);
}

float eclipseFactor(vec3 P) {
    if (uShadowsOn == 0) return 1.0;
    vec3 toS = uSunPos - P;
    float dS = length(toS);
    vec3 s = toS / dS;
    float rs = asin(min(uSunRadius / dS, 1.0));
    float vis = 1.0;
    for (int i = 0; i < 6; i++) {
        if (i >= uOccCount) break;
        vec3 toC = uOcc[i].xyz - P;
        float dC = length(toC);
        if (dC <= uOcc[i].w * 1.001 || dC >= dS) continue;   // ponto dentro do oclusor / oclusor atrás do Sol
        vec3 c = toC / dC;
        float cosd = dot(s, c);
        if (cosd <= 0.0) continue;
        float d = atan(length(cross(s, c)), cosd);           // mais preciso que acos para ângulos pequenos
        float rc = asin(min(uOcc[i].w / dC, 1.0));
        vis *= 1.0 - discOverlapFrac(rs, rc, d);
    }
    return vis;
}

// --- 2) Sombra dos anéis de Saturno (interseção raio-plano + textura) -----
uniform int   uRingOn;
uniform vec3  uRingCenter;
uniform vec3  uRingNormal;
uniform vec2  uRingRadii;    // raio interno/externo (unidades do mundo)
uniform sampler2D uRingTex;

float ringShadow(vec3 P, vec3 L) {
    if (uRingOn == 0 || uShadowsOn == 0) return 1.0;
    float denom = dot(L, uRingNormal);
    if (abs(denom) < 1e-5) return 1.0;
    float t = dot(uRingCenter - P, uRingNormal) / denom;
    if (t <= 0.0) return 1.0;
    float r = length(P + L * t - uRingCenter);
    if (r < uRingRadii.x || r > uRingRadii.y) return 1.0;
    float a = texture(uRingTex, vec2((r - uRingRadii.x) / (uRingRadii.y - uRingRadii.x), 0.5)).a;
    return 1.0 - a * 0.95;
}

// --- 3) Shadow mapping (sombra projetada pelo astronauta) -----------------
// Mapa de profundidade renderizado do ponto de vista do Sol com projeção
// ORTOGRÁFICA centrada no astronauta (o Sol é ~direcional nessa escala).
uniform sampler2DShadow uShadowMap;
uniform mat4  uLightVP;
uniform int   uShadowMapOn;
uniform float uShadowTexel;
uniform float uShadowNormalOffset;
uniform float uShadowBias;

float shadowMapFactor(vec3 P, vec3 Ng) {
    if (uShadowMapOn == 0 || uShadowsOn == 0) return 1.0;
    vec4 lp = uLightVP * vec4(P + Ng * uShadowNormalOffset, 1.0);
    vec3 c = lp.xyz / lp.w * 0.5 + 0.5;
    if (c.x <= 0.0 || c.x >= 1.0 || c.y <= 0.0 || c.y >= 1.0 || c.z >= 1.0 || c.z <= 0.0) return 1.0;
    // PCF 5x5: cada amostra já faz comparação + filtragem bilinear em hardware
    float s = 0.0;
    for (int i = -2; i <= 2; i++)
        for (int j = -2; j <= 2; j++)
            s += texture(uShadowMap, vec3(c.xy + vec2(i, j) * uShadowTexel, c.z - uShadowBias));
    s /= 25.0;
    // desvanece perto da borda do mapa
    vec2 e = min(c.xy, 1.0 - c.xy);
    return mix(1.0, s, smoothstep(0.0, 0.05, min(e.x, e.y)));
}

// --- Lanterna do capacete (luz SPOT) --------------------------------------
uniform int   uSpotOn;
uniform vec3  uSpotPos;
uniform vec3  uSpotDir;
uniform vec2  uSpotCos;      // cos(ângulo interno), cos(ângulo externo)
uniform vec3  uSpotColor;
uniform float uSpotRange;

vec3 spotLight(vec3 P, vec3 N, vec3 V, vec3 albedo, float shininess, float specK) {
    if (uSpotOn == 0) return vec3(0.0);
    vec3 Lv = uSpotPos - P;
    float d = length(Lv);
    if (d > uSpotRange) return vec3(0.0);
    vec3 L = Lv / d;
    float cone = smoothstep(uSpotCos.y, uSpotCos.x, dot(-L, uSpotDir));
    float win = pow(clamp(1.0 - pow(d / uSpotRange, 4.0), 0.0, 1.0), 2.0);   // janela suave
    float atten = win / (d * d + 1e-8);                                       // inverso do quadrado
    float NdotL = max(dot(N, L), 0.0);
    vec3 c = albedo * NdotL + specK * blinnPhong(N, L, V, shininess) * NdotL;
    return c * uSpotColor * atten * cone;
}
