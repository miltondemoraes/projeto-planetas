#version 330 core
// Estrelas como GL_POINTS no infinito (só a rotação da câmera importa)
layout(location = 0) in vec3 aDir;
layout(location = 1) in vec3 aColor;    // cor × fluxo (escala de magnitudes)
uniform mat4 uViewProj;
uniform float uPointScale;
uniform float uStarIntensity;
out vec3 vColor;
void main() {
    vec4 p = uViewProj * vec4(aDir, 1.0);
    gl_Position = p.xyww;                 // z = w -> profundidade máxima
    float flux = max(max(aColor.r, aColor.g), aColor.b);
    float size = clamp(1.6 + sqrt(flux) * 2.2, 1.5, 6.0) * uPointScale;
    gl_PointSize = size;
    vColor = aColor * uStarIntensity * (2.5 / (size * size * 0.35));
}
