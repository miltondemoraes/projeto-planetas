#version 330 core
in vec3 vColor;
out vec4 fragColor;
void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    float a = exp(-r2 * 4.0);           // perfil gaussiano (PSF)
    fragColor = vec4(vColor * a, 1.0);
}
