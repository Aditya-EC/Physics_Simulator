"""GLSL 3.30 core shader sources."""

MESH_VS = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
uniform mat4 uModel;
uniform mat4 uViewProj;
uniform mat3 uNormalMat;
uniform mat4 uLightVP;
out vec3 vWorld;
out vec3 vNormal;
out vec3 vColor;
out vec4 vLightPos;
void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    vWorld = world.xyz;
    vNormal = uNormalMat * aNormal;
    vColor = aColor;
    vLightPos = uLightVP * world;
    gl_Position = uViewProj * world;
}
"""

MESH_FS = """
#version 330 core
in vec3 vWorld;
in vec3 vNormal;
in vec3 vColor;
in vec4 vLightPos;
out vec4 fragColor;

uniform vec3 uBaseColor;
uniform bool uUseVertexColor;
uniform float uOpacity;
uniform float uMetallic;
uniform float uRoughness;
uniform vec3 uCamPos;
uniform vec3 uLightDir;      // unit vector pointing towards the light
uniform float uLightIntensity;
uniform float uAmbient;
uniform vec3 uSkyColor;
uniform vec3 uGroundColor;
uniform float uEnvStrength;
uniform bool uShadows;
uniform sampler2D uShadowMap;
uniform float uShadowTexel;
uniform vec3 uHighlight;
uniform float uHighlightAmount;
uniform bool uUnlit;

const float PI = 3.14159265;

float shadowFactor(vec3 N) {
    if (!uShadows) return 1.0;
    vec3 p = vLightPos.xyz / vLightPos.w * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0) return 1.0;
    float ndl = max(dot(N, uLightDir), 0.0);
    float bias = max(0.0025 * (1.0 - ndl), 0.0006);
    float lit = 0.0;
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y) {
            float d = texture(uShadowMap, p.xy + vec2(x, y) * uShadowTexel).r;
            lit += (p.z - bias) > d ? 0.0 : 1.0;
        }
    return lit / 9.0;
}

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec3 base = uUseVertexColor ? vColor : uBaseColor;
    base = pow(base, vec3(2.2));
    if (uUnlit) {
        fragColor = vec4(pow(base, vec3(1.0 / 2.2)), uOpacity);
        return;
    }
    vec3 N = normalize(vNormal);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(uCamPos - vWorld);
    float rough = clamp(uRoughness, 0.04, 1.0);
    float a = rough * rough;
    vec3 F0 = mix(vec3(0.04), base, uMetallic);
    vec3 diffuseColor = base * (1.0 - uMetallic);

    // key light (GGX specular + Lambert diffuse)
    vec3 L = uLightDir;
    vec3 H = normalize(L + V);
    float NdL = max(dot(N, L), 0.0);
    float NdV = max(dot(N, V), 1e-4);
    float NdH = max(dot(N, H), 0.0);
    float VdH = max(dot(V, H), 0.0);
    float a2 = a * a;
    float dn = NdH * NdH * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * dn * dn);
    float k = (rough + 1.0) * (rough + 1.0) / 8.0;
    float G = (NdV / (NdV * (1.0 - k) + k)) * (NdL / (NdL * (1.0 - k) + k));
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdH, 5.0);
    vec3 spec = D * G * F / max(4.0 * NdV * NdL, 1e-4);
    vec3 kd = (1.0 - F) * (1.0 - uMetallic);
    float sh = shadowFactor(N);
    vec3 color = (kd * diffuseColor / PI + spec) * NdL * uLightIntensity * sh;

    // camera-side fill light (no specular) keeps unlit faces readable
    vec3 Lf = normalize(V + vec3(0.0, 0.0, 0.35));
    color += diffuseColor * max(dot(N, Lf), 0.0) * 0.22 * uLightIntensity / PI;

    // hemisphere ambient + simple environment reflection
    float hemi = N.z * 0.5 + 0.5;
    vec3 ambient = mix(uGroundColor, uSkyColor, hemi);
    color += diffuseColor * ambient * uAmbient;
    vec3 R = reflect(-V, N);
    // blurred two-tone environment: sharper horizon for smooth surfaces
    float horizon = mix(0.08, 0.6, rough);
    vec3 env = mix(uGroundColor, uSkyColor, smoothstep(-horizon, horizon, R.z));
    vec3 Fenv = F0 + (max(vec3(1.0 - rough), F0) - F0) * pow(1.0 - NdV, 5.0);
    color += env * Fenv * uEnvStrength * (1.0 - 0.45 * rough) * (0.65 + 0.35 * sh);

    // selection / hover highlight (rim)
    if (uHighlightAmount > 0.0) {
        float rim = pow(1.0 - NdV, 2.0);
        vec3 hl = pow(uHighlight, vec3(2.2));
        color = mix(color, hl * 1.4, clamp(rim * 0.9 * uHighlightAmount, 0.0, 1.0));
        color += hl * 0.06 * uHighlightAmount;
    }

    color = aces(color * 1.1);
    fragColor = vec4(pow(color, vec3(1.0 / 2.2)), uOpacity);
}
"""

DEPTH_VS = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uLightMVP;
void main() { gl_Position = uLightMVP * vec4(aPos, 1.0); }
"""

DEPTH_FS = """
#version 330 core
void main() {}
"""

LINE_VS = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
uniform mat4 uMVP;
uniform mat4 uModel;
out vec4 vColor;
out vec3 vWorld;
void main() {
    vColor = aColor;
    vWorld = (uModel * vec4(aPos, 1.0)).xyz;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

# Expands each line segment into a screen-aligned quad of uWidth pixels
# (core-profile OpenGL does not guarantee wide lines). Segments are clipped
# against the near plane first.
LINE_GS = """
#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
uniform vec2 uViewport;
uniform float uWidth;
in vec4 vColor[];
in vec3 vWorld[];
out vec4 gColor;
out vec3 gWorld;
out float gEdge;
void main() {
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    vec4 c0 = vColor[0], c1 = vColor[1];
    vec3 w0 = vWorld[0], w1 = vWorld[1];
    float d0 = p0.z + p0.w;
    float d1 = p1.z + p1.w;
    if (d0 < 0.0 && d1 < 0.0) return;
    if (d0 < 0.0) { float t = d0 / (d0 - d1); p0 = mix(p0, p1, t); c0 = mix(c0, c1, t); w0 = mix(w0, w1, t); }
    if (d1 < 0.0) { float t = d1 / (d1 - d0); p1 = mix(p1, p0, t); c1 = mix(c1, c0, t); w1 = mix(w1, w0, t); }
    vec2 s0 = p0.xy / p0.w * uViewport * 0.5;
    vec2 s1 = p1.xy / p1.w * uViewport * 0.5;
    vec2 dir = s1 - s0;
    float len = length(dir);
    dir = len > 1e-6 ? dir / len : vec2(1.0, 0.0);
    vec2 n = vec2(-dir.y, dir.x) * (uWidth * 0.5 + 0.5);
    vec2 o0 = n / (uViewport * 0.5) * p0.w;
    vec2 o1 = n / (uViewport * 0.5) * p1.w;
    gColor = c0; gWorld = w0; gEdge = 1.0;  gl_Position = vec4(p0.xy + o0, p0.zw); EmitVertex();
    gColor = c0; gWorld = w0; gEdge = -1.0; gl_Position = vec4(p0.xy - o0, p0.zw); EmitVertex();
    gColor = c1; gWorld = w1; gEdge = 1.0;  gl_Position = vec4(p1.xy + o1, p1.zw); EmitVertex();
    gColor = c1; gWorld = w1; gEdge = -1.0; gl_Position = vec4(p1.xy - o1, p1.zw); EmitVertex();
    EndPrimitive();
}
"""

LINE_FS = """
#version 330 core
in vec4 gColor;
in vec3 gWorld;
in float gEdge;
out vec4 fragColor;
uniform float uWidth;
uniform vec3 uFadeCenter;
uniform float uFadeRadius;   // <= 0 disables distance fading
void main() {
    float px = (uWidth * 0.5 + 0.5);
    float dist = abs(gEdge) * px;
    float aa = clamp(px - dist + 0.5, 0.0, 1.0);   // 1-pixel anti-aliased edge
    float fade = 1.0;
    if (uFadeRadius > 0.0) {
        float r = length(gWorld - uFadeCenter);
        fade = 1.0 - smoothstep(uFadeRadius * 0.55, uFadeRadius, r);
    }
    fragColor = vec4(gColor.rgb, gColor.a * aa * fade);
    if (fragColor.a <= 0.003) discard;
}
"""

BACKGROUND_VS = """
#version 330 core
out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.999999, 1.0);
}
"""

BACKGROUND_FS = """
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform vec3 uTop;
uniform vec3 uBottom;
void main() {
    float t = clamp(vUV.y, 0.0, 1.0);
    vec3 c = mix(uBottom, uTop, smoothstep(0.0, 1.0, t));
    // subtle vignette
    vec2 d = vUV - 0.5;
    c *= 1.0 - 0.18 * dot(d, d);
    fragColor = vec4(c, 1.0);
}
"""

SHADOW_CATCHER_FS = """
#version 330 core
in vec3 vWorld;
in vec3 vNormal;
in vec3 vColor;
in vec4 vLightPos;
out vec4 fragColor;
uniform sampler2D uShadowMap;
uniform float uShadowTexel;
uniform float uStrength;
uniform vec3 uFadeCenter;
uniform float uFadeRadius;
void main() {
    vec3 p = vLightPos.xyz / vLightPos.w * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0) discard;
    float lit = 0.0;
    for (int x = -2; x <= 2; ++x)
        for (int y = -2; y <= 2; ++y) {
            float d = texture(uShadowMap, p.xy + vec2(x, y) * uShadowTexel).r;
            lit += (p.z - 0.0015) > d ? 0.0 : 1.0;
        }
    float shade = 1.0 - lit / 25.0;
    float fade = 1.0 - smoothstep(uFadeRadius * 0.5, uFadeRadius, length(vWorld - uFadeCenter));
    if (shade * fade <= 0.002) discard;
    fragColor = vec4(0.0, 0.0, 0.0, shade * uStrength * fade);
}
"""
