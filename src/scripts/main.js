// ============================================
// Canvas
// ============================================
const canvas = document.getElementById("canvas");

// ============================================
// State
// ============================================
const state = {
  cellSize: 18,
  waveAmplitude: 0.5,
  waveSpeed: 1.0,
  noiseIntensity: 0.125,
  vignetteIntensity: 0.5,
  vignetteRadius: 0.5,
  brightnessAdjust: 0.0,
  contrastAdjust: 1.25,
  timeSpeed: 1.5,
  hue: 180,
  saturation: 0.5,
  threshold1: 0.25,
  threshold2: 0.3,
  threshold3: 0.4,
  threshold4: 0.5,
  threshold5: 0.65,
  noiseSeed: Math.random().toString(36).substring(2, 8),
};

function hashSeed(seed) {
  if (!seed) return 0;
  let hash = 0;
  for (let i = 0; i < seed.length; i++) {
    const char = seed.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }
  return Math.abs(hash) % 10000;
}

// ============================================
// Presets
// ============================================
const presets = {
  default: {
    cellSize: 10,
    waveAmplitude: 0.15,
    waveSpeed: 0.5,
    noiseIntensity: 0.0,
    vignetteIntensity: 0.5,
    vignetteRadius: 0.5,
    brightnessAdjust: 0.0,
    contrastAdjust: 1.0,
    timeSpeed: 1.0,
    hue: 180,
    saturation: 0.5,
    threshold1: 0.25,
    threshold2: 0.3,
    threshold3: 0.4,
    threshold4: 0.5,
    threshold5: 0.65,
  },
  terminal: {
    cellSize: 12,
    waveAmplitude: 0.2,
    waveSpeed: 0.4,
    noiseIntensity: 0.025,
    vignetteIntensity: 0.65,
    vignetteRadius: 0.5,
    brightnessAdjust: -0.15,
    contrastAdjust: 1.5,
    timeSpeed: 0.8,
    hue: 120,
    saturation: 0.9,
    threshold1: 0.08,
    threshold2: 0.18,
    threshold3: 0.28,
    threshold4: 0.38,
    threshold5: 0.48,
  },
  retro: {
    cellSize: 16,
    waveAmplitude: 0.15,
    waveSpeed: 0.6,
    noiseIntensity: 0.035,
    vignetteIntensity: 0.8,
    vignetteRadius: 0.6,
    brightnessAdjust: -0.05,
    contrastAdjust: 1.5,
    timeSpeed: 0.7,
    hue: 35,
    saturation: 0.85,
    threshold1: 0.15,
    threshold2: 0.3,
    threshold3: 0.5,
    threshold4: 0.7,
    threshold5: 1.0,
  },
  cosmic: {
    cellSize: 6,
    waveAmplitude: 0.5,
    waveSpeed: 0.5,
    noiseIntensity: 0.02,
    vignetteIntensity: 0.4,
    vignetteRadius: 0.8,
    brightnessAdjust: 0.0,
    contrastAdjust: 1,
    timeSpeed: 1.0,
    hue: 270,
    saturation: 0.7,
    threshold1: 0.08,
    threshold2: 0.18,
    threshold3: 0.3,
    threshold4: 0.44,
    threshold5: 0.58,
  },
  fog: {
    cellSize: 8,
    waveAmplitude: 0.7,
    waveSpeed: 0.5,
    noiseIntensity: 0.05,
    vignetteIntensity: 0.8,
    vignetteRadius: 0.75,
    brightnessAdjust: 0.2,
    contrastAdjust: 1,
    timeSpeed: 0.5,
    hue: 200,
    saturation: 0.3,
    threshold1: 0.15,
    threshold2: 0.28,
    threshold3: 0.4,
    threshold4: 0.52,
    threshold5: 0.65,
  },
  red: {
    cellSize: 9.477363992838812,
    waveAmplitude: 0.19866627466604492,
    waveSpeed: 0.6728127000674906,
    noiseIntensity: 0.08127384570359468,
    vignetteIntensity: 0.9012567147515466,
    vignetteRadius: 0.6647325291352247,
    brightnessAdjust: -0.0859122219113787,
    contrastAdjust: 0.6656278165227767,
    timeSpeed: 1.1643690924738626,
    hue: 352.8274855467056,
    saturation: 0.8534068514321124,
    threshold1: 0.1679224645262462,
    threshold2: 0.4497505323208811,
    threshold3: 0.6843052202787598,
    threshold4: 0.658074487889079,
    threshold5: 0.7430026789072516,
  },
};

let isPaused = false;
let autoPaused = false;
let animationFrameId = null;
let time = 0;

// ============================================
// WebGL Setup
// ============================================
const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true });
if (!gl) {
  throw new Error("WebGL 2.0 not supported");
}

const vertexShaderSource = `#version 300 es
  in vec2 a_position;
  out vec2 v_uv;

  void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
  }
`;

const noiseFragmentShaderSource = `#version 300 es
  precision highp float;

  in vec2 v_uv;
  out vec4 fragColor;

  uniform float u_time;
  uniform vec2 u_resolution;
  uniform float u_waveAmplitude;
  uniform float u_waveSpeed;
  uniform float u_noiseIntensity;
  uniform float u_vignetteIntensity;
  uniform float u_vignetteRadius;
  uniform float u_brightnessAdjust;
  uniform float u_contrastAdjust;
  uniform float u_noiseSeed;

  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

  float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);

    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;

    i = mod289(i);
    vec4 p = permute(permute(permute(
              i.z + vec4(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0));

    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);

    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;

    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);

    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
  }

  float fbm(vec3 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for (int i = 0; i < 4; i++) {
      value += amplitude * snoise(p * frequency);
      amplitude *= 0.5;
      frequency *= 2.0;
    }
    return value;
  }

  void main() {
    vec2 center = vec2(0.5, 0.5);
    float dist = length(v_uv - center);

    float aspect = u_resolution.x / u_resolution.y;
    vec2 uv = v_uv;
    uv.x *= aspect;

    vec2 drift = u_time * (0.02 + 0.02 * u_waveSpeed) * vec2(0.3, 0.2);

    float warpTime = u_time * max(0.025, 0.04 * u_waveSpeed);

    vec2 q = vec2(
      fbm(vec3(uv + drift, warpTime + u_noiseSeed)),
      fbm(vec3(uv + drift + vec2(5.2, 1.3), warpTime * 0.9 + u_noiseSeed))
    );

    vec2 driftR = u_time * (0.016 + 0.016 * u_waveSpeed) * vec2(0.25, 0.15);
    vec2 r = vec2(
      fbm(vec3(uv + 4.0 * q + vec2(1.7, 9.2) + driftR, warpTime * 0.8 + u_noiseSeed)),
      fbm(vec3(uv + 4.0 * q + vec2(8.3, 2.8) + driftR, warpTime * 0.7 + u_noiseSeed))
    );

    float warpStrength = u_waveAmplitude * 1.5;
    vec2 warpedUV = uv + warpStrength * r + drift;

    float density = fbm(vec3(warpedUV * 4.0, warpTime * 0.5 + u_noiseSeed)) * 0.5 + 0.5;

    density += (snoise(vec3(uv * 50.0 + drift * 10.0, u_noiseSeed)) * 0.5 + 0.5) * u_noiseIntensity;

    float visible = smoothstep(0.35, 0.70, density);

    float edgeFade = 1.0 - smoothstep(u_vignetteRadius * 0.5, u_vignetteRadius, dist) * u_vignetteIntensity;
    visible *= edgeFade;

    visible = (visible + u_brightnessAdjust) * u_contrastAdjust;
    visible = clamp(visible, 0.0, 1.0);

    fragColor = vec4(vec3(visible), 1.0);
  }
`;

const glyphFragmentShaderSource = `#version 300 es
  precision highp float;

  in vec2 v_uv;
  out vec4 fragColor;

  uniform sampler2D u_noiseTexture;
  uniform vec2 u_resolution;
  uniform float u_cellSize;
  uniform float u_hue;
  uniform float u_saturation;
  uniform float u_threshold1;
  uniform float u_threshold2;
  uniform float u_threshold3;
  uniform float u_threshold4;
  uniform float u_threshold5;

  vec3 hsl2rgb(float h, float s, float l) {
    float c = (1.0 - abs(2.0 * l - 1.0)) * s;
    float hp = h / 60.0;
    float x = c * (1.0 - abs(mod(hp, 2.0) - 1.0));
    vec3 rgb;
    if (hp < 1.0) rgb = vec3(c, x, 0.0);
    else if (hp < 2.0) rgb = vec3(x, c, 0.0);
    else if (hp < 3.0) rgb = vec3(0.0, c, x);
    else if (hp < 4.0) rgb = vec3(0.0, x, c);
    else if (hp < 5.0) rgb = vec3(x, 0.0, c);
    else rgb = vec3(c, 0.0, x);
    float m = l - c * 0.5;
    return rgb + m;
  }

  float drawDot(vec2 uv) {
    vec2 center = vec2(0.5, 0.5);
    float dist = length(uv - center);
    return smoothstep(0.2, 0.15, dist);
  }

  float drawDash(vec2 uv) {
    float h = smoothstep(0.35, 0.4, uv.y) * smoothstep(0.65, 0.6, uv.y);
    float w = smoothstep(0.15, 0.2, uv.x) * smoothstep(0.85, 0.8, uv.x);
    return h * w;
  }

  float drawPlus(vec2 uv) {
    float horiz = smoothstep(0.35, 0.4, uv.y) * smoothstep(0.65, 0.6, uv.y) *
                  smoothstep(0.1, 0.15, uv.x) * smoothstep(0.9, 0.85, uv.x);
    float vert = smoothstep(0.35, 0.4, uv.x) * smoothstep(0.65, 0.6, uv.x) *
                 smoothstep(0.1, 0.15, uv.y) * smoothstep(0.9, 0.85, uv.y);
    return max(horiz, vert);
  }

  float drawO(vec2 uv) {
    vec2 center = vec2(0.5, 0.5);
    float dist = length(uv - center);
    float outer = smoothstep(0.4, 0.35, dist);
    float inner = smoothstep(0.2, 0.25, dist);
    return outer * inner;
  }

  float drawX(vec2 uv) {
    vec2 c = uv - 0.5;
    float d1 = abs(c.x - c.y);
    float d2 = abs(c.x + c.y);
    float line1 = smoothstep(0.15, 0.1, d1);
    float line2 = smoothstep(0.15, 0.1, d2);
    float bounds = smoothstep(0.45, 0.4, abs(c.x)) * smoothstep(0.45, 0.4, abs(c.y));
    return max(line1, line2) * bounds;
  }

  float getGlyph(float brightness, vec2 localUV) {
    if (brightness < u_threshold1) {
      return 0.0;
    } else if (brightness < u_threshold2) {
      return drawDot(localUV);
    } else if (brightness < u_threshold3) {
      return drawDash(localUV);
    } else if (brightness < u_threshold4) {
      return drawPlus(localUV);
    } else if (brightness < u_threshold5) {
      return drawO(localUV);
    } else {
      return drawX(localUV);
    }
  }

  void main() {
    vec2 cellCount = u_resolution / u_cellSize;
    vec2 cellCoord = floor(v_uv * cellCount);
    vec2 cellUV = (cellCoord + 0.5) / cellCount;

    float brightness = texture(u_noiseTexture, cellUV).r;

    vec2 localUV = fract(v_uv * cellCount);

    float glyphValue = getGlyph(brightness, localUV);

    vec3 glyphColor = hsl2rgb(u_hue, u_saturation, 0.5 + brightness * 0.3);
    vec3 bgColor = vec3(0.02, 0.02, 0.02);

    vec3 finalColor = mix(bgColor, glyphColor, glyphValue * brightness);

    fragColor = vec4(finalColor, 1.0);
  }
`;

function compileShader(type, source) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Failed to create shader");

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error("Shader compilation error: " + info);
  }

  return shader;
}

function createProgram(vertSource, fragSource) {
  const vertShader = compileShader(gl.VERTEX_SHADER, vertSource);
  const fragShader = compileShader(gl.FRAGMENT_SHADER, fragSource);

  const program = gl.createProgram();
  if (!program) throw new Error("Failed to create program");

  gl.attachShader(program, vertShader);
  gl.attachShader(program, fragShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    throw new Error("Program link error: " + info);
  }

  return program;
}

const noiseProgram = createProgram(
  vertexShaderSource,
  noiseFragmentShaderSource,
);
const glyphProgram = createProgram(
  vertexShaderSource,
  glyphFragmentShaderSource,
);

const noiseUniforms = {
  time: gl.getUniformLocation(noiseProgram, "u_time"),
  resolution: gl.getUniformLocation(noiseProgram, "u_resolution"),
  waveAmplitude: gl.getUniformLocation(noiseProgram, "u_waveAmplitude"),
  waveSpeed: gl.getUniformLocation(noiseProgram, "u_waveSpeed"),
  noiseIntensity: gl.getUniformLocation(noiseProgram, "u_noiseIntensity"),
  vignetteIntensity: gl.getUniformLocation(noiseProgram, "u_vignetteIntensity"),
  vignetteRadius: gl.getUniformLocation(noiseProgram, "u_vignetteRadius"),
  brightnessAdjust: gl.getUniformLocation(noiseProgram, "u_brightnessAdjust"),
  contrastAdjust: gl.getUniformLocation(noiseProgram, "u_contrastAdjust"),
  noiseSeed: gl.getUniformLocation(noiseProgram, "u_noiseSeed"),
};

const glyphUniforms = {
  noiseTexture: gl.getUniformLocation(glyphProgram, "u_noiseTexture"),
  resolution: gl.getUniformLocation(glyphProgram, "u_resolution"),
  cellSize: gl.getUniformLocation(glyphProgram, "u_cellSize"),
  hue: gl.getUniformLocation(glyphProgram, "u_hue"),
  saturation: gl.getUniformLocation(glyphProgram, "u_saturation"),
  threshold1: gl.getUniformLocation(glyphProgram, "u_threshold1"),
  threshold2: gl.getUniformLocation(glyphProgram, "u_threshold2"),
  threshold3: gl.getUniformLocation(glyphProgram, "u_threshold3"),
  threshold4: gl.getUniformLocation(glyphProgram, "u_threshold4"),
  threshold5: gl.getUniformLocation(glyphProgram, "u_threshold5"),
};

const quadBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
gl.bufferData(
  gl.ARRAY_BUFFER,
  new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
  gl.STATIC_DRAW,
);

const quadVAO = gl.createVertexArray();
gl.bindVertexArray(quadVAO);

const positionLoc = gl.getAttribLocation(noiseProgram, "a_position");
gl.enableVertexAttribArray(positionLoc);
gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

let framebuffer = null;
let noiseTexture = null;

function createFramebuffer(width, height) {
  if (framebuffer) gl.deleteFramebuffer(framebuffer);
  if (noiseTexture) gl.deleteTexture(noiseTexture);

  noiseTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, noiseTexture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    width,
    height,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    null,
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(
    gl.FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    noiseTexture,
    0,
  );

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

// ============================================
// Resize
// ============================================
function resize() {
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const rect = canvas.getBoundingClientRect();
  const width = Math.floor(rect.width * dpr);
  const height = Math.floor(rect.height * dpr);

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    createFramebuffer(width, height);
  }
}

resize();
window.addEventListener("resize", () => {
  resize();
  if (isPaused) renderFrame();
});

// ============================================
// Render Loop
// ============================================
let lastTime = 0;

function renderFrame() {
  resize();

  const width = canvas.width;
  const height = canvas.height;

  // Pass 1: Render noise to framebuffer
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.viewport(0, 0, width, height);
  gl.useProgram(noiseProgram);

  gl.uniform1f(noiseUniforms.time, time);
  gl.uniform2f(noiseUniforms.resolution, width, height);
  gl.uniform1f(noiseUniforms.waveAmplitude, state.waveAmplitude);
  gl.uniform1f(noiseUniforms.waveSpeed, state.waveSpeed);
  gl.uniform1f(noiseUniforms.noiseIntensity, state.noiseIntensity);
  gl.uniform1f(noiseUniforms.vignetteIntensity, state.vignetteIntensity);
  gl.uniform1f(noiseUniforms.vignetteRadius, state.vignetteRadius);
  gl.uniform1f(noiseUniforms.brightnessAdjust, state.brightnessAdjust);
  gl.uniform1f(noiseUniforms.contrastAdjust, state.contrastAdjust);
  gl.uniform1f(noiseUniforms.noiseSeed, hashSeed(state.noiseSeed));

  gl.bindVertexArray(quadVAO);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  // Pass 2: Render glyphs to screen
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, width, height);
  gl.useProgram(glyphProgram);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, noiseTexture);
  gl.uniform1i(glyphUniforms.noiseTexture, 0);

  gl.uniform2f(glyphUniforms.resolution, width, height);
  gl.uniform1f(
    glyphUniforms.cellSize,
    state.cellSize * (window.devicePixelRatio || 1),
  );
  gl.uniform1f(glyphUniforms.hue, state.hue);
  gl.uniform1f(glyphUniforms.saturation, state.saturation);
  gl.uniform1f(glyphUniforms.threshold1, state.threshold1);
  gl.uniform1f(glyphUniforms.threshold2, state.threshold2);
  gl.uniform1f(glyphUniforms.threshold3, state.threshold3);
  gl.uniform1f(glyphUniforms.threshold4, state.threshold4);
  gl.uniform1f(glyphUniforms.threshold5, state.threshold5);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

function render(currentTime) {
  const deltaTime = (currentTime - lastTime) / 1000;
  lastTime = currentTime;

  if (!isPaused) {
    time += deltaTime * state.timeSpeed;
  }

  renderFrame();
  animationFrameId = requestAnimationFrame(render);
}

function setPaused(paused) {
  if (isPaused === paused) return;
  isPaused = paused;

  if (isPaused) {
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
  } else {
    lastTime = performance.now();
    animationFrameId = requestAnimationFrame(render);
  }
}

// Apply the default preset on startup
Object.assign(state, presets.default);
animationFrameId = requestAnimationFrame(render);

// Auto-pause when tab hidden to save resources
function checkAutoPause() {
  const shouldPause = document.hidden || !document.hasFocus();

  if (shouldPause && !isPaused) {
    autoPaused = true;
    setPaused(true);
  } else if (!shouldPause && autoPaused) {
    autoPaused = false;
    setPaused(false);
  }
}

document.addEventListener("visibilitychange", checkAutoPause);
window.addEventListener("blur", checkAutoPause);
window.addEventListener("focus", checkAutoPause);
checkAutoPause();

// ============================================
// Public API — window.AsciiClouds
// ============================================
window.AsciiClouds = {
  /** Current state object (read/write properties directly) */
  state,

  /** Available built-in presets */
  presets,

  /**
   * Update one or more options. Changes take effect on the next frame.
   * @param {Partial<typeof state>} options
   */
  set(options) {
    Object.assign(state, options);
    if (isPaused) renderFrame();
  },

  /** Apply a built-in preset by name */
  applyPreset(name) {
    const preset = presets[name];
    if (!preset) throw new Error(`Unknown preset: "${name}"`);
    Object.assign(state, preset);
    if (isPaused) renderFrame();
  },

  /** Pause the animation */
  pause() {
    setPaused(true);
    autoPaused = false;
  },

  /** Resume the animation */
  resume() {
    autoPaused = false;
    setPaused(false);
  },

  /** Whether the animation is currently paused */
  get paused() {
    return isPaused;
  },

  /** Export the current frame as a PNG blob */
  exportPNG() {
    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => {
        blob ? resolve(blob) : reject(new Error("Failed to export PNG"));
      }, "image/png");
    });
  },

  /** Download the current frame as a PNG file */
  downloadPNG(filename) {
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.download = filename || `ascii-clouds-${Date.now()}.png`;
      link.href = url;
      link.click();
      URL.revokeObjectURL(url);
    }, "image/png");
  },

  /** Get a snapshot of all current settings */
  getState() {
    return { ...state };
  },

  /** Randomize all settings */
  randomize() {
    const ranges = {
      cellSize: [4, 32],
      waveAmplitude: [0.05, 0.2],
      waveSpeed: [0.35, 1.0],
      noiseIntensity: [0, 0.1],
      vignetteIntensity: [0, 1],
      vignetteRadius: [0.05, 1],
      brightnessAdjust: [-0.25, 0.25],
      contrastAdjust: [0.5, 2],
      timeSpeed: [0.5, 1.5],
      hue: [0, 360],
      saturation: [0, 1],
      threshold1: [0, 0.5],
      threshold2: [0, 0.6],
      threshold3: [0.1, 0.7],
      threshold4: [0.2, 0.8],
      threshold5: [0.3, 1.0],
    };
    for (const [key, [min, max]] of Object.entries(ranges)) {
      state[key] = min + Math.random() * (max - min);
    }
    state.noiseSeed = Math.random().toString(36).substring(2, 8);
    if (isPaused) renderFrame();
  },
};
