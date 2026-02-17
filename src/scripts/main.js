import {
  vertexShaderSource,
  noiseFragmentSource,
  glyphFragmentSource,
  createProgram,
  hashSeed,
} from "./noise.js";

// WE doesn't support HSL colors in properties (unless you add three sliders),
// so convert from RGB (0-255 range) to HSL
function rgbToHsl(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const d = max - min;
  const l = (max + min) / 2;

  if (d === 0) return { h: 0, s: 0 };

  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h;
  if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
  else if (max === g) h = ((b - r) / d + 2) / 6;
  else h = ((r - g) / d + 4) / 6;

  return { h: h * 360, s };
}

function applyWEColor(colorString) {
  const parts = colorString.split(" ").map(Number);
  if (parts.length < 3 || parts.some(isNaN)) return;
  const { h, s } = rgbToHsl(parts[0], parts[1], parts[2]);
  params.hue = h;
  params.saturation = s;
}

// this is not at all the default parameters because the
// original site randomises them kinda,
// so i just took what i felt looked good for myself
const params = {
  cellSize: 10,
  waveAmplitude: 0.15,
  waveSpeed: 0.5,
  noiseIntensity: 0,
  vignetteIntensity: 0.5,
  vignetteRadius: 0.5,
  brightnessAdjust: 0,
  contrastAdjust: 1,
  timeSpeed: 1,
  hue: 180,
  saturation: 0.5,
  threshold1: 0.25,
  threshold2: 0.3,
  threshold3: 0.4,
  threshold4: 0.5,
  threshold5: 0.65,
  noiseSeed: Math.random().toString(36).substring(2, 8),
};

// webgl stuff
const canvas = document.getElementById("canvas");
const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true });
if (!gl) throw new Error("WebGL 2.0 not supported");

const noiseProgram = createProgram(gl, vertexShaderSource, noiseFragmentSource);
const glyphProgram = createProgram(gl, vertexShaderSource, glyphFragmentSource);

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

const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
const posLoc = gl.getAttribLocation(noiseProgram, "a_position");
gl.enableVertexAttribArray(posLoc);
gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

// framebuffer
let fbo = null;
let fboTexture = null;

function resizeFBO(w, h) {
  if (fbo) gl.deleteFramebuffer(fbo);
  if (fboTexture) gl.deleteTexture(fboTexture);

  fboTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, fboTexture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    w,
    h,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    null,
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(
    gl.FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    fboTexture,
    0,
  );
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function resize() {
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const rect = canvas.getBoundingClientRect();
  const w = Math.floor(rect.width * dpr);
  const h = Math.floor(rect.height * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    resizeFBO(w, h);
  }
}

resize();
window.addEventListener("resize", resize);

let elapsedTime = 0;
let lastTimestamp = 0;

function render() {
  resize();
  const w = canvas.width;
  const h = canvas.height;

  // noise
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.viewport(0, 0, w, h);
  gl.useProgram(noiseProgram);

  gl.uniform1f(noiseUniforms.time, elapsedTime);
  gl.uniform2f(noiseUniforms.resolution, w, h);
  gl.uniform1f(noiseUniforms.waveAmplitude, params.waveAmplitude);
  gl.uniform1f(noiseUniforms.waveSpeed, params.waveSpeed);
  gl.uniform1f(noiseUniforms.noiseIntensity, params.noiseIntensity);
  gl.uniform1f(noiseUniforms.vignetteIntensity, params.vignetteIntensity);
  gl.uniform1f(noiseUniforms.vignetteRadius, params.vignetteRadius);
  gl.uniform1f(noiseUniforms.brightnessAdjust, params.brightnessAdjust);
  gl.uniform1f(noiseUniforms.contrastAdjust, params.contrastAdjust);
  gl.uniform1f(noiseUniforms.noiseSeed, hashSeed(params.noiseSeed));

  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  // glyph
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, w, h);
  gl.useProgram(glyphProgram);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, fboTexture);
  gl.uniform1i(glyphUniforms.noiseTexture, 0);

  gl.uniform2f(glyphUniforms.resolution, w, h);
  gl.uniform1f(
    glyphUniforms.cellSize,
    params.cellSize * (window.devicePixelRatio || 1),
  );
  gl.uniform1f(glyphUniforms.hue, params.hue);
  gl.uniform1f(glyphUniforms.saturation, params.saturation);
  gl.uniform1f(glyphUniforms.threshold1, params.threshold1);
  gl.uniform1f(glyphUniforms.threshold2, params.threshold2);
  gl.uniform1f(glyphUniforms.threshold3, params.threshold3);
  gl.uniform1f(glyphUniforms.threshold4, params.threshold4);
  gl.uniform1f(glyphUniforms.threshold5, params.threshold5);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

// ---------- Animation loop ----------
let started = false;

function frame(timestamp) {
  if (!started) {
    lastTimestamp = timestamp;
    started = true;
  }
  const dt = (timestamp - lastTimestamp) / 1000;
  lastTimestamp = timestamp;
  elapsedTime += dt * params.timeSpeed;
  render();
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);

// config
const presetMap = ["default", "terminal", "retro", "cosmic", "fog"];

window.wallpaperPropertyListener = {
  applyUserProperties(properties) {
    // Preset must be applied first so individual sliders can override after
    if (properties.preset != null) {
      const name = presetMap[properties.preset.value];
      if (name) applyPreset(name);
    }

    if (properties.cellsize) params.cellSize = properties.cellsize.value;
    if (properties.waveamplitude)
      params.waveAmplitude = properties.waveamplitude.value;
    if (properties.wavespeed) params.waveSpeed = properties.wavespeed.value;
    if (properties.noiseintensity)
      params.noiseIntensity = properties.noiseintensity.value;
    if (properties.vignetteintensity)
      params.vignetteIntensity = properties.vignetteintensity.value;
    if (properties.vignetteradius)
      params.vignetteRadius = properties.vignetteradius.value;
    if (properties.brightnessadjust)
      params.brightnessAdjust = properties.brightnessadjust.value;
    if (properties.contrastadjust)
      params.contrastAdjust = properties.contrastadjust.value;
    if (properties.timespeed) params.timeSpeed = properties.timespeed.value;
    if (properties.glyphcolor) applyWEColor(properties.glyphcolor.value);
    if (properties.threshold1) params.threshold1 = properties.threshold1.value;
    if (properties.threshold2) params.threshold2 = properties.threshold2.value;
    if (properties.threshold3) params.threshold3 = properties.threshold3.value;
    if (properties.threshold4) params.threshold4 = properties.threshold4.value;
    if (properties.threshold5) params.threshold5 = properties.threshold5.value;
    if (properties.noiseseed) params.noiseSeed = properties.noiseseed.value;
  },
};

// ---------- Built-in presets ----------
const presets = {
  default: {
    cellSize: 10,
    waveAmplitude: 0.15,
    waveSpeed: 0.5,
    noiseIntensity: 0,
    vignetteIntensity: 0.5,
    vignetteRadius: 0.5,
    brightnessAdjust: 0,
    contrastAdjust: 1,
    timeSpeed: 1,
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
    threshold5: 1,
  },
  cosmic: {
    cellSize: 6,
    waveAmplitude: 0.5,
    waveSpeed: 0.5,
    noiseIntensity: 0.02,
    vignetteIntensity: 0.4,
    vignetteRadius: 0.8,
    brightnessAdjust: 0,
    contrastAdjust: 1,
    timeSpeed: 1,
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
};

function applyPreset(name) {
  const preset = presets[name];
  if (!preset) return;
  Object.assign(params, preset);
}
