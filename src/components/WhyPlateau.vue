<template>
  <section id="why" class="why-section">
    <div class="shell">
      <div class="why-heading">
        <div>
          <p class="kicker">Why continuation?</p>
          <h2>A distant target can be<br>invisible to the gradient.</h2>
        </div>
        <p>
          Move the rendered circle toward the reference. With a pixel-wise loss,
          two non-overlapping silhouettes produce the same error no matter how far
          apart they are. The objective is flat, so gradient descent has no direction
          to follow until the circles begin to overlap.
        </p>
      </div>

      <div class="demo-card">
        <div class="control-row">
          <div>
            <span class="control-label">Rendered circle center</span>
            <b>x = {{ centerX.toFixed(3) }}</b>
          </div>
          <input
            v-model.number="centerX"
            class="position-slider"
            type="range"
            min="0.10"
            max="0.90"
            step="0.001"
            aria-label="Rendered circle x position"
          >
          <button type="button" @click="reset">Reset</button>
        </div>

        <div class="render-grid">
          <figure>
            <figcaption><span>01</span> Current render</figcaption>
            <canvas ref="currentCanvas" width="480" height="250"></canvas>
            <p>The optimizable circle</p>
          </figure>
          <figure>
            <figcaption><span>02</span> Reference image</figcaption>
            <canvas ref="referenceCanvas" width="480" height="250"></canvas>
            <p>The fixed target on the right</p>
          </figure>
          <figure>
            <figcaption><span>03</span> Absolute error</figcaption>
            <canvas ref="errorCanvas" width="480" height="250"></canvas>
            <p>|render − reference|</p>
          </figure>
        </div>

        <div class="loss-panel">
          <div class="loss-copy">
            <p class="mini-kicker">Pixel-wise MSE landscape</p>
            <h3>Flat when there is no overlap.</h3>
            <p>
              The highlighted point tracks the current circle position. On the
              plateau, changing <em>x</em> does not change the number of erroneous
              pixels, so the derivative is approximately zero.
            </p>
            <div class="metrics">
              <div><span>MSE</span><b>{{ mse.toFixed(4) }}</b></div>
              <div><span>|∂L/∂x|</span><b>{{ gradient.toFixed(4) }}</b></div>
              <div><span>Region</span><b :class="{ active: !onPlateau }">{{ onPlateau ? 'Plateau' : 'Informative' }}</b></div>
            </div>
          </div>
          <div class="curve-wrap">
            <canvas ref="curveCanvas" width="900" height="300"></canvas>
            <span class="axis-label y-label">Loss</span>
            <span class="axis-label x-label">Circle center x</span>
            <span class="target-label init-pose-label">init x<sub>0</sub></span>
            <span class="target-label">target x*</span>
          </div>
        </div>
      </div>

      <div class="continuation-block">
      <div class="continuation-intro">
        <div>
          <p class="kicker">Why continuation helps</p>
          <h2>Move the basin,<br>then follow it.</h2>
        </div>
        <p>
          Sample the generated path as ten video frames and optimize each one in
          sequence. When a new frame arrives, the landscape jumps and the previous
          solution moves uphill; optimization then rolls it into the new minimum
          before advancing to the next frame.
        </p>
      </div>

      <div class="demo-card continuation-card">
        <div class="control-row">
          <div>
            <span class="control-label">Generated video sequence</span>
            <b>Frame {{ frameNumber }} / {{ frameCount }}</b>
          </div>
          <input
            v-model.number="continuationT"
            class="position-slider"
            type="range"
            min="0"
            max="1"
            step="0.001"
            aria-label="Continuation video time"
            @input="pauseAnimation"
          >
          <button type="button" @click="toggleAnimation">{{ isPlaying ? 'Pause' : continuationT >= 1 ? 'Replay' : 'Play video' }}</button>
        </div>

        <div class="continuation-grid">
          <figure>
            <figcaption><span>01</span> Generated video frame</figcaption>
            <canvas ref="videoCanvas" width="480" height="250"></canvas>
            <p>Discrete reference at x<sub>t</sub> = {{ intermediateX.toFixed(3) }}</p>
          </figure>
          <figure>
            <figcaption><span>02</span> Tracked solution</figcaption>
            <canvas ref="trackedCanvas" width="480" height="250"></canvas>
            <p>The optimized circle follows the moving minimum</p>
          </figure>
          <figure class="continuation-curve">
            <figcaption><span>03</span> Advected loss landscape</figcaption>
            <div class="curve-wrap">
              <canvas ref="continuationCurveCanvas" width="900" height="300"></canvas>
              <span class="axis-label y-label">Loss</span>
              <span class="axis-label x-label">Circle center x</span>
              <span class="target-label init-pose-label">init x<sub>0</sub></span>
              <span class="target-label">target x*</span>
              <span class="curve-legend"><i></i>Current frame <i></i>Direct final reference</span>
            </div>
            <p>The point rises after each jump, then rolls down the new basin</p>
          </figure>
        </div>
        <div class="frame-strip" aria-label="Ten generated video frames">
          <span
            v-for="(_, index) in frameCenters"
            :key="index"
            :class="{ complete: index < frameIndex, active: index === frameIndex }"
          >{{ String(index + 1).padStart(2, '0') }}</span>
        </div>
        <div class="continuation-progress" aria-hidden="true"><span :style="{ width: `${continuationT * 100}%` }"></span></div>
        <p class="continuation-status">
          <b>{{ continuationStatus }}</b>
          <span>local solve {{ Math.round(rollProgress * 100) }}% · x = {{ trackedX.toFixed(3) }} → x<sub>t</sub> = {{ intermediateX.toFixed(3) }}</span>
        </p>
      </div>

      <div class="takeaway">
        <span>Key observation</span>
        <p>
          The target image says what the solution should look like, but not how to
          reach it. Video numerical continuation supplies intermediate references
          that move the basin toward the optimizer.
        </p>
      </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const startX = 0.20
const centerX = ref(startX)
const targetX = 0.78
const radius = 0.13
const aspect = 480 / 250
const currentCanvas = ref<HTMLCanvasElement | null>(null)
const referenceCanvas = ref<HTMLCanvasElement | null>(null)
const errorCanvas = ref<HTMLCanvasElement | null>(null)
const curveCanvas = ref<HTMLCanvasElement | null>(null)
const videoCanvas = ref<HTMLCanvasElement | null>(null)
const trackedCanvas = ref<HTMLCanvasElement | null>(null)
const continuationCurveCanvas = ref<HTMLCanvasElement | null>(null)
const continuationT = ref(0)
const isPlaying = ref(false)

const frameCount = 10
const frameCenters = Array.from(
  { length: frameCount },
  (_, index) => startX + (targetX - startX) * index / (frameCount - 1)
)
const sequenceState = computed(() => {
  if (continuationT.value <= 0) return { index: 0, local: 1 }
  if (continuationT.value >= 1) return { index: frameCount - 1, local: 1 }
  const scaled = continuationT.value * (frameCount - 1)
  const lowerFrame = Math.floor(scaled)
  const fraction = scaled - lowerFrame
  if (fraction < 0.000001) return { index: lowerFrame, local: 1 }
  return { index: Math.min(frameCount - 1, lowerFrame + 1), local: fraction }
})
const frameIndex = computed(() => sequenceState.value.index)
const frameNumber = computed(() => frameIndex.value + 1)
const intermediateX = computed(() => frameCenters[frameIndex.value])
const rollProgress = computed(() => Math.min(1, sequenceState.value.local / 0.78))
const trackedX = computed(() => {
  if (frameIndex.value === 0) return startX
  const from = frameCenters[frameIndex.value - 1]
  const to = frameCenters[frameIndex.value]
  const easedRoll = 1 - Math.pow(1 - rollProgress.value, 3)
  return from + (to - from) * easedRoll
})
const continuationStatus = computed(() => {
  if (continuationT.value >= 1) return 'Final target reached'
  if (frameIndex.value === 0) return 'Initialized at the first video frame'
  return `Optimizing frame ${frameNumber.value} of ${frameCount}`
})
type Surface = {
  gl: WebGLRenderingContext
  program: WebGLProgram
  mode: number
}

const surfaces: Surface[] = []
const continuationSurfaces: Surface[] = []
let animationFrame: number | null = null

const normalizedLoss = (x: number): number => {
  const d = Math.abs(x - targetX) * aspect
  if (d >= 2 * radius) return 1
  const overlap = 2 * radius * radius * Math.acos(d / (2 * radius))
    - 0.5 * d * Math.sqrt(Math.max(0, 4 * radius * radius - d * d))
  return 1 - overlap / (Math.PI * radius * radius)
}

const plateauMse = 2 * Math.PI * radius * radius / aspect
const mse = computed(() => normalizedLoss(centerX.value) * plateauMse)
const gradient = computed(() => {
  const h = 0.0005
  return Math.abs((normalizedLoss(centerX.value + h) - normalizedLoss(centerX.value - h)) / (2 * h) * plateauMse)
})
const onPlateau = computed(() => Math.abs(centerX.value - targetX) * aspect >= 2 * radius)

const vertexSource = `
attribute vec2 a_position;
void main() { gl_Position = vec4(a_position, 0.0, 1.0); }
`

const fragmentSource = `
precision highp float;
uniform vec2 u_resolution;
uniform float u_x;
uniform float u_target;
uniform float u_final_target;
uniform float u_init;
uniform float u_radius;
uniform int u_mode;

float disk(vec2 uv, float cx) {
  float ar = u_resolution.x / u_resolution.y;
  vec2 p = vec2(uv.x * ar, uv.y);
  vec2 c = vec2(cx * ar, 0.5);
  float aa = 1.5 / u_resolution.y;
  return 1.0 - smoothstep(u_radius - aa, u_radius + aa, distance(p, c));
}

float lossAt(float cx, float target) {
  float ar = 1.92;
  float d = abs(cx - target) * ar;
  float r = u_radius;
  if (d >= 2.0 * r) return 1.0;
  float overlap = 2.0 * r * r * acos(d / (2.0 * r))
    - 0.5 * d * sqrt(max(0.0, 4.0 * r * r - d * d));
  return 1.0 - overlap / (3.14159265359 * r * r);
}

float continuationLoss(float cx) { return lossAt(cx, u_target); }

float lineMask(float value, float target, float width) {
  return 1.0 - smoothstep(width, width * 1.8, abs(value - target));
}

float segmentMask(vec2 p, vec2 a, vec2 b, float width) {
  vec2 pa = p - a;
  vec2 ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return 1.0 - smoothstep(width, width * 1.7, length(pa - ba * h));
}

void main() {
  vec2 uv = gl_FragCoord.xy / u_resolution;
  vec3 paper = vec3(0.965, 0.953, 0.925);
  vec3 ink = vec3(0.082, 0.137, 0.118);
  vec3 green = vec3(0.094, 0.310, 0.231);
  vec3 lime = vec3(0.804, 0.898, 0.420);
  vec3 orange = vec3(0.933, 0.490, 0.290);

  if (u_mode < 3) {
    float current = disk(uv, u_x);
    float target = disk(uv, u_target);
    float value = u_mode == 0 ? current : (u_mode == 1 ? target : abs(current - target));
    vec3 color = paper;
    if (u_mode == 2) {
      color = mix(vec3(0.075, 0.102, 0.090), orange, value);
    } else {
      color = mix(paper, u_mode == 0 ? green : ink, value);
    }
    gl_FragColor = vec4(color, 1.0);
    return;
  }

  vec3 color = paper;
  float left = 0.08;
  float right = 0.96;
  float bottom = 0.14;
  float top = 0.88;
  float axisX = lineMask(uv.x, left, 1.2 / u_resolution.x);
  float axisY = lineMask(uv.y, bottom, 1.2 / u_resolution.y);
  color = mix(color, ink, max(axisX * step(bottom, uv.y), axisY * step(left, uv.x)) * 0.65);

  for (int i = 1; i < 5; i++) {
    float gy = bottom + (top - bottom) * float(i) / 5.0;
    color = mix(color, ink, lineMask(uv.y, gy, 0.45 / u_resolution.y) * 0.10);
  }

  float dash = step(0.5, fract(uv.y * 28.0));
  float initGraphX = left + (u_init - 0.10) / 0.80 * (right - left);
  float finalGraphX = left + (u_final_target - 0.10) / 0.80 * (right - left);
  color = mix(color, ink, lineMask(uv.x, initGraphX, 1.2 / u_resolution.x) * dash * 0.22);
  color = mix(color, orange, lineMask(uv.x, finalGraphX, 1.2 / u_resolution.x) * dash * 0.34);

  float graphX = clamp((uv.x - left) / (right - left), 0.0, 1.0);
  float cx = mix(0.10, 0.90, graphX);
  float loss = continuationLoss(cx);
  float curveY = mix(bottom, top, loss);
  if (u_mode == 4) {
    float directY = mix(bottom, top, lossAt(cx, u_final_target));
    color = mix(color, orange, lineMask(uv.y, directY, 2.5 / u_resolution.y) * 0.24);
  }
  color = mix(color, green, lineMask(uv.y, curveY, 2.2 / u_resolution.y));

  vec2 point = vec2(
    left + (u_x - 0.10) / 0.80 * (right - left),
    mix(bottom, top, continuationLoss(u_x))
  );
  vec2 delta = (uv - point) * vec2(u_resolution.x / u_resolution.y, 1.0);
  float outerRadius = u_mode == 4 ? 0.045 : 0.020;
  float innerRadius = u_mode == 4 ? 0.032 : 0.010;
  float dotOuter = 1.0 - smoothstep(outerRadius, outerRadius + 0.004, length(delta));
  float dotInner = 1.0 - smoothstep(innerRadius, innerRadius + 0.003, length(delta));
  if (u_mode == 4) {
    vec2 scale = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 tip = point + vec2(0.0, 0.050);
    vec2 top = point + vec2(0.0, 0.245);
    vec2 leftHead = point + vec2(-0.018, 0.078);
    vec2 rightHead = point + vec2(0.018, 0.078);
    float arrow = segmentMask(uv * scale, top * scale, tip * scale, 1.8 / u_resolution.y);
    arrow = max(arrow, segmentMask(uv * scale, leftHead * scale, tip * scale, 1.8 / u_resolution.y));
    arrow = max(arrow, segmentMask(uv * scale, rightHead * scale, tip * scale, 1.8 / u_resolution.y));
    color = mix(color, orange, arrow);
  }
  color = mix(color, paper, dotOuter);
  color = mix(color, orange, dotInner);
  gl_FragColor = vec4(color, 1.0);
}
`

function compileShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)
  if (!shader) throw new Error('Unable to create WebGL shader')
  gl.shaderSource(shader, source)
  gl.compileShader(shader)
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || 'WebGL shader compilation failed')
  }
  return shader
}

function createSurface(canvas: HTMLCanvasElement, mode: number): Surface {
  const gl = canvas.getContext('webgl', { antialias: true })
  if (!gl) throw new Error('WebGL is unavailable in this browser')
  const program = gl.createProgram()
  if (!program) throw new Error('Unable to create WebGL program')
  gl.attachShader(program, compileShader(gl, gl.VERTEX_SHADER, vertexSource))
  gl.attachShader(program, compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource))
  gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) || 'WebGL program linking failed')
  }
  const buffer = gl.createBuffer()
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW)
  const position = gl.getAttribLocation(program, 'a_position')
  gl.enableVertexAttribArray(position)
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0)
  return { gl, program, mode }
}

function draw(): void {
  surfaces.forEach(({ gl, program, mode }) => {
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height)
    gl.useProgram(program)
    gl.uniform2f(gl.getUniformLocation(program, 'u_resolution'), gl.canvas.width, gl.canvas.height)
    gl.uniform1f(gl.getUniformLocation(program, 'u_x'), centerX.value)
    gl.uniform1f(gl.getUniformLocation(program, 'u_target'), targetX)
    gl.uniform1f(gl.getUniformLocation(program, 'u_final_target'), targetX)
    gl.uniform1f(gl.getUniformLocation(program, 'u_init'), startX)
    gl.uniform1f(gl.getUniformLocation(program, 'u_radius'), radius)
    gl.uniform1i(gl.getUniformLocation(program, 'u_mode'), mode)
    gl.drawArrays(gl.TRIANGLES, 0, 6)
  })
}

function drawContinuation(): void {
  continuationSurfaces.forEach(({ gl, program, mode }) => {
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height)
    gl.useProgram(program)
    gl.uniform2f(gl.getUniformLocation(program, 'u_resolution'), gl.canvas.width, gl.canvas.height)
    gl.uniform1f(gl.getUniformLocation(program, 'u_x'), trackedX.value)
    gl.uniform1f(gl.getUniformLocation(program, 'u_target'), intermediateX.value)
    gl.uniform1f(gl.getUniformLocation(program, 'u_final_target'), targetX)
    gl.uniform1f(gl.getUniformLocation(program, 'u_init'), startX)
    gl.uniform1f(gl.getUniformLocation(program, 'u_radius'), radius)
    gl.uniform1i(gl.getUniformLocation(program, 'u_mode'), mode)
    gl.drawArrays(gl.TRIANGLES, 0, 6)
  })
}

function reset(): void { centerX.value = startX }

function pauseAnimation(): void {
  isPlaying.value = false
  if (animationFrame !== null) cancelAnimationFrame(animationFrame)
  animationFrame = null
}

function toggleAnimation(): void {
  if (isPlaying.value) {
    pauseAnimation()
    return
  }
  if (continuationT.value >= 1) continuationT.value = 0
  isPlaying.value = true
  const duration = 12000
  const startedAt = performance.now() - continuationT.value * duration
  const animate = (now: number): void => {
    continuationT.value = Math.min(1, (now - startedAt) / duration)
    if (continuationT.value < 1 && isPlaying.value) {
      animationFrame = requestAnimationFrame(animate)
    } else {
      isPlaying.value = false
      animationFrame = null
    }
  }
  animationFrame = requestAnimationFrame(animate)
}

onMounted(() => {
  const canvases = [currentCanvas.value, referenceCanvas.value, errorCanvas.value, curveCanvas.value]
  canvases.forEach((canvas, mode) => { if (canvas) surfaces.push(createSurface(canvas, mode)) })
  const continuationCanvases = [trackedCanvas.value, videoCanvas.value, continuationCurveCanvas.value]
  const continuationModes = [0, 1, 4]
  continuationCanvases.forEach((canvas, index) => {
    if (canvas) continuationSurfaces.push(createSurface(canvas, continuationModes[index]))
  })
  draw()
  drawContinuation()
})

watch(centerX, draw)
watch(continuationT, drawContinuation)
onBeforeUnmount(pauseAnimation)
</script>

<style scoped>
.why-section { padding: 68px 0; background: var(--green); color: white; }
.shell { width: min(1100px, calc(100% - 48px)); margin: 0 auto; }
.why-heading { display: grid; grid-template-columns: 1.1fr .9fr; align-items: end; gap: 54px; margin-bottom: 32px; }
.kicker, .mini-kicker { color: var(--lime); font-family: 'AliBold'; font-size: 12px; letter-spacing: .16em; text-transform: uppercase; }
h2 { margin: 0; font-family: 'AliHeavy'; font-size: clamp(36px, 4.4vw, 54px); line-height: 1.01; letter-spacing: -.04em; }
.why-heading > p { margin: 0 0 2px; color: rgba(255,255,255,.68); font-size: 15px; line-height: 1.55; }
.demo-card { padding: 20px 24px; background: #f7f4ed; color: var(--ink); }
.control-row { padding-bottom: 15px; display: grid; grid-template-columns: 195px 1fr auto; align-items: center; gap: 24px; border-bottom: 1px solid var(--line); }
.control-row > div { display: flex; flex-direction: column; gap: 4px; }.control-label { color: var(--muted); font-size: 12px; }.control-row b { color: var(--green); font-family: 'AliBold'; font-size: 20px; }
.position-slider { width: 100%; height: 4px; appearance: none; background: linear-gradient(90deg, var(--orange), var(--lime)); border-radius: 0; cursor: ew-resize; }
.position-slider::-webkit-slider-thumb { width: 22px; height: 22px; appearance: none; background: var(--paper); border: 6px solid var(--green); border-radius: 50%; }.position-slider::-moz-range-thumb { width: 12px; height: 12px; background: var(--paper); border: 6px solid var(--green); border-radius: 50%; }
button { padding: 10px 16px; background: transparent; color: var(--green); border: 1px solid var(--green); font-family: 'AliBold'; cursor: pointer; }
.render-grid { padding: 17px 0 19px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
figure { margin: 0; }.render-grid figcaption { margin-bottom: 10px; display: flex; align-items: center; gap: 8px; font-family: 'AliBold'; font-size: 13px; }.render-grid figcaption span { color: var(--orange); font-size: 10px; }.render-grid canvas { width: 100%; display: block; border: 1px solid var(--line); }.render-grid figure > p { margin: 9px 0 0; color: var(--muted); font-size: 12px; }
.loss-panel { padding-top: 18px; display: grid; grid-template-columns: .62fr 1.38fr; align-items: center; gap: 34px; border-top: 1px solid var(--line); }
.loss-copy h3 { margin: 5px 0 8px; font-family: 'AliHeavy'; font-size: 25px; letter-spacing: -.025em; }.loss-copy > p:not(.mini-kicker) { margin-bottom: 0; color: var(--muted); font-size: 13px; line-height: 1.48; }.mini-kicker { margin-bottom: 0; color: var(--orange); font-size: 10px; }
.metrics { margin-top: 13px; display: grid; grid-template-columns: repeat(3, 1fr); border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }.metrics div { padding: 8px 8px 8px 0; display: flex; flex-direction: column; gap: 3px; }.metrics span { color: var(--muted); font-size: 9px; letter-spacing: .1em; text-transform: uppercase; }.metrics b { font-family: 'AliBold'; font-size: 13px; }.metrics b.active { color: var(--orange); }
.curve-wrap { position: relative; padding: 2px 8px 18px 24px; }.curve-wrap canvas { width: 100%; display: block; }.axis-label, .target-label { position: absolute; color: var(--muted); font-size: 9px; letter-spacing: .08em; text-transform: uppercase; }.y-label { top: 50%; left: 0; transform: rotate(-90deg) translateX(-50%); transform-origin: left top; }.x-label { right: 8px; bottom: 0; }.target-label { top: 2px; left: 82.8%; color: var(--orange); transform: translateX(-50%); white-space: nowrap; }.init-pose-label { left: 19%; color: var(--muted); }.target-label sub { font-size: .72em; }
.continuation-block { margin-top: 68px; padding-top: 62px; border-top: 1px solid rgba(255,255,255,.24); }
.continuation-intro { margin-bottom: 28px; display: grid; grid-template-columns: 1.1fr .9fr; align-items: end; gap: 54px; }
.continuation-intro > p { margin: 0 0 2px; color: rgba(255,255,255,.68); font-size: 15px; line-height: 1.55; }
.continuation-grid { padding: 17px 0 13px; display: grid; grid-template-columns: .8fr .8fr 1.4fr; align-items: start; gap: 16px; }
.continuation-grid figcaption { margin-bottom: 10px; display: flex; align-items: center; gap: 8px; font-family: 'AliBold'; font-size: 13px; }
.continuation-grid figcaption span { color: var(--orange); font-size: 10px; }
.continuation-grid canvas { width: 100%; display: block; border: 1px solid var(--line); }
.continuation-grid figure > p { margin: 9px 0 0; color: var(--muted); font-size: 12px; }
.continuation-grid sub { font-size: .72em; }
.continuation-curve .curve-wrap { padding-top: 0; }
.continuation-curve canvas { border: 0; }
.continuation-curve .target-label { top: 18px; }
.curve-legend { position: absolute; top: 0; right: 10px; display: flex; align-items: center; gap: 5px; color: var(--muted); font-size: 8px; letter-spacing: .06em; text-transform: uppercase; }
.curve-legend i { width: 13px; height: 2px; margin-left: 7px; background: var(--green); }
.curve-legend i:first-child { margin-left: 0; }
.curve-legend i:nth-of-type(2) { background: var(--orange); opacity: .42; }
.frame-strip { margin-top: 1px; display: grid; grid-template-columns: repeat(10, 1fr); border-top: 1px solid var(--line); border-left: 1px solid var(--line); }
.frame-strip span { padding: 7px 0; color: var(--muted); border-right: 1px solid var(--line); font-family: 'AliBold'; font-size: 9px; text-align: center; transition: color .12s, background .12s; }
.frame-strip span.complete { background: rgba(202,229,107,.22); color: var(--green); }
.frame-strip span.active { background: var(--green); color: white; }
.continuation-progress { height: 4px; overflow: hidden; background: var(--line); }
.continuation-progress span { height: 100%; display: block; background: linear-gradient(90deg, var(--orange), var(--lime)); transition: width .04s linear; }
.continuation-status { margin: 10px 0 0; display: flex; justify-content: space-between; gap: 20px; color: var(--muted); font-size: 11px; letter-spacing: .04em; }
.continuation-status b { color: var(--green); font-family: 'AliBold'; }
.takeaway { padding: 17px 0 0; display: grid; grid-template-columns: 165px 1fr; gap: 28px; }.takeaway span { color: var(--lime); font-family: 'AliBold'; font-size: 10px; letter-spacing: .14em; text-transform: uppercase; }.takeaway p { max-width: 900px; margin: 0; color: rgba(255,255,255,.76); font-size: 15px; line-height: 1.45; }
@media (max-width: 900px) { .why-heading, .loss-panel, .continuation-intro { grid-template-columns: 1fr; gap: 28px; }.control-row { grid-template-columns: 1fr auto; }.position-slider { grid-column: 1/-1; grid-row: 2; }.render-grid, .continuation-grid { grid-template-columns: 1fr; }.render-grid figure { display: grid; grid-template-columns: 1fr 1fr; align-items: center; gap: 12px; }.render-grid figcaption, .render-grid figure > p { grid-column: 1/-1; }.continuation-grid canvas { max-height: 280px; object-fit: contain; }.takeaway { grid-template-columns: 1fr; gap: 10px; } }
@media (max-width: 640px) { .why-section { padding: 76px 0; }.shell { width: calc(100% - 28px); }.demo-card { padding: 18px; }.control-row { grid-template-columns: 1fr auto; gap: 16px; }.render-grid figure { display: block; }.render-grid canvas { margin-top: 10px; }.loss-panel { gap: 26px; }.metrics { grid-template-columns: 1fr; }.metrics div { border-bottom: 1px solid var(--line); }.metrics div:last-child { border: 0; }.curve-wrap { padding-left: 20px; }.target-label { display: none; }.continuation-block { margin-top: 52px; padding-top: 50px; }.continuation-status { align-items: flex-start; flex-direction: column; gap: 4px; }.curve-legend { display: none; } }
</style>
