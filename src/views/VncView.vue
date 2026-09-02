<template>
  <main>
    <nav class="nav shell">
      <a class="brand" href="#top"><span class="brand-mark"></span>VNC</a>
      <div class="nav-links">
        <a href="#why">Why</a>
        <a href="#method">Method</a>
        <a href="#results">Results</a>
        <a class="nav-resource" href="https://drive.google.com/file/d/1-f5QKNjVe0_oHZNXZmii1a9vJLKIE5wH/view?usp=sharing" target="_blank" rel="noopener">Paper ↗</a>
        <a class="nav-resource" href="https://github.com/SuikaSibyl/vnc" target="_blank" rel="noopener">Code ↗</a>
      </div>
    </nav>

    <header id="top" class="hero shell">
      <div class="hero-copy">
        <p class="eyebrow"><span></span>SIGGRAPH Asia 2026</p>
        <h1>Video Numerical Continuation <em>for Differentiable Inverse Rendering</em></h1>
        <div class="author">
          <div class="author-list">
            <a href="https://suikasibyl.github.io/" target="_blank" rel="noopener">Haolin Lu</a>
            <a href="https://ishit.github.io/" target="_blank" rel="noopener">Ishit Mehta</a>
            <a href="https://haodong2000.github.io/" target="_blank" rel="noopener">Haodong Li</a>
            <span>Yu Fu</span>
            <a href="https://cseweb.ucsd.edu/~tzli/" target="_blank" rel="noopener">Tzu-Mao Li</a>
            <a href="https://cseweb.ucsd.edu/~ravir/" target="_blank" rel="noopener">Ravi Ramamoorthi</a>
          </div>
          <p class="author-affiliation">University of California San Diego</p>
        </div>
        <p class="dek">Use a generated video as a continuous path through a difficult inverse-rendering objective.</p>
        <div class="actions">
          <a class="button primary" href="https://drive.google.com/file/d/1-f5QKNjVe0_oHZNXZmii1a9vJLKIE5wH/view?usp=sharing" target="_blank" rel="noopener">Paper <b>↗</b></a>
          <a class="button secondary" href="https://github.com/SuikaSibyl/vnc" target="_blank" rel="noopener">Code <b>↗</b></a>
          <a class="button secondary" href="#results">Results <b>↓</b></a>
        </div>
        <div class="scope-line"><span>Geometry</span><i></i><span>Pose</span><i></i><span>Motion</span><i></i><span>Light</span><i></i><span>Material</span></div>
      </div>
      <div class="hero-media">
        <div class="media-label"><b>01</b><span>Generated continuation path</span></div>
        <video autoplay muted loop playsinline preload="metadata" :poster="media('house-target.png')">
          <source :src="media('house.mp4')" type="video/mp4">
        </video>
        <div class="corner corner-a"></div><div class="corner corner-b"></div>
      </div>
    </header>

    <section id="abstract" class="abstract-section">
      <div class="shell split-heading">
        <div><p class="kicker">Abstract</p><h2>Turn one hard objective into a path of nearby ones.</h2></div>
        <p>
          Differentiable rendering enables gradient-based scene parameter recovery, but its objectives are highly non-convex and prone to poor local minima. Given initial and target images, we synthesize a smooth video trajectory and optimize scene parameters sequentially across its frames. This numerical continuation guides parameters along a tractable path and enables convergence from challenging initializations where standard methods stall.
        </p>
      </div>
    </section>

    <WhyPlateau />

    <section id="method" class="method shell">
      <div class="section-heading"><p class="kicker">Method</p><h2>Optimize through the frames between.</h2></div>
      <div class="pipeline">
        <article>
          <span class="step">01</span>
          <div class="frame"><img :src="media('house-init.png')" alt="Initial scene configuration"></div>
          <h3>Render the initialization</h3><p>Start with parameters that may be far from the desired solution.</p>
        </article>
        <div class="arrow">→</div>
        <article>
          <span class="step">02</span>
          <div class="frame"><video ref="generatedVideo" autoplay muted loop playsinline preload="metadata"><source :src="media('house.mp4')" type="video/mp4"></video></div>
          <h3>Generate a trajectory</h3><p>Condition a video model on the first and last frames to infer intermediate states.</p>
        </article>
        <div class="arrow">→</div>
        <article>
          <span class="step">03</span>
          <div class="frame"><video ref="optimizationVideo" autoplay muted loop playsinline preload="metadata"><source :src="resultMedia('resources/pose/house/vnc.mp4')" type="video/mp4"></video></div>
          <h3>Follow and converge</h3><p>Optimize through the scheduled references and watch the rendered pose converge.</p>
        </article>
      </div>
      <div class="method-note">
        <span>λ = 0</span><div ref="methodProgress" class="progress"><i></i></div><span>λ = 1</span>
        <p>The animated schedule advances through the generated references while the renderer and image loss remain unchanged.</p>
      </div>
    </section>

    <ResultsExplorer />

    <footer class="footer shell">
      <div><span class="brand-mark"></span><b>VNC</b></div>
      <p>Video Numerical Continuation for Differentiable Inverse Rendering.</p>
      <a href="https://github.com/SuikaSibyl/vnc" target="_blank" rel="noopener">GitHub ↗</a>
    </footer>
  </main>
</template>

<script lang="ts">
import { defineComponent } from 'vue'
import WhyPlateau from '../components/WhyPlateau.vue'
import ResultsExplorer from '../components/ResultsExplorer.vue'

export default defineComponent({
  name: 'VncView',
  components: { WhyPlateau, ResultsExplorer },
  data() {
    return { methodSyncFrame: 0 }
  },
  methods: {
    media(filename: string): string { return `${process.env.BASE_URL}media/${filename}` },
    resultMedia(path: string): string { return `${process.env.BASE_URL}results/${path}` },
    startMethodSync(): void {
      const generated = this.$refs.generatedVideo as HTMLVideoElement
      const optimization = this.$refs.optimizationVideo as HTMLVideoElement
      const progress = this.$refs.methodProgress as HTMLDivElement
      const sync = (): void => {
        if (generated.duration > 0 && optimization.duration > 0) {
          const phase = generated.currentTime / generated.duration
          const targetTime = phase * optimization.duration
          const synchronizedRate = optimization.duration / generated.duration
          if (Math.abs(optimization.playbackRate - synchronizedRate) > 0.001) optimization.playbackRate = synchronizedRate
          if (!optimization.seeking && Math.abs(optimization.currentTime - targetTime) > 0.12) optimization.currentTime = targetTime
          if (generated.paused && !optimization.paused) optimization.pause()
          if (!generated.paused && optimization.paused) optimization.play().catch(() => undefined)
          progress.style.setProperty('--method-progress', `${phase * 100}%`)
        }
        this.methodSyncFrame = requestAnimationFrame(sync)
      }
      this.methodSyncFrame = requestAnimationFrame(sync)
    },
  },
  mounted() { this.startMethodSync() },
  beforeUnmount() { if (this.methodSyncFrame) cancelAnimationFrame(this.methodSyncFrame) },
})
</script>

<style scoped>
.shell { width: min(1100px, calc(100% - 48px)); margin: 0 auto; }
.nav { height: 86px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--line); }
.brand, .footer b { display: flex; align-items: center; gap: 10px; font-family: 'AliHeavy'; font-size: 20px; letter-spacing: .12em; text-decoration: none; }
.brand-mark { width: 17px; height: 17px; display: inline-block; background: var(--lime); border: 5px solid var(--green); border-radius: 50%; }
.nav-links { display: flex; gap: 32px; }
.nav-links a { color: var(--muted); font-family: 'AliMedium'; font-size: 14px; text-decoration: none; }.nav-links .nav-resource { color: var(--green); font-family: 'AliBold'; }
.hero { min-width: 0; min-height: 680px; padding: 48px 0 64px; display: grid; grid-template-columns: 1.05fr .95fr; align-items: center; gap: 54px; }
.hero-copy, .hero-media { min-width: 0; }
.eyebrow, .kicker { color: var(--green); font-family: 'AliBold'; font-size: 12px; letter-spacing: .16em; text-transform: uppercase; }
.eyebrow { display: flex; align-items: center; gap: 10px; }
.eyebrow span { width: 34px; height: 2px; background: var(--orange); }
h1, h2, h3, p { margin-top: 0; }
h1 { margin: 20px 0 20px; font-family: 'AliHeavy'; font-size: clamp(48px, 4.8vw, 70px); line-height: .98; letter-spacing: -.052em; }
h1 em { display: block; margin-top: 8px; color: var(--green); font-family: 'AliLight'; font-size: .66em; font-style: normal; font-weight: 400; line-height: 1.05; }
.author { margin-bottom: 20px; }
.author-list { display: flex; flex-wrap: wrap; gap: 5px 15px; }
.author-list a, .author-list span { color: var(--green); font-family: 'AliBold'; font-size: 16px; }
.author-list a { text-decoration-color: transparent; text-underline-offset: 3px; transition: text-decoration-color .18s; }
.author-list a:hover { text-decoration-color: var(--orange); }
.author-affiliation { margin: 8px 0 0; color: var(--muted); font-size: 14px; }
.dek { max-width: 600px; color: var(--muted); font-size: 17px; line-height: 1.5; }
.actions { margin: 24px 0 28px; display: flex; flex-wrap: wrap; gap: 12px; }
.button { min-width: 150px; padding: 15px 20px; display: inline-flex; justify-content: space-between; gap: 28px; border: 1px solid var(--green); font-family: 'AliMedium'; font-size: 14px; text-decoration: none; transition: transform .18s; }
.button:hover { transform: translateY(-2px); }
.button.primary { background: var(--green); color: white; }.button.secondary { color: var(--green); }
.scope-line { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; color: var(--muted); font-size: 10px; letter-spacing: .1em; text-transform: uppercase; }
.scope-line i { width: 3px; height: 3px; background: var(--orange); border-radius: 50%; }
.hero-media { position: relative; padding: 21px; background: var(--green); box-shadow: 16px 16px 0 var(--lime); }
.hero-media video { width: 100%; aspect-ratio: 1/1; display: block; background: #d8ddd5; object-fit: cover; }
.media-label { margin-bottom: 17px; display: flex; justify-content: space-between; color: rgba(255,255,255,.68); font-size: 11px; letter-spacing: .12em; text-transform: uppercase; }.media-label b { color: var(--lime); }
.corner { position: absolute; width: 30px; height: 30px; }.corner-a { top: -12px; left: -12px; border-top: 2px solid var(--orange); border-left: 2px solid var(--orange); }.corner-b { right: -12px; bottom: -12px; border-right: 2px solid var(--orange); border-bottom: 2px solid var(--orange); }
.abstract-section { padding: 110px 0; background: #e9e5dc; }
.split-heading { display: grid; grid-template-columns: 1.15fr .85fr; align-items: end; gap: 90px; }
.section-heading { margin-bottom: 52px; }
.section-heading h2, .split-heading h2 { margin-bottom: 0; font-family: 'AliHeavy'; font-size: clamp(36px, 4.4vw, 54px); line-height: 1.04; letter-spacing: -.04em; }
.split-heading > p { margin-bottom: 4px; color: var(--muted); font-size: 17px; line-height: 1.72; }
.method { padding: 118px 0; }
.pipeline { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; align-items: start; gap: 18px; }
.pipeline article { position: relative; }.step { position: absolute; z-index: 2; top: 12px; left: 12px; padding: 5px 8px; background: var(--lime); color: var(--green); font-family: 'AliBold'; font-size: 11px; }
.frame { overflow: hidden; aspect-ratio: 4/3; background: #ddd; border: 1px solid var(--line); }.frame img, .frame video { width: 100%; height: 100%; display: block; object-fit: cover; }
.pipeline h3 { margin: 21px 0 8px; font-family: 'AliBold'; font-size: 21px; }.pipeline p { color: var(--muted); line-height: 1.55; }.arrow { margin-top: 112px; color: var(--orange); font-size: 28px; }
.method-note { margin-top: 54px; padding: 22px 0; display: grid; grid-template-columns: auto minmax(120px, 1fr) auto 1.8fr; align-items: center; gap: 16px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); color: var(--green); font-family: 'AliBold'; font-size: 12px; }.method-note p { margin: 0 0 0 26px; color: var(--muted); font-family: 'AliRegular'; font-size: 14px; }.progress { position: relative; height: 4px; background: rgba(24,79,59,.16); }.progress i { width: var(--method-progress, 0%); height: 100%; display: block; background: linear-gradient(90deg, var(--orange), var(--lime)); }.progress::after { content: ''; position: absolute; top: 50%; left: var(--method-progress, 0%); width: 12px; height: 12px; background: var(--paper); border: 4px solid var(--green); border-radius: 50%; transform: translate(-50%, -50%); }
.footer { min-height: 160px; display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 30px; }.footer > div { display: flex; align-items: center; gap: 10px; }.footer p { margin: 0; color: var(--muted); white-space: nowrap; }.footer > a { color: var(--green); font-family: 'AliBold'; text-decoration: none; }
@media (max-width: 900px) { .hero { grid-template-columns: 1fr; gap: 64px; }.hero-media { width: min(620px, calc(100% - 20px)); }.split-heading { grid-template-columns: 1fr; gap: 26px; }.pipeline { grid-template-columns: 1fr; gap: 28px; }.arrow { display: none; }.method-note { grid-template-columns: auto 1fr auto; }.method-note p { grid-column: 1/-1; margin-left: 0; } }
@media (max-width: 640px) { .shell { width: calc(100% - 28px); }.nav-links { gap: 14px; }.nav-links a:not(.nav-resource) { display: none; }.hero { min-height: auto; padding: 44px 0 64px; }h1 { max-width: 100%; font-size: 46px; overflow-wrap: anywhere; }h1 em { font-size: .56em; line-height: 1.15; white-space: normal; }.hero-media { width: 100%; max-width: calc(100vw - 40px); padding: 16px; box-shadow: 12px 12px 0 var(--lime); }.media-label span { display: none; }.abstract-section, .method { padding: 76px 0; }.footer { padding: 42px 0; grid-template-columns: 1fr; gap: 14px; }.footer p { white-space: normal; } }
</style>
