<template>
  <section id="results" class="results-section">
    <div class="shell">
      <div class="results-heading">
        <div>
          <p class="kicker">Selected results</p>
          <h2>One method,<br>many applications.</h2>
        </div>
        <p>
          Browse representative inverse-rendering tasks across pose, geometry,
          articulation, lighting, and material. Videos within each case are
          synchronized so convergence behavior can be compared frame by frame.
        </p>
      </div>

      <div class="application-selector">
        <span>Choose application</span>
        <div>
          <button
            v-for="application in applications"
            :key="application.name"
            type="button"
            :class="{ active: activeCase.category === application.name }"
            @click="openCasePicker(application.name)"
          >
            {{ application.name }} <small>{{ application.count }}</small>
          </button>
        </div>
        <p><b>{{ activeCase.name }}</b><small>Current case</small></p>
      </div>

      <div class="comparison-stage">
        <header class="case-header">
          <div>
            <p>{{ activeCase.category }} · {{ activeCase.parameter }}</p>
            <h3>{{ activeCase.name }}</h3>
          </div>
          <p>{{ activeCase.description }}</p>
          <span class="sync-badge"><i></i>Synchronized playback</span>
        </header>

        <div class="matrix-scroll">
          <div class="matrix-grid" :style="{ '--column-count': activeCase.columns.length }">
            <section v-for="(column, columnIndex) in activeCase.columns" :key="`${activeCase.id}-${column.title}-${columnIndex}`" class="matrix-column">
              <p class="column-title"><span>0{{ columnIndex + 1 }}</span>{{ column.title }}</p>
              <article
                v-for="cell in column.cells"
                :key="`${activeCase.id}-${cell.id}`"
                class="matrix-cell"
                :class="{ featured: cell.featured }"
              >
                <div class="cell-title">
                  <b>{{ cell.label }}</b>
                  <span>{{ cell.kind === 'image' ? 'Still' : cell.kind === 'video' ? 'Video' : 'N/A' }}</span>
                </div>
                <div class="matrix-frame" :class="{ contain: activeCase.contain }">
                  <img
                    v-if="cell.kind === 'image'"
                    :src="resultMedia(cell.src)"
                    :alt="`${activeCase.name} ${cell.label}`"
                    loading="lazy"
                  >
                  <div v-else-if="cell.kind === 'na'" class="not-available">Not available</div>
                  <video
                    v-else
                    :ref="setVideoRef"
                    autoplay
                    muted
                    loop
                    playsinline
                    preload="metadata"
                    controls
                  >
                    <source :src="resultMedia(cell.src)" type="video/mp4">
                  </video>
                </div>
              </article>
            </section>
          </div>
        </div>

        <footer class="case-footer">
          <p><span>Reading the matrix</span> Initialization and target appear first; baselines and our generated-reference optimization follow.</p>
          <b>{{ activeCase.videoCount }} synchronized videos</b>
        </footer>
      </div>
    </div>

    <Teleport to="body">
      <div v-if="casePickerOpen" class="case-picker-backdrop" @click.self="closeCasePicker">
        <section class="case-picker" role="dialog" aria-modal="true" :aria-label="`${pickerCategory} result cases`">
          <header>
            <div>
              <p>Application gallery</p>
              <h3>{{ pickerCategory }} cases</h3>
            </div>
            <span>{{ visibleCases.length }} cases</span>
            <button type="button" aria-label="Close case picker" @click="closeCasePicker">×</button>
          </header>
          <div class="case-gallery">
            <button
              v-for="(item, index) in visibleCases"
              :key="item.id"
              type="button"
              :class="{ selected: item.id === activeCase.id }"
              @click="openCase(item.id)"
            >
              <div class="case-thumb">
                <img :src="resultMedia(item.thumb)" :alt="`${item.name} preview`" loading="lazy">
                <span>0{{ index + 1 }}</span>
              </div>
              <p><small>{{ item.parameter }}</small><b>{{ item.name }}</b></p>
              <i>Open case →</i>
            </button>
          </div>
        </section>
      </div>
    </Teleport>
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import rawCases from '../data/resultsCases.json'

type ResultCell = {
  id: string
  label: string
  kind: 'image' | 'video' | 'na'
  src: string
  featured?: boolean
}

type ResultCase = {
  id: string
  name: string
  category: string
  parameter: string
  description: string
  thumb: string
  contain?: boolean
  columns: Array<{ title: string; cells: ResultCell[] }>
  videoCount: number
}

type RawCell = { id: string; label: string; kind: 'image' | 'video' | 'na'; src?: string; video?: string }
type RawCase = {
  id: string
  name: string
  category: string
  description?: string
  thumb: string
  freeAspect?: boolean
  matrixColumns: Array<{ title: string; cells: RawCell[] }>
}

const parameterByCategory: Record<string, string> = {
  Pose: 'Rigid transform',
  Geometry: 'Mesh vertices',
  Skeleton: 'Joint rotations',
  Lighting: 'Light position',
  Material: 'Material parameters',
}
const descriptionByCategory: Record<string, string> = {
  Pose: 'Rigid pose recovery from a challenging distant initialization.',
  Geometry: 'Geometry reconstruction with image-space differentiable rendering.',
  Skeleton: 'Articulated pose recovery across many coupled joint parameters.',
  Lighting: 'Inverse lighting from shading and shadow observations.',
  Material: 'Differentiable recovery of spatially varying material parameters.',
}

const cases: ResultCase[] = (rawCases as RawCase[]).map(item => {
  const category = item.category === 'Lights' ? 'Lighting' : item.category
  const columns = item.matrixColumns.map(column => ({
    title: column.title,
    cells: column.cells.map(cell => ({
      id: cell.id,
      label: cell.label,
      kind: cell.kind,
      src: cell.src || cell.video || '',
      featured: /ours/i.test(cell.label),
    })),
  }))
  return {
    id: item.id,
    name: item.id === 'house' ? 'Fantasy house' : item.name,
    category,
    parameter: parameterByCategory[category],
    description: item.description || descriptionByCategory[category],
    thumb: item.thumb,
    contain: Boolean(item.freeAspect),
    columns,
    videoCount: columns.flatMap(column => column.cells).filter(cell => cell.kind === 'video').length,
  }
})

const applicationOrder = ['Pose', 'Geometry', 'Skeleton', 'Lighting', 'Material']
const applications = applicationOrder.map(name => ({ name, count: cases.filter(item => item.category === name).length }))

const activeCaseId = ref(cases[0].id)
const activeCase = computed(() => cases.find(item => item.id === activeCaseId.value) || cases[0])
const casePickerOpen = ref(false)
const pickerCategory = ref('Pose')
const visibleCases = computed(() => cases.filter(item => item.category === pickerCategory.value))
let videoRefs: HTMLVideoElement[] = []
let cleanupSync = (): void => {}

function resultMedia(path: string): string { return `${process.env.BASE_URL}results/${path}` }

function setVideoRef(element: unknown): void {
  if (element instanceof HTMLVideoElement && !videoRefs.includes(element)) videoRefs.push(element)
}

function startSync(): void {
  cleanupSync()
  const videos = videoRefs.filter(video => video.isConnected)
  if (!videos.length) return
  const master = videos[0]
  let animationFrame = 0

  const syncFollowers = (): void => {
    for (const video of videos.slice(1)) {
      if (Math.abs(video.currentTime - master.currentTime) > 0.08) video.currentTime = master.currentTime
      if (video.paused && !master.paused) video.play().catch(() => undefined)
      if (!video.paused && master.paused) video.pause()
    }
  }
  const tick = (): void => {
    syncFollowers()
    animationFrame = requestAnimationFrame(tick)
  }
  const onSeeked = (): void => syncFollowers()
  videos.forEach(video => video.addEventListener('seeked', onSeeked))
  Promise.all(videos.map(video => video.play().catch(() => undefined))).then(() => {
    syncFollowers()
    animationFrame = requestAnimationFrame(tick)
  })
  cleanupSync = () => {
    if (animationFrame) cancelAnimationFrame(animationFrame)
    videos.forEach(video => video.removeEventListener('seeked', onSeeked))
  }
}

async function selectCase(id: string): Promise<void> {
  if (id === activeCaseId.value) return
  cleanupSync()
  videoRefs = []
  activeCaseId.value = id
  await nextTick()
  startSync()
}

function openCasePicker(category: string): void {
  pickerCategory.value = category
  casePickerOpen.value = true
}

function closeCasePicker(): void { casePickerOpen.value = false }

async function openCase(id: string): Promise<void> {
  closeCasePicker()
  await selectCase(id)
}

function onKeydown(event: KeyboardEvent): void {
  if (event.key === 'Escape') closeCasePicker()
}

onMounted(async () => {
  window.addEventListener('keydown', onKeydown)
  await nextTick()
  startSync()
})
onBeforeUnmount(() => {
  cleanupSync()
  window.removeEventListener('keydown', onKeydown)
})
</script>

<style scoped>
.results-section { padding: 82px 0; background: #e9e5dc; }
.shell { width: min(1100px, calc(100% - 48px)); margin: 0 auto; }
.results-heading { display: grid; grid-template-columns: 1.05fr .95fr; align-items: end; gap: 64px; margin-bottom: 30px; }
.kicker { color: var(--green); font-family: 'AliBold'; font-size: 12px; letter-spacing: .16em; text-transform: uppercase; }
h2, h3, p { margin-top: 0; }
h2 { margin-bottom: 0; font-family: 'AliHeavy'; font-size: clamp(36px, 4.4vw, 54px); line-height: 1.02; letter-spacing: -.04em; }
.results-heading > p { margin-bottom: 3px; color: var(--muted); font-size: 14px; line-height: 1.58; }
.application-selector { min-height: 60px; padding: 9px 12px 9px 16px; display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 14px; background: var(--green); color: white; }
.application-selector > span { color: var(--lime); font-family: 'AliBold'; font-size: 9px; letter-spacing: .12em; text-transform: uppercase; white-space: nowrap; }
.application-selector > div { display: flex; gap: 5px; }
.application-selector button { padding: 9px 10px; background: rgba(255,255,255,.08); color: white; border: 1px solid rgba(255,255,255,.2); font-family: 'AliBold'; font-size: 10px; cursor: pointer; transition: color .16s, background .16s; }
.application-selector button:hover, .application-selector button.active { background: var(--lime); color: var(--green); }
.application-selector button small { margin-left: 4px; opacity: .6; font-size: 8px; }
.application-selector > p { margin: 0; display: flex; align-items: flex-end; flex-direction: column; gap: 2px; }
.application-selector > p b { font-family: 'AliBold'; font-size: 11px; }.application-selector > p small { color: rgba(255,255,255,.55); font-size: 8px; letter-spacing: .1em; text-transform: uppercase; }
.comparison-stage { background: #f7f4ed; border-bottom: 5px solid var(--green); }
.case-header { padding: 17px 18px; display: grid; grid-template-columns: .72fr 1.28fr auto; align-items: center; gap: 24px; border-bottom: 1px solid var(--line); }
.case-header p { margin-bottom: 0; color: var(--muted); font-size: 13px; line-height: 1.5; }
.case-header > div > p { margin-bottom: 4px; color: var(--orange); font-size: 9px; letter-spacing: .13em; text-transform: uppercase; }
.case-header h3 { margin: 0; font-family: 'AliHeavy'; font-size: 25px; letter-spacing: -.025em; }
.sync-badge { display: flex; align-items: center; gap: 7px; color: var(--green); font-family: 'AliBold'; font-size: 10px; letter-spacing: .08em; text-transform: uppercase; white-space: nowrap; }
.sync-badge i { width: 7px; height: 7px; background: var(--lime); border: 2px solid var(--green); border-radius: 50%; }
.matrix-scroll { padding: 15px 18px 16px; overflow-x: auto; }
.matrix-grid { min-width: 720px; display: grid; grid-template-columns: repeat(var(--column-count), minmax(0, 250px)); align-items: start; justify-content: center; gap: 10px; }
.matrix-column { min-width: 0; }
.column-title { margin: 0 0 7px; display: flex; align-items: center; gap: 7px; color: var(--muted); font-family: 'AliBold'; font-size: 9px; letter-spacing: .09em; text-transform: uppercase; }
.column-title span { color: var(--orange); }
.matrix-cell { margin-bottom: 9px; padding: 7px; background: #eeebe3; border: 1px solid var(--line); }
.matrix-cell.featured { background: var(--green); border-color: var(--green); box-shadow: 6px 6px 0 var(--lime); }
.cell-title { min-height: 23px; padding: 0 2px 6px; display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.cell-title b { overflow: hidden; font-family: 'AliBold'; font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
.cell-title span { color: var(--muted); font-size: 8px; letter-spacing: .1em; text-transform: uppercase; }
.featured .cell-title b { color: white; }.featured .cell-title span { color: var(--lime); }
.matrix-frame { overflow: hidden; aspect-ratio: 4/3; background: #fff; }
.matrix-frame img, .matrix-frame video { width: 100%; height: 100%; display: block; object-fit: contain; }
.matrix-frame.contain img, .matrix-frame.contain video { object-fit: contain; }
.matrix-frame video { background: #f8f8f6; }
.not-available { width: 100%; height: 100%; display: grid; place-items: center; color: var(--muted); font-size: 10px; letter-spacing: .1em; text-transform: uppercase; }
.case-footer { padding: 11px 18px; display: flex; align-items: center; justify-content: space-between; gap: 28px; border-top: 1px solid var(--line); }
.case-footer p { margin: 0; color: var(--muted); font-size: 12px; line-height: 1.5; }
.case-footer p span { margin-right: 8px; color: var(--orange); font-family: 'AliBold'; font-size: 9px; letter-spacing: .1em; text-transform: uppercase; }
.case-footer > b { color: var(--green); font-family: 'AliBold'; font-size: 11px; white-space: nowrap; }
.case-picker-backdrop { position: fixed; z-index: 100; inset: 0; padding: 38px; display: grid; place-items: center; overflow-y: auto; background: rgba(11,28,22,.82); backdrop-filter: blur(5px); }
.case-picker { width: min(1080px, 100%); max-height: calc(100vh - 76px); overflow-y: auto; background: #f7f4ed; box-shadow: 14px 14px 0 var(--lime); }
.case-picker > header { position: sticky; z-index: 2; top: 0; padding: 18px 20px; display: grid; grid-template-columns: 1fr auto auto; align-items: center; gap: 20px; background: var(--green); color: white; }
.case-picker header p { margin: 0 0 3px; color: var(--lime); font-family: 'AliBold'; font-size: 9px; letter-spacing: .13em; text-transform: uppercase; }.case-picker header h3 { margin: 0; font-family: 'AliHeavy'; font-size: 27px; }.case-picker header > span { color: rgba(255,255,255,.62); font-size: 10px; }.case-picker header > button { width: 36px; height: 36px; background: transparent; color: white; border: 1px solid rgba(255,255,255,.36); font-size: 24px; line-height: 1; cursor: pointer; }
.case-gallery { padding: 18px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.case-gallery > button { min-width: 0; padding: 8px; background: #eeebe3; color: var(--ink); border: 1px solid var(--line); text-align: left; cursor: pointer; transition: transform .16s, border-color .16s; }.case-gallery > button:hover { transform: translateY(-3px); border-color: var(--green); }.case-gallery > button.selected { border: 2px solid var(--green); }
.case-thumb { position: relative; overflow: hidden; aspect-ratio: 4/3; background: white; }.case-thumb img { width: 100%; height: 100%; display: block; object-fit: contain; }.case-thumb > span { position: absolute; right: 7px; bottom: 7px; padding: 4px 6px; background: var(--green); color: var(--lime); font-family: 'AliBold'; font-size: 8px; }
.case-gallery button > p { margin: 9px 2px 4px; display: flex; flex-direction: column; gap: 2px; }.case-gallery button small { color: var(--orange); font-size: 8px; letter-spacing: .1em; text-transform: uppercase; }.case-gallery button b { overflow: hidden; font-family: 'AliBold'; font-size: 14px; text-overflow: ellipsis; white-space: nowrap; }.case-gallery button > i { margin: 0 2px 2px; display: block; color: var(--muted); font-size: 9px; font-style: normal; }
@media (max-width: 900px) { .results-heading { grid-template-columns: 1fr; gap: 20px; }.application-selector { grid-template-columns: 1fr; }.application-selector > div { overflow-x: auto; }.application-selector > p { display: none; }.case-header { grid-template-columns: 1fr auto; gap: 10px 20px; }.case-header > p { grid-column: 1/-1; grid-row: 2; }.matrix-scroll { padding-right: 14px; padding-left: 14px; }.case-gallery { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 640px) { .results-section { padding: 64px 0; }.shell { width: calc(100% - 28px); }.application-selector { padding: 12px 14px; }.application-selector > div { width: 100%; }.application-selector button { flex: 0 0 auto; }.case-header { padding: 16px 14px; grid-template-columns: 1fr; }.case-header > p { grid-column: auto; grid-row: auto; }.sync-badge { margin-top: 3px; }.matrix-grid { min-width: 700px; }.case-footer { padding: 12px 14px; align-items: flex-start; flex-direction: column; gap: 6px; }.case-picker-backdrop { padding: 14px; place-items: start center; }.case-picker { max-height: calc(100vh - 28px); box-shadow: 7px 7px 0 var(--lime); }.case-picker > header { grid-template-columns: 1fr auto; }.case-picker header > span { display: none; }.case-gallery { padding: 12px; grid-template-columns: repeat(2, 1fr); gap: 9px; } }
</style>
