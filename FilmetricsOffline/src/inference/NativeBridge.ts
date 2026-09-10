/**
 * Fully offline JS inference via onnxruntime-react-native.
 * No custom Kotlin, no backend. Model: src/assets/anti_spoofing.onnx (4-class 15ep)
 * copied to DocumentDirectory on first init, then InferenceSession.create(path).
 */
import { Image } from 'react-native'
import RNFS from 'react-native-fs'
import { InferenceSession, Tensor } from 'onnxruntime-react-native'
import jpeg from 'jpeg-js'
import { Buffer } from 'buffer'
import { InferenceResult, FaceDetection } from '../types'

const MODEL_ASSET = require('../assets/anti_spoofing.onnx')
const MODEL_NAME = 'anti_spoofing.onnx'
const INPUT_SIZE = 640
const CONF_THRESHOLD = 0.4
const NMS_IOU = 0.45
const LABELS = ['real_without_mask', 'real_with_mask', 'fake_without_mask', 'fake_with_mask'] as const
// Overlay in Home is fixed 640x480 preview space
const PREVIEW_W = 640
const PREVIEW_H = 480

let session: InferenceSession | null = null
let initialized = false

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

function iou(a: { x1: number; y1: number; x2: number; y2: number }, b: { x1: number; y1: number; x2: number; y2: number }): number {
  const ix1 = Math.max(a.x1, b.x1)
  const iy1 = Math.max(a.y1, b.y1)
  const ix2 = Math.min(a.x2, b.x2)
  const iy2 = Math.min(a.y2, b.y2)
  const iw = Math.max(0, ix2 - ix1)
  const ih = Math.max(0, iy2 - iy1)
  const inter = iw * ih
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
  return union <= 0 ? 0 : inter / union
}

async function ensureModelFile(): Promise<string> {
  const dest = `${RNFS.DocumentDirectoryPath}/${MODEL_NAME}`
  const exists = await RNFS.exists(dest)
  if (exists) {
    const stat = await RNFS.stat(dest)
    console.log(`[ORT] model exists: ${dest} size=${stat.size}`)
    if (Number(stat.size) > 1024 * 1024) return dest
    console.log('[ORT] model file too small, re-copying...')
    await RNFS.unlink(dest).catch(() => {})
  }
  // Android: copy from native assets (android/app/src/main/assets/)
  try {
    console.log('[ORT] copying model from native assets...')
    await RNFS.copyFileAssets(MODEL_NAME, dest)
    const stat = await RNFS.stat(dest)
    console.log(`[ORT] copied size=${stat.size}`)
    return dest
  } catch (e) {
    console.log('[ORT] copyFileAssets failed, trying Metro download fallback', e)
    // Fallback dev: download bundled asset via Metro server
    const asset = Image.resolveAssetSource(MODEL_ASSET)
    if (!asset?.uri) throw new Error('Cannot resolve model asset')
    console.log('[ORT] downloading from', asset.uri)
    const dl = RNFS.downloadFile({ fromUrl: asset.uri, toFile: dest })
    const res = await dl.promise
    if (res.statusCode !== 200) throw new Error(`Model download HTTP ${res.statusCode}`)
    return dest
  }
}

async function photoToTensor(uri: string): Promise<{ tensor: Tensor; scale: number; padX: number; padY: number; origW: number; origH: number }> {
  const path = uri.replace('file://', '')
  const b64 = await RNFS.readFile(path, 'base64')
  const bytes = Buffer.from(b64, 'base64')
  const decoded = jpeg.decode(bytes, { useTArray: true })
  const origW = decoded.width
  const origH = decoded.height
  const scale = Math.min(INPUT_SIZE / origW, INPUT_SIZE / origH)
  const nw = Math.round(origW * scale)
  const nh = Math.round(origH * scale)
  const padX = Math.floor((INPUT_SIZE - nw) / 2)
  const padY = Math.floor((INPUT_SIZE - nh) / 2)
  // letterbox into RGB buffer filled with 114
  const rgb = new Uint8Array(INPUT_SIZE * INPUT_SIZE * 3).fill(114)
  const src = decoded.data // RGBA
  for (let y = 0; y < nh; y++) {
    const srcY = Math.min(origH - 1, Math.floor((y / nh) * origH))
    for (let x = 0; x < nw; x++) {
      const srcX = Math.min(origW - 1, Math.floor((x / nw) * origW))
      const si = (srcY * origW + srcX) * 4
      const di = ((y + padY) * INPUT_SIZE + (x + padX)) * 3
      rgb[di] = src[si]
      rgb[di + 1] = src[si + 1]
      rgb[di + 2] = src[si + 2]
    }
  }
  const chw = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE)
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
      chw[c * INPUT_SIZE * INPUT_SIZE + i] = rgb[i * 3 + c] / 255.0
    }
  }
  const tensor = new Tensor('float32', chw, [1, 3, INPUT_SIZE, INPUT_SIZE])
  return { tensor, scale, padX, padY, origW, origH }
}

function decodeYOLOv8(
  data: Float32Array | number[],
  origW: number,
  origH: number,
  scale: number,
  padX: number,
  padY: number
): FaceDetection[] {
  const anchors = 8400
  type Box = { label: (typeof LABELS)[number]; conf: number; x1: number; y1: number; x2: number; y2: number }
  const cands: Box[] = []
  for (let i = 0; i < anchors; i++) {
    const cx = data[0 * anchors + i] as number
    const cy = data[1 * anchors + i] as number
    const w = data[2 * anchors + i] as number
    const h = data[3 * anchors + i] as number
    let best = 0
    let bestScore = -Infinity
    for (let c = 0; c < 4; c++) {
      const s = sigmoid(data[(4 + c) * anchors + i] as number)
      if (s > bestScore) {
        bestScore = s
        best = c
      }
    }
    if (bestScore < CONF_THRESHOLD) continue
    const x1n = cx - w / 2
    const y1n = cy - h / 2
    const x2n = cx + w / 2
    const y2n = cy + h / 2
    // back to original photo pixels
    const x1o = Math.min(Math.max((x1n - padX) / scale, 0), origW)
    const y1o = Math.min(Math.max((y1n - padY) / scale, 0), origH)
    const x2o = Math.min(Math.max((x2n - padX) / scale, 0), origW)
    const y2o = Math.min(Math.max((y2n - padY) / scale, 0), origH)
    if (x2o <= x1o || y2o <= y1o) continue
    // scale to 640x480 preview space used by FaceOverlay
    cands.push({
      label: LABELS[best],
      conf: bestScore,
      x1: (x1o / origW) * PREVIEW_W,
      y1: (y1o / origH) * PREVIEW_H,
      x2: (x2o / origW) * PREVIEW_W,
      y2: (y2o / origH) * PREVIEW_H,
    })
  }
  // NMS
  cands.sort((a, b) => b.conf - a.conf)
  const kept: Box[] = []
  while (cands.length > 0) {
    const best = cands.shift()!
    kept.push(best)
    for (let i = cands.length - 1; i >= 0; i--) {
      if (iou(best, cands[i]) > NMS_IOU) cands.splice(i, 1)
    }
  }
  return kept.map((k) => ({
    label: k.label,
    confidence: k.conf,
    bbox: {
      x: Math.round(k.x1),
      y: Math.round(k.y1),
      w: Math.round(k.x2 - k.x1),
      h: Math.round(k.y2 - k.y1),
    },
  }))
}

class YoloBridge {
  async initModel(): Promise<boolean> {
    try {
      const path = await ensureModelFile()
      console.log('[ORT] creating session from', path)
      // NOTE: this ORT build needs file:// scheme (bare path throws "No content provider")
      session = await InferenceSession.create(`file://${path}`)
      initialized = true
      console.log('[ORT] session ready, inputs=', session.inputNames)
      return true
    } catch (error) {
      console.error('JS ORT init failed:', error)
      throw error
    }
  }

  async runInference(frame: { uri: string } | any): Promise<InferenceResult> {
    if (!initialized || !session) {
      throw new Error('Model not initialized. Call initModel() first.')
    }
    const uri = frame?.uri as string | undefined
    if (!uri) throw new Error('runInference expects { uri } from takePhoto (offline JS)')
    const { tensor, scale, padX, padY, origW, origH } = await photoToTensor(uri)
    const feeds: Record<string, Tensor> = {}
    feeds[session.inputNames[0]] = tensor
    const out = await session.run(feeds)
    const firstKey = session.outputNames[0]
    const t = out[firstKey] as unknown as { data: Float32Array | number[] }
    const faces = decodeYOLOv8(t.data, origW, origH, scale, padX, padY)
    return { faces }
  }

  async release(): Promise<void> {
    session = null
    initialized = false
  }

  isInitialized(): boolean {
    return initialized
  }
}

export const yoloBridge = new YoloBridge()
export default yoloBridge
