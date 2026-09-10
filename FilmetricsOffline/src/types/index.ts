/**
 * TypeScript types for React Native app
 */

export interface BoundingBox {
  x: number
  y: number
  w: number
  h: number
}

export type LivenessLabel =
  | 'real_without_mask'
  | 'real_with_mask'
  | 'fake_without_mask'
  | 'fake_with_mask'
  // backwards-compat with old 2-class model
  | 'real'
  | 'fake'

export interface FaceDetection {
  label: LivenessLabel
  confidence: number
  bbox: BoundingBox
}

export const isRealLabel = (label: string): boolean =>
  label === 'real' || label === 'real_without_mask' || label === 'real_with_mask'

export const isFakeLabel = (label: string): boolean =>
  label === 'fake' || label === 'fake_without_mask' || label === 'fake_with_mask'

export const hasMaskLabel = (label: string): boolean =>
  label === 'real_with_mask' || label === 'fake_with_mask'

export interface InferenceResult {
  faces: FaceDetection[]
  latency_ms?: number
}
