import React, { useState, useEffect, useRef } from 'react'
import { View, StyleSheet, Text, StatusBar, Alert, TouchableOpacity } from 'react-native'
import CameraView, { CameraViewHandle } from '../camera/CameraView'
import FaceOverlay from '../overlay/FaceOverlay'
import { detector } from '../inference/Detector'
import { FaceDetection, isRealLabel, isFakeLabel, hasMaskLabel } from '../types'

const VIDEO_WIDTH = 640
const VIDEO_HEIGHT = 480
const OFFLINE_INTERVAL_MS = 1500 // photo -> on-device ONNX -> overlay, no backend

export default function Home() {
  const [detections, setDetections] = useState<FaceDetection[]>([])
  const [isInitialized, setIsInitialized] = useState(false)
  const [isActive, setIsActive] = useState(true)
  const [facing, setFacing] = useState<'front' | 'back'>('front')
  const [scanning, setScanning] = useState(false) // single-shot busy flag
  const [error, setError] = useState<string | null>(null)
  const cameraRef = useRef<CameraViewHandle>(null)
  const busyRef = useRef(false)

  useEffect(() => {
    initializeDetector()
    return () => {
      detector.release()
    }
  }, [])

  // Tap-to-scan: one photo -> on-device ORT -> overlay. No continuous loop.
  const scanOnce = async () => {
    if (busyRef.current || !cameraRef.current || scanning) return
    setScanning(true)
    busyRef.current = true
    try {
      const photo = await cameraRef.current.takePhoto()
      const uri = photo.path.startsWith('file://') ? photo.path : `file://${photo.path}`
      console.log(`[SCAN] photo file: ${uri}`)
      const results = await detector.detect({ uri, mirror: facing === 'front' })
      setDetections(results)
      if (results.length > 0) setError(null)
    } catch (e: any) {
      const msg = e?.message ?? 'offline inference failed'
      console.error('[SCAN] failed:', msg, e)
      setError(msg)
    } finally {
      busyRef.current = false
      setScanning(false)
    }
  }

  const initializeDetector = async () => {
    try {
      const success = await detector.initialize()
      setIsInitialized(success)
      if (!success) {
        Alert.alert('Error', 'Failed to load offline model (mobile/assets/anti_spoofing.onnx)')
      }
    } catch (error) {
      console.error('Initialization error:', error)
      Alert.alert('Error', 'Failed to initialize detection model')
    }
  }

  const getStatusText = () => {
    if (!isInitialized) {
      return 'Loading offline model...'
    }
    if (detections.length === 0) {
      return 'No faces detected (offline)'
    }
    const hasReal = detections.some((d) => isRealLabel(d.label))
    const hasFake = detections.some((d) => isFakeLabel(d.label))
    const hasMask = detections.some((d) => hasMaskLabel(d.label))
    if (hasReal && !hasFake) {
      return hasMask ? 'REAL 😷 (offline)' : 'REAL (offline)'
    } else if (hasFake && !hasReal) {
      return hasMask ? 'FAKE 😷 (offline)' : 'FAKE (offline)'
    } else {
      return hasMask ? 'MIXED 😷 (offline)' : 'MIXED (offline)'
    }
  }

  const getStatusColor = () => {
    if (detections.length === 0) {
      return '#6b7280'
    }
    const hasReal = detections.some((d) => isRealLabel(d.label))
    const hasFake = detections.some((d) => isFakeLabel(d.label))
    if (hasReal && !hasFake) {
      return '#10b981'
    } else if (hasFake && !hasReal) {
      return '#ef4444'
    } else {
      return '#f59e0b'
    }
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      <View style={styles.cameraContainer}>
        <CameraView ref={cameraRef} isActive={isActive} facing={facing} />
        <FaceOverlay detections={detections} videoWidth={VIDEO_WIDTH} videoHeight={VIDEO_HEIGHT} />
      </View>
      <View style={styles.statusContainer}>
        <View style={[styles.statusBadge, { backgroundColor: getStatusColor() }]}>
          <Text style={styles.statusText}>{getStatusText()}</Text>
        </View>
        {error && <Text style={styles.error}>{error}</Text>}
        {detections.length > 0 && (
          <Text style={styles.detectionCount}>
            {detections.length} face{detections.length !== 1 ? 's' : ''} detected
          </Text>
        )}
        <TouchableOpacity
          style={[styles.btn, styles.scanBtn, (!isInitialized || scanning) && styles.btnDisabled]}
          onPress={scanOnce}
          disabled={!isInitialized || scanning}
        >
          <Text style={styles.btnText}>
            {!isInitialized ? 'Loading model…' : scanning ? 'Scanning…' : '📷 TAP TO SCAN'}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.btn}
          onPress={() => {
            setDetections([])
            setFacing((f) => (f === 'front' ? 'back' : 'front'))
          }}
        >
          <Text style={styles.btnText}>Flip to {facing === 'front' ? 'back' : 'front'}</Text>
        </TouchableOpacity>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  cameraContainer: { flex: 1, position: 'relative' },
  statusContainer: { padding: 20, alignItems: 'center', backgroundColor: 'rgba(0, 0, 0, 0.8)' },
  statusBadge: { paddingHorizontal: 24, paddingVertical: 12, borderRadius: 24, marginBottom: 8 },
  statusText: { color: 'white', fontSize: 18, fontWeight: 'bold' },
  detectionCount: { color: '#9ca3af', fontSize: 14 },
  error: { color: '#ef4444', fontSize: 12, marginBottom: 8 },
  btn: { marginTop: 10, backgroundColor: '#1f2937', padding: 10, borderRadius: 8 },
  scanBtn: { backgroundColor: '#065f46', paddingHorizontal: 32, paddingVertical: 14 },
  btnDisabled: { opacity: 0.5 },
  btnText: { color: '#fff', fontWeight: 'bold' },
})
