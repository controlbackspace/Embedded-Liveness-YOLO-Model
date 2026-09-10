import React, { useEffect, useState, forwardRef, useRef, useImperativeHandle } from 'react'
import { View, StyleSheet, Text, TouchableOpacity, AppState } from 'react-native'
import { Camera, CameraType } from 'expo-camera'

export interface CameraViewHandle {
  takePhoto: () => Promise<{ path: string }>
}

interface CameraViewProps {
  isActive: boolean
  facing?: 'front' | 'back'
}

// Offline v1 on expo-camera (bare RN + expo modules).
// takePictureAsync file -> ORT JS {uri}. Fully on-device, no backend.
// (Replaced VisionCamera v3 whose session kept dropping the photo output.)
const CameraView = forwardRef<CameraViewHandle, CameraViewProps>(({ isActive, facing = 'front' }, ref) => {
  const cameraRef = useRef<Camera>(null)
  const [hasPermission, setHasPermission] = useState<boolean | null>(null)
  const [ready, setReady] = useState(false)
  // Small capture = fast base64 + fast JS decode. 12MP takes 30s in JS; 640x480 takes ~1s.
  const [pictureSize, setPictureSize] = useState<string | undefined>(undefined)

  useEffect(() => {
    checkCameraPermission()
    const sub = AppState.addEventListener('change', (s) => {
      if (s === 'active') checkCameraPermission()
    })
    return () => sub.remove()
  }, [])

  const pickPictureSize = async () => {
    try {
      // v13: instance method on Camera ref (static Camera.getAvailablePictureSizesAsync doesn't exist)
      const refNow = cameraRef.current as any
      const sizes: string[] | undefined = await refNow?.getAvailablePictureSizesAsync?.()
      console.log('[CAM] available sizes:', (sizes ?? []).join(','))
      const list = sizes ?? []
      // prefer 640x480, else smallest available (fast JS decode)
      const area = (s: string) => {
        const m = s.split('x').map(Number)
        return m.length === 2 && m.every(Number.isFinite) ? m[0] * m[1] : Infinity
      }
      const exact = list.find((s) => s === '640x480')
      const sorted = [...list].sort((a, b) => area(a) - area(b))
      const pick = exact ?? sorted[0]
      if (pick) {
        console.log('[CAM] using pictureSize:', pick)
        setPictureSize(pick)
      } else {
        console.log('[CAM] no sizes reported, using default (full res)')
      }
    } catch (e) {
      console.log('[CAM] pictureSizes query failed, using default', e)
    }
  }

  useImperativeHandle(ref, () => ({
    takePhoto: async () => {
      if (!cameraRef.current) throw new Error('Camera not ready')
      if (!ready) throw new Error('Camera initializing…')
      // Timeout: expo takePictureAsync can hang forever on some HALs — fail loud
      const capture = cameraRef.current.takePictureAsync({
        quality: 0.5,
        skipProcessing: true,
        shutterSound: false,
      })
      const timeout = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('takePhoto timeout 10s')), 10000)
      )
      const photo = await Promise.race([capture, timeout])
      if (!photo?.uri) throw new Error('No photo captured')
      return { path: photo.uri }
    },
  }))

  const checkCameraPermission = async () => {
    const cur = await Camera.getCameraPermissionsAsync()
    if (cur.granted) {
      setHasPermission(true)
      return
    }
    const req = await Camera.requestCameraPermissionsAsync()
    setHasPermission(req.granted)
  }

  if (hasPermission === null) {
    return (
      <View style={styles.container}>
        <Text style={styles.debug}>Requesting camera permission…</Text>
      </View>
    )
  }

  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <Text style={styles.debug}>Camera permission denied.{'\n'}Grant in Settings, then tap retry.</Text>
        <TouchableOpacity style={styles.btn} onPress={checkCameraPermission}>
          <Text style={styles.btnText}>Retry permission</Text>
        </TouchableOpacity>
      </View>
    )
  }

  return (
    <View style={styles.container}>
      {isActive && (
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          type={facing === 'front' ? CameraType.front : CameraType.back}
          pictureSize={pictureSize}
          onCameraReady={() => {
            setReady(true)
            pickPictureSize() // ref exists now; shrinks captures for fast scans
          }}
          onMountError={(e) => console.error('[CAM] mount error', e.message)}
        />
      )}
      {!ready && <Text style={styles.debug}>Starting {facing} camera…</Text>}
    </View>
  )
})

export default CameraView

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: 'black', justifyContent: 'center', alignItems: 'center' },
  debug: { color: '#9ca3af', textAlign: 'center', padding: 16 },
  btn: { marginTop: 8, backgroundColor: '#1f2937', padding: 10, borderRadius: 8 },
  btnText: { color: '#fff', fontWeight: 'bold' },
})
