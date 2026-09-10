import React, { useEffect, useState, forwardRef, useRef, useImperativeHandle } from 'react'
import { View, StyleSheet, Alert } from 'react-native'
import { Camera, useCameraDevice } from 'react-native-vision-camera'

export interface CameraViewHandle {
  takePhoto: () => Promise<{ path: string }>
}

interface CameraViewProps {
  isActive: boolean
}

// Offline v1: photo-mode (takePhoto file -> YoloModule {uri}).
// Live pixel-buffer worklets can't serialize to the bridge, so we poll takePhoto ~1/s.
// Fully on-device, no backend.
const CameraView = forwardRef<CameraViewHandle, CameraViewProps>(({ isActive }, ref) => {
  const device = useCameraDevice('front')
  const cameraRef = useRef<Camera>(null)
  const [hasPermission, setHasPermission] = useState<boolean | null>(null)

  useEffect(() => {
    checkCameraPermission()
  }, [])

  useImperativeHandle(ref, () => ({
    takePhoto: async () => {
      if (!cameraRef.current) throw new Error('Camera not ready')
      const photo = await cameraRef.current.takePhoto({ flash: 'off' })
      return { path: photo.path }
    },
  }))

  const checkCameraPermission = async () => {
    const status = await Camera.getCameraPermissionStatus()
    if (status === 'not-determined') {
      const newStatus = await Camera.requestCameraPermission()
      setHasPermission(newStatus === 'granted')
    } else {
      setHasPermission(status === 'granted')
    }

    if (status !== 'granted') {
      Alert.alert('Permission Required', 'Camera permission is required for face detection')
    }
  }

  if (hasPermission === null) {
    return <View style={styles.container} />
  }

  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <View style={styles.errorContainer} />
      </View>
    )
  }

  if (!device) {
    return <View style={styles.container} />
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isActive}
        photo={true}
      />
    </View>
  )
})

export default CameraView

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
})
