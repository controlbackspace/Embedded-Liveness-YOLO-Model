import React from 'react'
import { View, StyleSheet, Text } from 'react-native'
import { FaceDetection, isRealLabel, hasMaskLabel } from '../types'

interface FaceOverlayProps {
  detections: FaceDetection[]
  videoWidth: number
  videoHeight: number
}

export default function FaceOverlay({
  detections,
  videoWidth,
  videoHeight,
}: FaceOverlayProps) {
  return (
    <View style={[styles.container, { width: videoWidth, height: videoHeight }]}>
      {detections.map((detection, index) => {
        const { bbox, label, confidence } = detection
        const isReal = isRealLabel(label)
        const masked = hasMaskLabel(label)
        const shortLabel = label
          .replace('real_without_mask', 'REAL')
          .replace('real_with_mask', 'REAL+MASK')
          .replace('fake_without_mask', 'FAKE')
          .replace('fake_with_mask', 'FAKE+MASK')
          .toUpperCase()
        const borderColor = isReal ? '#10b981' : '#ef4444'

        return (
          <View
            key={index}
            style={[
              styles.box,
              {
                left: bbox.x,
                top: bbox.y,
                width: bbox.w,
                height: bbox.h,
                borderColor,
              },
            ]}
          >
            <View
              style={[
                styles.label,
                {
                  backgroundColor: isReal ? '#10b981' : '#ef4444',
                },
              ]}
            >
              <Text style={styles.labelText}>
                {shortLabel} {Math.round(confidence * 100)}%{masked ? ' 😷' : ''}
              </Text>
            </View>
          </View>
        )
      })}
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 0,
    left: 0,
    pointerEvents: 'none',
  },
  box: {
    position: 'absolute',
    borderWidth: 2,
    borderRadius: 4,
  },
  label: {
    position: 'absolute',
    top: -28,
    left: 0,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  labelText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
})
