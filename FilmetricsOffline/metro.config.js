const { getDefaultConfig } = require('expo/metro-config');
const { mergeConfig } = require('@react-native/metro-config');

/**
 * Metro configuration
 * https://facebook.github.io/metro/docs/configuration
 *
 * @type {import('metro-config').MetroConfig}
 */
const defaultConfig = getDefaultConfig(__dirname);

const config = {
  resolver: {
    // extend defaults (png/jpg/ttf/...) — don't replace, or bundle 500s
    assetExts: [...defaultConfig.resolver.assetExts, 'onnx', 'pt', 'bin', 'tflite'],
  },
};

module.exports = mergeConfig(defaultConfig, config);
