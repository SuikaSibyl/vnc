const { defineConfig } = require('@vue/cli-service')

module.exports = defineConfig({
  transpileDependencies: true,
  publicPath: process.env.NODE_ENV === 'production' ? '/vnc/' : '/',
  configureWebpack: {
    watchOptions: {
      ignored: /node_modules/,
      poll: 1000,
    },
  },
  chainWebpack: (config) => {
    // The standalone `npm run typecheck` command performs the same validation.
    // Avoid an extra IPC worker in development so `npm run serve` also works on
    // machines whose per-user inotify instance limit is already exhausted.
    if (process.env.NODE_ENV === 'development') {
      config.plugins.delete('fork-ts-checker')
    }
  },
})
