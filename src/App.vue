<template><router-view /></template>

<style>
@import './assets/font.css';
:root { --ink: #15231e; --muted: #647069; --paper: #f7f4ed; --line: rgba(21,35,30,.14); --green: #184f3b; --lime: #cde56b; --orange: #ee7d4a; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
html, body, #app { margin: 0; min-height: 100%; }
body { overflow-x: hidden; background: var(--paper); color: var(--ink); font-family: 'AliRegular', Inter, system-ui, sans-serif; -webkit-font-smoothing: antialiased; }
a { color: inherit; }
</style>
