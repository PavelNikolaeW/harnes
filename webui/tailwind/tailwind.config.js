/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "../../src/harnes/webui/templates/**/*.html",
  ],
  theme: {
    extend: {
      colors: {
        ink:        '#1a1a1a',
        soft:       '#555',
        cream:      '#fdfdfb',
        paper:      '#f3f1ec',
        decision:   '#fffaee',
        edge:       '#d8d4cc',
        edgex:      '#888',
        accent:     '#664433',
        accentSoft: '#aa7788',
        link:       '#334455',
      },
      fontFamily: {
        sans: ['-apple-system','BlinkMacSystemFont','"Segoe UI"','"Helvetica Neue"','Arial','sans-serif'],
        mono: ['"SF Mono"','Menlo','Consolas','monospace'],
      },
    },
  },
  plugins: [],
}
