import '@testing-library/jest-dom/vitest'

// jsdom does not implement matchMedia; Recharts/ResponsiveContainer may use it.
if (!window.matchMedia) {
  window.matchMedia = (query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  })
}

// jsdom has no layout engine; give containers a non-zero size so Recharts
// renders instead of bailing out.
if (!window.ResizeObserver) {
  window.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}
