import { describe, expect, it } from 'vitest'
import {
  formatCurrency,
  formatNumber,
  formatPercent,
  formatScore,
  MODEL_OPTIONS,
} from './format.js'

describe('formatNumber', () => {
  it('formats thousands separators', () => {
    expect(formatNumber(284807)).toBe('284,807')
  })
  it('returns a dash for missing values', () => {
    expect(formatNumber(null)).toBe('-')
  })
})

describe('formatCurrency', () => {
  it('formats an amount', () => {
    expect(formatCurrency(122.21)).toBe('$122.21')
  })
})

describe('formatPercent', () => {
  it('formats a ratio as a percentage', () => {
    expect(formatPercent(0.8469)).toBe('84.7%')
  })
  it('reports n/a when undefined', () => {
    expect(formatPercent(null)).toBe('n/a')
  })
})

describe('formatScore', () => {
  it('renders four decimals', () => {
    expect(formatScore(0.21155192)).toBe('0.2116')
  })
})

describe('MODEL_OPTIONS', () => {
  it('offers the three trained models', () => {
    expect(MODEL_OPTIONS.map((m) => m.value)).toEqual([
      'xgboost',
      'random_forest',
      'isolation_forest',
    ])
  })
})
