/** Small formatting helpers for the results table. */

export const MODEL_OPTIONS = [
  { value: 'xgboost', label: 'XGBoost' },
  { value: 'random_forest', label: 'Random Forest' },
  { value: 'isolation_forest', label: 'Isolation Forest' },
]

export function formatNumber(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-'
  return Number(value).toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

export function formatCurrency(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '-'
  return `$${Number(value).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`
}

export function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'n/a'
  return `${(Number(value) * 100).toFixed(digits)}%`
}

export function formatScore(score) {
  if (score === null || score === undefined) return '-'
  return Number(score).toFixed(4)
}

/** Download a result set as a CSV file in the browser. */
export function downloadCsv(filename, rows, columns) {
  if (!rows || rows.length === 0) return
  const cols = columns || Object.keys(rows[0])
  const escape = (value) => {
    if (value === null || value === undefined) return ''
    const str = String(value)
    return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str
  }
  const lines = [cols.join(',')]
  rows.forEach((row) => {
    lines.push(cols.map((c) => escape(row[c])).join(','))
  })
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}
