import { useCallback, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  FileSpreadsheet,
  Loader2,
  ShieldCheck,
  UploadCloud,
  X,
} from 'lucide-react'
import { api } from '../services/api.js'
import {
  MODEL_OPTIONS,
  downloadCsv,
  formatCurrency,
  formatNumber,
  formatPercent,
  formatScore,
} from '../lib/format.js'

const PAGE_SIZE = 50

function Stat({ label, value, tone = 'default' }) {
  const toneClass = {
    default: 'text-zinc-100',
    fraud: 'text-red-500',
    legit: 'text-emerald-400',
  }[tone]
  return (
    <div className="card p-5">
      <p className="text-xs font-medium uppercase tracking-wide text-zinc-500">{label}</p>
      <p className={`mt-2 truncate text-3xl font-semibold tabular-nums ${toneClass}`}>{value}</p>
    </div>
  )
}

function Metric({ label, value }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-zinc-500">{label}</p>
      <p className="mt-1 text-lg font-semibold tabular-nums text-zinc-100">{value}</p>
    </div>
  )
}

function Verdict({ prediction }) {
  if (prediction === 1) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-red-900 bg-red-950/40 px-2.5 py-1 text-xs font-medium text-red-400">
        <AlertTriangle className="h-3.5 w-3.5" />
        Fraud
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-zinc-800 bg-zinc-900 px-2.5 py-1 text-xs font-medium text-zinc-400">
      <CheckCircle2 className="h-3.5 w-3.5" />
      Legitimate
    </span>
  )
}

export default function Analyze({ health, onHome }) {
  const [file, setFile] = useState(null)
  const [model, setModel] = useState('xgboost')
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState('all')
  const [page, setPage] = useState(1)
  const inputRef = useRef(null)

  const pickFile = useCallback((next) => {
    setResult(null)
    setError(null)
    setPage(1)
    setFilter('all')
    setFile(next || null)
  }, [])

  const onDrop = (event) => {
    event.preventDefault()
    setDragging(false)
    const dropped = event.dataTransfer?.files?.[0]
    if (dropped) pickFile(dropped)
  }

  const runAnalyze = async () => {
    if (!file) {
      setError('Choose a CSV file first.')
      return
    }
    setBusy(true)
    setError(null)
    setResult(null)
    setProgress(0)
    try {
      const data = await api.analyze(file, model, setProgress)
      setResult(data)
      setPage(1)
      setFilter('all')
    } catch (err) {
      setError(err.message || 'Analysis failed.')
    } finally {
      setBusy(false)
    }
  }

  const rows = result?.results ?? []
  const counts = useMemo(
    () => ({
      all: rows.length,
      fraud: rows.filter((r) => r.prediction === 1).length,
      legit: rows.filter((r) => r.prediction === 0).length,
    }),
    [rows],
  )

  const filtered = useMemo(
    () =>
      rows.filter(
        (r) =>
          filter === 'all' || (filter === 'fraud' ? r.prediction === 1 : r.prediction === 0),
      ),
    [rows, filter],
  )

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const safePage = Math.min(page, pageCount)
  const pageRows = filtered.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE)
  const hasActual = rows.some((r) => r.actual_class !== null && r.actual_class !== undefined)

  const exportCsv = () =>
    downloadCsv(
      'fraudshield-results.csv',
      rows.map((r) => ({
        row: r.row,
        amount: r.amount,
        verdict: r.label,
        score: r.score,
        actual_class: r.actual_class ?? '',
      })),
      ['row', 'amount', 'verdict', 'score', 'actual_class'],
    )

  const filters = [
    ['all', 'All'],
    ['fraud', 'Fraud'],
    ['legit', 'Legitimate'],
  ]

  return (
    <div className="space-y-8">
      <section>
        <button type="button" className="btn-ghost mb-4 px-3 py-1.5 text-xs" onClick={onHome}>
          <ArrowLeft className="h-3.5 w-3.5" />
          Home
        </button>
        <h2 className="text-2xl font-semibold tracking-tight">Upload the dataset</h2>
        <p className="mt-1 max-w-2xl text-sm text-zinc-400">
          Upload a transaction CSV and every row is classified as{' '}
          <span className="text-red-400">fraud</span> or{' '}
          <span className="text-emerald-400">legitimate</span>. The model uses the factors it was
          trained on — <span className="text-zinc-200">Time</span>,{' '}
          <span className="text-zinc-200">V1–V28</span> and{' '}
          <span className="text-zinc-200">Amount</span> — but your column names can be anything
          (any case, or aliases like <code className="text-zinc-300">amount</code>,{' '}
          <code className="text-zinc-300">timestamp</code>,{' '}
          <code className="text-zinc-300">is_fraud</code>). A{' '}
          <code className="text-zinc-300">Class</code> column is never used as an input — it is
          only shown for comparison when present.
        </p>
      </section>

      {health && !health.models_trained && (
        <div className="rounded-2xl border border-amber-900/50 bg-amber-950/20 p-5 text-sm text-amber-400">
          No trained model found. Run{' '}
          <code className="rounded bg-black/50 px-1.5 py-0.5 text-zinc-200">
            python training/train_models.py
          </code>{' '}
          in the <span className="text-zinc-200">backend</span> folder, then reload this page.
        </div>
      )}

      {/* ---------------------------------------------------------- upload */}
      <section className="card p-6">
        <div
          onDragOver={(event) => {
            event.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border border-dashed px-6 py-10 text-center transition-colors ${
            dragging ? 'border-white bg-zinc-900' : 'border-zinc-800 hover:border-zinc-600'
          }`}
        >
          {file ? (
            <FileSpreadsheet className="h-8 w-8 text-zinc-400" />
          ) : (
            <UploadCloud className="h-8 w-8 text-zinc-500" />
          )}
          <div>
            <p className="text-sm font-medium text-zinc-200">
              {file ? file.name : 'Drag & drop a transaction CSV here'}
            </p>
            <p className="mt-1 text-xs text-zinc-500">
              {file
                ? `${(file.size / 1048576).toFixed(1)} MB`
                : 'or click to browse — CSV up to 512 MB, any column naming'}
            </p>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={(event) => pickFile(event.target.files?.[0])}
          />
        </div>

        <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div className="sm:w-56">
            <label className="label" htmlFor="model">
              Model
            </label>
            <select
              id="model"
              className="input"
              value={model}
              onChange={(event) => setModel(event.target.value)}
            >
              {MODEL_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div className="flex gap-2">
            {file && (
              <button
                type="button"
                className="btn-ghost"
                disabled={busy}
                onClick={() => pickFile(null)}
              >
                <X className="h-4 w-4" />
                Clear
              </button>
            )}
            <button
              type="button"
              className="btn-primary"
              disabled={busy || !file}
              onClick={runAnalyze}
            >
              {busy ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <ShieldCheck className="h-4 w-4" />
              )}
              {busy ? (progress < 100 ? `Uploading ${progress}%` : 'Analyzing…') : 'Analyze'}
            </button>
          </div>
        </div>
      </section>

      {error && (
        <div className="rounded-2xl border border-red-900/60 bg-red-950/30 p-5 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* --------------------------------------------------------- results */}
      {result && (
        <>
          <section className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <Stat label="Transactions" value={formatNumber(result.total_rows)} />
            <Stat label="Fraud" value={formatNumber(result.fraud_rows)} tone="fraud" />
            <Stat label="Legitimate" value={formatNumber(result.legitimate_rows)} tone="legit" />
            <Stat label="Model" value={result.model_display_name} />
          </section>

          {result.evaluation && (
            <section className="card p-5">
              <h3 className="text-sm font-semibold text-zinc-200">
                Accuracy vs the file's own Class column
              </h3>
              <p className="mt-1 text-xs text-zinc-500">
                Ground-truth labels from the uploaded file, used only to score the model.
              </p>
              <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Metric label="Precision" value={formatPercent(result.evaluation.precision)} />
                <Metric label="Recall" value={formatPercent(result.evaluation.recall)} />
                <Metric label="F1" value={formatPercent(result.evaluation.f1)} />
                <Metric
                  label="Fraud missed"
                  value={formatNumber(result.evaluation.confusion_matrix.false_negative)}
                />
              </div>
            </section>
          )}

          {result.rejected_rows > 0 && (
            <section className="rounded-2xl border border-amber-900/50 bg-amber-950/20 p-5 text-sm text-amber-400">
              {formatNumber(result.rejected_rows)} row(s) could not be scored (missing or
              non-numeric feature values) and were skipped.
              {result.rejected.length > 0 && (
                <ul className="mt-2 list-disc space-y-0.5 pl-5 text-xs text-amber-500/80">
                  {result.rejected.slice(0, 5).map((row) => (
                    <li key={row.row}>
                      Transaction #{formatNumber(row.row)} — {row.reason}
                    </li>
                  ))}
                </ul>
              )}
            </section>
          )}

          <section className="card overflow-hidden">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-900 p-4">
              <div className="flex flex-wrap gap-2">
                {filters.map(([key, label]) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => {
                      setFilter(key)
                      setPage(1)
                    }}
                    className={`btn px-3 py-1.5 text-xs ${
                      filter === key
                        ? 'bg-white text-black'
                        : 'border border-zinc-800 text-zinc-400 hover:text-white'
                    }`}
                  >
                    {label} ({formatNumber(counts[key])})
                  </button>
                ))}
              </div>
              <button
                type="button"
                className="btn-ghost text-xs"
                onClick={exportCsv}
                disabled={!rows.length}
              >
                Download results (CSV)
              </button>
            </div>

            <div className="scroll-thin overflow-x-auto">
              <table className="min-w-full">
                <thead className="border-b border-zinc-900">
                  <tr>
                    <th className="th">#</th>
                    <th className="th">Amount</th>
                    <th className="th">Verdict</th>
                    <th className="th">Risk score</th>
                    {hasActual && <th className="th">Actual</th>}
                  </tr>
                </thead>
                <tbody>
                  {pageRows.length === 0 && (
                    <tr>
                      <td className="td text-zinc-500" colSpan={hasActual ? 5 : 4}>
                        No transactions in this view.
                      </td>
                    </tr>
                  )}
                  {pageRows.map((row) => (
                    <tr
                      key={row.row}
                      className="border-b border-zinc-900/70 last:border-0 hover:bg-zinc-900/40"
                    >
                      <td className="td tabular-nums text-zinc-500">{formatNumber(row.row)}</td>
                      <td className="td tabular-nums">{formatCurrency(row.amount)}</td>
                      <td className="td">
                        <Verdict prediction={row.prediction} />
                      </td>
                      <td className="td tabular-nums">{formatScore(row.score)}</td>
                      {hasActual && (
                        <td className="td text-zinc-500">
                          {row.actual_class === 1
                            ? 'Fraud'
                            : row.actual_class === 0
                              ? 'Legitimate'
                              : '—'}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="flex items-center justify-between gap-3 border-t border-zinc-900 p-4 text-xs text-zinc-500">
              <span>
                Showing {filtered.length ? (safePage - 1) * PAGE_SIZE + 1 : 0}–
                {Math.min(safePage * PAGE_SIZE, filtered.length)} of {formatNumber(filtered.length)}
                {result.results_truncated && ` (capped at ${formatNumber(rows.length)} rows)`}
              </span>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  className="btn-ghost px-3 py-1.5 text-xs"
                  disabled={safePage <= 1}
                  onClick={() => setPage(safePage - 1)}
                >
                  Previous
                </button>
                <span>
                  Page {safePage} / {pageCount}
                </span>
                <button
                  type="button"
                  className="btn-ghost px-3 py-1.5 text-xs"
                  disabled={safePage >= pageCount}
                  onClick={() => setPage(safePage + 1)}
                >
                  Next
                </button>
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  )
}
