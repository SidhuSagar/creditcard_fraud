import { ArrowRight, FileSpreadsheet, Gauge, ScanSearch, ShieldCheck } from 'lucide-react'

const STEPS = [
  {
    icon: FileSpreadsheet,
    title: '1. Upload the dataset',
    body: 'Drop in creditcard.csv, or any CSV with the Time, V1–V28 and Amount factors under any column names.',
  },
  {
    icon: Gauge,
    title: '2. Every row is scored',
    body: 'A trained model assigns each transaction a fraud probability or anomaly score.',
  },
  {
    icon: ScanSearch,
    title: '3. Review fraud vs legitimate',
    body: 'Fraud is listed first, with filters, pagination and a CSV download of the results.',
  },
]

const FACTS = [
  ['284,807', 'real transactions used for training'],
  ['3', 'trained models'],
  ['492', 'frauds in the data'],
  ['0.17%', 'fraud rate'],
]

export default function Home({ health, onStart }) {
  return (
    <div className="space-y-12">
      <section className="text-center">
        <span className="inline-flex items-center gap-2 rounded-full border border-zinc-800 px-3 py-1 text-xs text-zinc-400">
          <ShieldCheck className="h-3.5 w-3.5" />
          Machine-learning fraud detection
        </span>
        <h2 className="mx-auto mt-6 max-w-2xl text-4xl font-semibold tracking-tight sm:text-5xl">
          Find the fraudulent transactions in your data
        </h2>
        <p className="mx-auto mt-4 max-w-xl text-sm text-zinc-400 sm:text-base">
          Upload a transaction file and FraudShield AI classifies every row as fraud or legitimate
          — in seconds, entirely on your machine.
        </p>
        <div className="mt-8 flex justify-center">
          <button type="button" className="btn-primary px-6 py-3 text-base" onClick={onStart}>
            Start analysis
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>
        {health && !health.models_trained && (
          <p className="mx-auto mt-4 max-w-xl text-xs text-amber-400">
            Heads up: no trained model was found. Run{' '}
            <code className="rounded bg-zinc-950 px-1.5 py-0.5 text-zinc-200">
              python training/train_models.py
            </code>{' '}
            in the backend folder first.
          </p>
        )}
      </section>

      <section className="grid gap-4 sm:grid-cols-3">
        {STEPS.map(({ icon: Icon, title, body }) => (
          <div key={title} className="card p-5">
            <Icon className="h-5 w-5 text-zinc-400" />
            <h3 className="mt-3 text-sm font-semibold text-zinc-100">{title}</h3>
            <p className="mt-1 text-sm text-zinc-400">{body}</p>
          </div>
        ))}
      </section>

      <section className="grid grid-cols-2 gap-px overflow-hidden rounded-2xl border border-zinc-900 bg-zinc-900 sm:grid-cols-4">
        {FACTS.map(([value, label]) => (
          <div key={label} className="bg-zinc-950 p-5 text-center">
            <p className="text-2xl font-semibold tabular-nums text-zinc-100">{value}</p>
            <p className="mt-1 text-xs text-zinc-500">{label}</p>
          </div>
        ))}
      </section>
    </div>
  )
}
