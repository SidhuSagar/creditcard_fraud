import { useEffect, useState } from 'react'
import { ShieldCheck } from 'lucide-react'
import Home from './pages/Home.jsx'
import Analyze from './pages/Analyze.jsx'
import { api } from './services/api.js'

function HealthPill({ health }) {
  const ready = Boolean(health?.models_trained)
  const label = !health ? 'Connecting…' : ready ? 'Model ready' : 'Model not trained'
  const dot = !health ? 'bg-zinc-600' : ready ? 'bg-emerald-500' : 'bg-amber-500'
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-zinc-800 px-3 py-1 text-xs text-zinc-400">
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      {label}
    </span>
  )
}

export default function App() {
  const [view, setView] = useState('home')
  const [health, setHealth] = useState(null)

  useEffect(() => {
    let active = true
    api
      .health()
      .then((data) => active && setHealth(data))
      .catch(() => active && setHealth(null))
    return () => {
      active = false
    }
  }, [])

  const goHome = () => setView('home')

  return (
    <div className="min-h-screen bg-black text-zinc-100">
      <header className="border-b border-zinc-900">
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-6 py-5">
          <button
            type="button"
            onClick={goHome}
            className="flex items-center gap-3 rounded-lg text-left"
            aria-label="Go to home"
          >
            <span className="flex h-9 w-9 items-center justify-center rounded-xl border border-zinc-800 bg-zinc-950">
              <ShieldCheck className="h-5 w-5 text-white" />
            </span>
            <span>
              <span className="block text-base font-semibold leading-tight tracking-tight">
                FraudShield AI
              </span>
              <span className="block text-xs text-zinc-500">Credit-card fraud detection</span>
            </span>
          </button>
          <HealthPill health={health} />
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-10">
        {view === 'home' ? (
          <Home health={health} onStart={() => setView('analyze')} />
        ) : (
          <Analyze health={health} onHome={goHome} />
        )}
      </main>

      <footer className="border-t border-zinc-900 px-6 py-6 text-center text-xs text-zinc-600">
        Educational machine-learning demo. Not connected to any bank or payment network. No real
        transaction is ever blocked.
      </footer>
    </div>
  )
}
