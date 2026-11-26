
import { useEffect, useRef, useState } from 'react'
import axios from 'axios'
import {
  Shield, Activity, AlertTriangle, Network,
  Database, Brain, RefreshCw, CheckCircle,
  TrendingUp, Zap, Play, Square, Radio
} from 'lucide-react'

const API = 'http://127.0.0.1:8000'
const WS = 'ws://127.0.0.1:8000/ws/process'

function App() {
  const [health, setHealth] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [processed, setProcessed] = useState(0)
  const [total, setTotal] = useState(0)
  const [rfAlerts, setRfAlerts] = useState(0)
  const [xgbAlerts, setXgbAlerts] = useState(0)
  const [alerts, setAlerts] = useState([])
  const [streamStatus, setStreamStatus] = useState('Ready')

  const socketRef = useRef(null)

  const loadData = async () => {
    setLoading(true)
    setError('')

    try {
      const [h, m] = await Promise.all([
        axios.get(`${API}/api/health`),
        axios.get(`${API}/api/metrics`)
      ])

      setHealth(h.data)
      setMetrics(m.data)
    } catch (err) {
      setError('Could not connect to the backend.')
      console.error(err)
    }

    setLoading(false)
  }

  useEffect(() => {
    loadData()

    return () => {
      if (socketRef.current) {
        socketRef.current.close()
      }
    }
  }, [])

  const startAnalysis = () => {
    if (running) return

    setError('')
    setRunning(true)
    setProgress(0)
    setProcessed(0)
    setTotal(0)
    setRfAlerts(0)
    setXgbAlerts(0)
    setAlerts([])
    setStreamStatus('Connecting to processing engine...')

    const ws = new WebSocket(WS)
    socketRef.current = ws

    ws.onopen = () => {
      setStreamStatus('Connected. Loading dataset...')
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        if (data.type === 'status') {
          setStreamStatus(data.message)
        }

        if (data.type === 'start') {
          setTotal(data.total)
          setStreamStatus('Analysis running')
        }

        if (data.type === 'progress') {
          setProcessed(data.processed)
          setTotal(data.total)
          setProgress(data.percentage)
          setRfAlerts(data.rf_suspicious)
          setXgbAlerts(data.xgb_suspicious)

          setAlerts(previous => {
            const incoming = data.latest_alerts || []
            const merged = [...incoming, ...previous]
            const unique = Array.from(
              new Map(merged.map(a => [a.address_id, a])).values()
            )
            return unique.slice(0, 30)
          })

          setStreamStatus('Processing historical Ethereum accounts...')
        }

        if (data.type === 'complete') {
          setProgress(100)
          setProcessed(data.processed ?? total)
          setStreamStatus('Analysis completed successfully')

          if (data.alerts) {
            setAlerts(data.alerts.slice(0, 30))
          }

          setRunning(false)
          ws.close()
        }

        if (data.type === 'error') {
          setError(data.message || 'Processing failed.')
          setStreamStatus('Processing error')
          setRunning(false)
          ws.close()
        }
      } catch (err) {
        console.error('WebSocket message error:', err)
      }
    }

    ws.onerror = () => {
      setError('WebSocket connection failed. Check that the backend is running.')
      setStreamStatus('Connection failed')
      setRunning(false)
    }

    ws.onclose = () => {
      setRunning(false)
      socketRef.current = null
    }
  }

  const stopAnalysis = () => {
    if (socketRef.current) {
      socketRef.current.close()
      socketRef.current = null
    }

    setRunning(false)
    setStreamStatus('Processing stopped by user')
  }

  const cards = [
    {
      title: 'Random Forest',
      value: metrics?.baseline?.['Random Forest']?.roc_auc,
      icon: Brain
    },
    {
      title: 'XGBoost',
      value: metrics?.baseline?.['XGBoost']?.roc_auc,
      icon: Activity
    },
    {
      title: 'GCN Model',
      value: metrics?.gcn?.roc_auc,
      icon: Network
    },
    {
      title: 'GAT Model',
      value: metrics?.gat?.roc_auc,
      icon: Shield
    }
  ]

  const robustness = metrics?.robustness || {}

  const statusItems = [
    ['FastAPI Backend', health?.api === 'online'],
    ['Random Forest', !!health?.random_forest],
    ['XGBoost', !!health?.xgboost],
    ['Model Metrics', !!metrics]
  ]

  return (
    <main className="min-h-screen bg-[#080d18] text-slate-200">

      <header className="border-b border-slate-800 bg-[#0d1423] px-8 py-5">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-cyan-500/10 p-3 text-cyan-400">
              <Shield size={27} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-wide text-white">
                ETHEREUM SENTINEL AI
              </h1>
              <p className="text-xs text-slate-500">
                Blockchain Threat Intelligence Platform
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-4 py-2 text-sm text-emerald-400">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            {health ? 'SYSTEM ONLINE' : 'CONNECTING'}
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl space-y-8 px-8 py-8">

        <section className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-sm text-cyan-400">
              SECURITY OVERVIEW / DASHBOARD
            </p>
            <h2 className="mt-2 text-3xl font-bold text-white">
              Threat Intelligence Center
            </h2>
            <p className="mt-2 text-slate-400">
              Machine learning analysis of Ethereum account behaviour.
            </p>
          </div>

          <button
            onClick={loadData}
            className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-4 py-3 text-sm hover:bg-slate-700"
          >
            <RefreshCw size={16} />
            Refresh data
          </button>
        </section>

        {error && (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-red-400">
            {error}
          </div>
        )}

        {/* HISTORICAL DATASET PROCESSING */}

        <section className="rounded-2xl border border-cyan-500/30 bg-[#101a2b] p-6">

          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 text-cyan-400">
                <Radio size={19} />
                <span className="text-xs font-semibold tracking-widest">
                  HISTORICAL DATASET ANALYSIS
                </span>
              </div>

              <h3 className="mt-3 text-2xl font-bold text-white">
                Ethereum Threat Detection Engine
              </h3>

              <p className="mt-2 text-sm text-slate-400">
                Batch inference across historical Ethereum account data.
              </p>
            </div>

            <div className="flex gap-3">
              <button
                onClick={startAnalysis}
                disabled={running}
                className="flex items-center gap-2 rounded-lg bg-cyan-500 px-5 py-3 font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Play size={17} />
                Start Analysis
              </button>

              <button
                onClick={stopAnalysis}
                disabled={!running}
                className="flex items-center gap-2 rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-red-400 hover:bg-red-500/20 disabled:opacity-30"
              >
                <Square size={15} />
                Stop
              </button>
            </div>
          </div>

          <div className="mt-6 flex items-center gap-2 text-sm text-slate-400">
            <span className={`h-2 w-2 rounded-full ${running ? 'animate-pulse bg-cyan-400' : progress === 100 ? 'bg-emerald-400' : 'bg-slate-500'}`} />
            {streamStatus}
          </div>

          <div className="mt-5">
            <div className="mb-2 flex justify-between text-sm">
              <span className="text-slate-400">Processing progress</span>
              <span className="font-semibold text-cyan-400">
                {progress.toFixed(2)}%
              </span>
            </div>

            <div className="h-3 overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full rounded-full bg-cyan-400 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">

            {[
              ['Accounts Processed', processed.toLocaleString()],
              ['Total Accounts', total.toLocaleString()],
              ['RF Suspicious', rfAlerts],
              ['XGBoost Suspicious', xgbAlerts]
            ].map(([label, value]) => (
              <div key={label} className="rounded-xl bg-slate-800/60 p-4">
                <p className="text-xs text-slate-400">{label}</p>
                <p className="mt-2 text-2xl font-bold text-white">
                  {value}
                </p>
              </div>
            ))}

          </div>

          <p className="mt-5 text-xs text-slate-500">
            This feature replays historical dataset processing. It is not a
            live Ethereum blockchain connection.
          </p>
        </section>

        {/* MODEL METRICS */}

        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {cards.map(({ title, value, icon: Icon }) => (
            <div
              key={title}
              className="rounded-2xl border border-slate-800 bg-[#101a2b] p-5"
            >
              <div className="flex items-center justify-between">
                <p className="text-sm text-slate-400">{title}</p>
                <Icon className="text-cyan-400" size={21} />
              </div>

              <p className="mt-5 text-3xl font-bold text-white">
                {typeof value === 'number'
                  ? `${(value * 100).toFixed(2)}%`
                  : '—'}
              </p>

              <p className="mt-2 text-xs text-slate-500">
                ROC-AUC performance
              </p>
            </div>
          ))}
        </section>

        {/* MODEL PERFORMANCE AND STATUS */}

        <section className="grid gap-5 lg:grid-cols-3">

          <div className="rounded-2xl border border-slate-800 bg-[#101a2b] p-6 lg:col-span-2">
            <div className="flex items-center gap-3">
              <Activity className="text-cyan-400" />
              <h3 className="font-semibold text-white">
                Model Performance
              </h3>
            </div>

            <p className="mt-3 text-sm text-slate-400">
              Comparative evaluation of machine learning and graph neural network models.
            </p>

            <div className="mt-6 space-y-5">
              {cards.map(({ title, value }) => {
                const score = typeof value === 'number' ? value : 0

                return (
                  <div key={title}>
                    <div className="mb-2 flex justify-between text-sm">
                      <span>{title}</span>
                      <span className="text-cyan-400">
                        {typeof value === 'number'
                          ? `${(score * 100).toFixed(2)}%`
                          : '—'}
                      </span>
                    </div>

                    <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                      <div
                        className="h-full rounded-full bg-cyan-500 transition-all"
                        style={{ width: `${score * 100}%` }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-[#101a2b] p-6">
            <div className="flex items-center gap-3">
              <Database className="text-cyan-400" />
              <h3 className="font-semibold text-white">
                System Status
              </h3>
            </div>

            <div className="mt-6 space-y-5">
              {statusItems.map(([name, online]) => (
                <div key={name} className="flex items-center justify-between">
                  <span className="text-sm text-slate-300">{name}</span>
                  {online
                    ? <CheckCircle size={19} className="text-emerald-400" />
                    : <AlertTriangle size={19} className="text-amber-400" />}
                </div>
              ))}
            </div>
          </div>

        </section>

        {/* LIVE ALERT FEED */}

        <section className="rounded-2xl border border-slate-800 bg-[#101a2b] p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <AlertTriangle className="text-amber-400" />
              <h3 className="font-semibold text-white">
                Suspicious Account Alerts
              </h3>
            </div>
            <span className="rounded-full bg-amber-500/10 px-3 py-1 text-xs text-amber-400">
              {alerts.length} displayed
            </span>
          </div>

          <div className="mt-5 overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-slate-800 text-xs text-slate-500">
                <tr>
                  <th className="p-3">Account ID</th>
                  <th className="p-3">Random Forest</th>
                  <th className="p-3">XGBoost</th>
                  <th className="p-3">Detection</th>
                </tr>
              </thead>

              <tbody>
                {alerts.map((alert) => (
                  <tr
                    key={alert.address_id}
                    className="border-b border-slate-800/60"
                  >
                    <td className="p-3 font-mono text-cyan-400">
                      {alert.address_id}
                    </td>

                    <td className="p-3">
                      {(alert.rf_probability * 100).toFixed(2)}%
                    </td>

                    <td className="p-3">
                      {(alert.xgb_probability * 100).toFixed(2)}%
                    </td>

                    <td className="p-3">
                      <span className="rounded-full bg-red-500/10 px-3 py-1 text-xs text-red-400">
                        Suspicious
                      </span>
                    </td>
                  </tr>
                ))}

                {alerts.length === 0 && (
                  <tr>
                    <td colSpan="4" className="p-8 text-center text-slate-500">
                      Start analysis to receive suspicious-account alerts.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        {/* ROBUSTNESS */}

        <section className="rounded-2xl border border-slate-800 bg-[#101a2b] p-6">
          <div className="flex items-center gap-3">
            <TrendingUp className="text-cyan-400" />
            <h3 className="font-semibold text-white">
              Graph Robustness Analysis
            </h3>
          </div>

          <p className="mt-2 text-sm text-slate-400">
            GCN performance under simulated transaction-edge removal.
          </p>

          <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {Object.entries(robustness).map(([level, result]) => (
              <div key={level} className="rounded-xl bg-slate-800/60 p-4">
                <p className="text-sm text-slate-400">
                  {level} edges removed
                </p>
                <p className="mt-3 text-2xl font-bold text-white">
                  {(result.f1 * 100).toFixed(2)}%
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  F1 Score
                </p>
                <p className="mt-3 text-xs text-cyan-400">
                  {result.remaining_edges.toLocaleString()} edges remaining
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* PROJECT INFORMATION */}

        <section className="rounded-2xl border border-cyan-500/20 bg-gradient-to-r from-cyan-500/10 to-blue-500/5 p-6">
          <div className="flex items-center gap-3">
            <Zap className="text-cyan-400" />
            <h3 className="font-semibold text-white">
              Ethereum Sentinel AI
            </h3>
          </div>

          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
            An experimental blockchain security platform combining
            conventional machine learning, graph neural networks,
            explainable AI and network robustness evaluation to study
            suspicious Ethereum account behaviour.
          </p>
        </section>

        {loading && (
          <p className="text-center text-sm text-cyan-400">
            Loading model information...
          </p>
        )}

        <footer className="border-t border-slate-800 pt-5 text-center text-xs text-slate-600">
          ETHEREUM SENTINEL AI · EXPERIMENTAL RESEARCH PLATFORM
        </footer>

      </div>
    </main>
  )
}

export default App
