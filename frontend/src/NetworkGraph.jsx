import { useEffect, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import axios from 'axios'

const API = 'http://127.0.0.1:8000'

export default function NetworkGraph() {
  const graphRef = useRef()
  const [graph, setGraph] = useState({ nodes: [], links: [] })
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios.get(`${API}/api/graph`)
      .then(({ data }) => {
        setGraph({
          nodes: data.nodes,
          links: data.edges.map(edge => ({
            source: edge.source,
            target: edge.target
          }))
        })
      })
      .catch(error => console.error('Graph loading failed:', error))
      .finally(() => setLoading(false))
  }, [])

  const findAccount = () => {
    const account = graph.nodes.find(
      node => String(node.account_id) === search.trim()
    )

    if (account) {
      setSelected(account)
      graphRef.current?.centerAt(account.x, account.y, 700)
      graphRef.current?.zoom(3, 700)
    } else {
      alert('Account not found')
    }
  }

  if (loading) return <div>Loading network...</div>

  return (
    <div className="w-full space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <input
          value={search}
          onChange={e => setSearch(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && findAccount()}
          placeholder="Search Ethereum account ID..."
          className="rounded-lg border border-slate-600 bg-slate-900 px-4 py-2 text-white"
        />

        <button
          onClick={findAccount}
          className="rounded-lg bg-blue-600 px-4 py-2 text-white"
        >
          Search
        </button>

        <span className="text-green-400">
          Legitimate: {graph.nodes.filter(n => n.label === 0).length}
        </span>

        <span className="text-red-400">
          Phishing: {graph.nodes.filter(n => n.label === 1).length}
        </span>
      </div>

      <div className="relative overflow-hidden rounded-xl border border-slate-700 bg-slate-950">
        <ForceGraph2D
          ref={graphRef}
          graphData={graph}
          width={900}
          height={550}
          backgroundColor="#020617"
          nodeId="id"
          nodeLabel={node => `Account: ${node.account_id}\nType: ${node.type}`}
          nodeColor={node => node.label === 1 ? '#ef4444' : '#22c55e'}
          nodeRelSize={4}
          linkColor={() => 'rgba(148,163,184,0.18)'}
          linkWidth={0.6}
          onNodeClick={node => setSelected(node)}
          cooldownTicks={100}
        />

        <div className="absolute bottom-3 left-3 flex gap-4 rounded-lg bg-slate-900/90 p-3 text-sm text-white">
          <span>🟢 Legitimate</span>
          <span>🔴 Phishing</span>
        </div>
      </div>

      {selected && (
        <div className="rounded-xl border border-slate-700 bg-slate-900 p-5 text-white">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-bold">Account Details</h3>
            <button onClick={() => setSelected(null)} className="text-slate-400">
              Close
            </button>
          </div>
          <p className="mt-3">Account ID: {selected.account_id}</p>
          <p>Graph Node: {selected.id}</p>
          <p>Label: {selected.label}</p>
          <p className={selected.label === 1 ? 'mt-2 font-bold text-red-400' : 'mt-2 font-bold text-green-400'}>
            Classification: {selected.type}
          </p>
        </div>
      )}
    </div>
  )
}
