import { useState } from 'react'

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [answer, setAnswer] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)

  // Function to handle PDF Upload
  const handleUpload = async (event) => {
    const file = event.target.files[0]
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    setUploading(true)
    try {
      const response = await fetch('http://127.0.0.1:8000/upload-pdf', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      alert(`Success: ${data.filename} indexed!`)
    } catch (error) {
      console.error("Upload error:", error)
      alert("Failed to upload PDF.")
    } finally {
      setUploading(false)
    }
  }

  // Function to handle Semantic Search
  const handleSearch = async () => {
    if (!query) return
    setLoading(true)
    setAnswer('')
    try {
      const response = await fetch(`http://127.0.0.1:8000/search?query=${encodeURIComponent(query)}&limit=3`)
      const data = await response.json()
      setAnswer(data.answer)
      setResults(data.results || [])
    } catch (error) {
      console.error("Search error:", error)
      alert("Check if Backend is running.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-900 text-white p-8 flex flex-col items-center">
      <h1 className="text-4xl font-bold text-blue-400 mb-8">Smart Context Manager</h1>

      <div className="w-full max-w-2xl space-y-6">
        
        {/* Step 1: Upload Section */}
        <div className="bg-slate-800 p-6 rounded-2xl shadow-xl border border-slate-700">
          <h2 className="text-sm font-bold text-slate-400 uppercase mb-4 tracking-widest">1. Knowledge Base</h2>
          <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-slate-700 border-dashed rounded-xl cursor-pointer hover:bg-slate-700/50 transition-all">
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <span className="text-2xl mb-2">📤</span>
              <p className="text-sm text-slate-400">
                {uploading ? "Indexing PDF..." : "Click to upload a PDF document"}
              </p>
            </div>
            <input type="file" className="hidden" accept=".pdf" onChange={handleUpload} disabled={uploading} />
          </label>
        </div>

        {/* Step 2: Search Section */}
        <div className="bg-slate-800 p-6 rounded-2xl shadow-xl border border-slate-700">
          <h2 className="text-sm font-bold text-slate-400 uppercase mb-4 tracking-widest">2. Smart Search</h2>
          <div className="flex gap-2 mb-6">
            <input 
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Ask anything about your docs..."
              className="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button 
              onClick={handleSearch}
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-500 px-6 py-3 rounded-xl font-bold transition-all"
            >
              {loading ? '...' : '🔍'}
            </button>
          </div>

          {/* AI Answer Result */}
          {answer && (
            <div className="mb-8 p-5 bg-blue-900/20 border-l-4 border-blue-500 rounded-r-xl">
              <p className="text-slate-100 text-lg leading-relaxed">{answer}</p>
            </div>
          )}

          {/* Source Chunks */}
          <div className="space-y-4">
            {results.map((res, index) => (
              <div key={index} className="p-4 bg-slate-900/30 border border-slate-700 rounded-lg opacity-80">
                <p className="text-xs text-blue-300/50 mb-1 font-mono">Similarity: {(res.similarity * 100).toFixed(2)}%</p>
                <p className="text-slate-400 text-sm italic">"{res.content.substring(0, 200)}..."</p>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  )
}

export default App