import { useState } from 'react'

/**
 * Smart Context Manager - Frontend
 * * This application allows users to upload PDF documents and perform semantic 
 * searches against the content using a RAG (Retrieval-Augmented Generation) pipeline.
 */
function App() {
  // --- State Management ---
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([]) // Stores the context chunks retrieved from the DB
  const [answer, setAnswer] = useState('')   // Stores the final AI-generated response
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [lastUpload, setLastUpload] = useState(null)

  // Configuration from Environment Variables
  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
  const SECURITY_TOKEN = import.meta.env.VITE_APP_SECURITY_TOKEN;

  /**
   * Handles PDF file upload to the backend.
   * Sends the file as multipart/form-data and updates the UI with document metadata.
   */
  const handleUpload = async (event) => {
    const file = event.target.files[0]
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    setUploading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/upload-pdf`, {
        method: 'POST',
        headers: { 'X-Custom-Token': SECURITY_TOKEN },
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Upload failed')
      }

      const data = await response.json()
      // Store metadata (page count, chunks, etc.) to show the user the file is indexed
      setLastUpload(data.metadata)
      
    } catch (error) {
      console.error("Upload error:", error)
      alert(error.message || "Failed to upload PDF.")
    } finally {
      setUploading(false)
    }
  }

  /**
   * Performs a semantic search.
   * NOTE: We increased the 'limit' to 10. 
   * This ensures that if Page 1 has many repetitive headers, the search logic 
   * will still "reach" the unique content on Page 2 (like your Languages/Idiomas section).
   */
  const handleSearch = async () => {
    if (!query) return
    setLoading(true)
    setAnswer('') // Clear previous answer while loading
    
    try {
      // FIX: Increased limit=10 to retrieve a wider context from the vector database
      const response = await fetch(`${API_BASE_URL}/search?query=${encodeURIComponent(query)}&limit=10`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'X-Custom-Token': SECURITY_TOKEN 
        }
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Search failed')
      }

      const data = await response.json()
      setAnswer(data.answer)
      setResults(data.results || [])
    } catch (error) {
      console.error("Search error:", error)
      alert("Error connecting to the Smart Context API.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-900 text-white p-8 flex flex-col items-center font-sans">
      <h1 className="text-4xl font-bold text-blue-400 mb-8 tracking-tight">Smart Context Manager</h1>

      <div className="w-full max-w-2xl space-y-6">
        
        {/* 1. KNOWLEDGE BASE: Document Upload Section */}
        <div className="bg-slate-800 p-6 rounded-2xl shadow-xl border border-slate-700">
          <h2 className="text-xs font-bold text-slate-500 uppercase mb-4 tracking-[0.2em]">1. Knowledge Base</h2>
          <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-slate-700 border-dashed rounded-xl cursor-pointer hover:bg-slate-700/50 transition-all active:scale-[0.98]">
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <span className="text-3xl mb-2">{uploading ? "🌀" : "📄"}</span>
              <p className="text-sm text-slate-400 font-medium">
                {uploading ? "Analyzing document structure..." : "Drop a PDF here or click to browse"}
              </p>
            </div>
            <input type="file" className="hidden" accept=".pdf" onChange={handleUpload} disabled={uploading} />
          </label>

          {/* Metadata Display: Shows details about the indexed PDF */}
          {lastUpload && (
            <div className="mt-4 p-4 bg-blue-500/5 border border-blue-500/20 rounded-xl flex items-center gap-4 animate-in fade-in slide-in-from-top-2">
              <div className="bg-blue-500/20 p-3 rounded-lg text-blue-400 text-xl">✅</div>
              <div className="flex-1 min-w-0">
                <p className="text-blue-100 font-semibold text-sm truncate">{lastUpload.original_name}</p>
                <p className="text-slate-500 text-xs mt-0.5">
                  {lastUpload.page_count} pages • {lastUpload.file_size} • {lastUpload.chunks_processed} chunks
                </p>
              </div>
            </div>
          )}
        </div>

        {/* 2. SMART SEARCH: Query and Results Section */}
        <div className="bg-slate-800 p-6 rounded-2xl shadow-xl border border-slate-700">
          <h2 className="text-xs font-bold text-slate-500 uppercase mb-4 tracking-[0.2em]">2. Smart Search</h2>
          <div className="flex gap-2 mb-6">
            <input 
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Ask a question about your indexed docs..."
              className="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all"
            />
            <button 
              onClick={handleSearch}
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-500 active:bg-blue-700 px-6 py-3 rounded-xl font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-900/20"
            >
              {loading ? '...' : 'Search'}
            </button>
          </div>

          {/* AI Perspective: The synthesized response from Gemini */}
          {answer && (
            <div className="mb-8 p-6 bg-slate-900/50 border border-slate-700 rounded-2xl animate-in fade-in zoom-in-95">
               <h3 className="text-xs font-bold text-blue-500 uppercase mb-3 tracking-widest flex items-center gap-2">
                <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></span>
                AI Perspective
              </h3>
              <p className="text-slate-200 text-lg leading-relaxed antialiased font-medium">{answer}</p>
            </div>
          )}

          {/* Retrieval Visibility: Shows the specific document chunks used for the answer */}
          <div className="space-y-4">
            {results.length > 0 && (
              <div className="flex items-center justify-between mb-2">
                <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Retrieved Context</p>
                <p className="text-[10px] text-slate-600 font-mono">Backend Threshold: 12.00%</p>
              </div>
            )}
            
            {results.map((res, index) => {
              const scorePercent = (res.similarity * 100).toFixed(2);
              const isHighConfidence = res.similarity > 0.25;
              
              return (
                <div key={index} className="group p-4 bg-slate-900/40 border border-slate-800 rounded-xl hover:border-slate-600 transition-colors">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${isHighConfidence ? 'bg-emerald-500/10 text-emerald-400' : 'bg-blue-500/10 text-blue-400'}`}>
                        Similarity: {scorePercent}%
                      </span>
                    </div>
                    {/* Visual confidence bar for quick UX feedback */}
                    <div className="w-24 h-1 bg-slate-800 rounded-full overflow-hidden">
                      <div 
                        className={`h-full transition-all duration-1000 ${isHighConfidence ? 'bg-emerald-500' : 'bg-blue-500'}`}
                        style={{ width: `${Math.min(scorePercent, 100)}%` }}
                      ></div>
                    </div>
                  </div>
                  <p className="text-slate-400 text-sm leading-relaxed italic group-hover:text-slate-300 transition-colors">
                    "{res.content.length > 250 ? res.content.substring(0, 250) + '...' : res.content}"
                  </p>
                </div>
              );
            })}
          </div>
        </div>

      </div>
    </div>
  )
}

export default App