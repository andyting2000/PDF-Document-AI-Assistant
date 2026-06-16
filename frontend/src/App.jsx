import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Send, Upload, Cpu, User, Terminal, Zap, MessageSquare, Sliders, Trash2, FileText } from 'lucide-react';

const CURATED_TEXT_MODELS = [
  'openai/gpt-oss-120b',
  'Qwen/Qwen3-235B-A22B-Instruct-2507',
  'meta-llama/Llama-3.1-8B-Instruct'
];

const MODEL_LABELS = {
  'openai/gpt-oss-120b': 'GPT-OSS 120B',
  'Qwen/Qwen3-235B-A22B-Instruct-2507': 'Qwen-3 235B',
  'meta-llama/Llama-3.1-8B-Instruct': 'LLaMa-3.1 8B'
};

export default function App() {
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'System Online. PDF document AI assistant is ready. Please upload your PDF document(s) to begin analysis.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [modelOptions, setModelOptions] = useState(CURATED_TEXT_MODELS);
  const [selectedModel, setSelectedModel] = useState(CURATED_TEXT_MODELS[0]);
  const [temperature, setTemperature] = useState(0.5);
  const [maxTokens, setMaxTokens] = useState(2000);
  const [topP, setTopP] = useState(0.9);
  const [frequencyPenalty, setFrequencyPenalty] = useState(0.0);
  const [presencePenalty, setPresencePenalty] = useState(0.0);

  const messagesEndRef = useRef(null);

  const [summary, setSummary] = useState("");
  const [backendHistory, setBackendHistory] = useState([]);

  // Fetch PDF list from backend
  useEffect(() => {
    const fetchPdfList = async () => {
      try {
        const res = await axios.get('http://127.0.0.1:8000/list-pdfs');
        setUploadedFiles(res.data.pdf_files || []);
      } catch (error) {
        setUploadedFiles([]);
      }
    };
    fetchPdfList();
  }, []);

  // Delete individual PDF
  const handleDeletePdf = async (filename) => {
    const confirmDelete = window.confirm(`Delete PDF "${filename}" and its embeddings? This cannot be undone.`);
    if (!confirmDelete) return;
    try {
      await axios.delete(`http://127.0.0.1:8000/delete-pdf?filename=${encodeURIComponent(filename)}`);
      setUploadedFiles(prev => prev.filter(f => f !== filename));
      setMessages(prev => [...prev, { role: 'bot', text: `>> Deleted PDF: ${filename}` }]);
    } catch (error) {
      alert('Error deleting PDF. Is the backend running?');
    }
  };

  useEffect(() => {
    const loadModels = async () => {
      try {
        const res = await axios.get('http://127.0.0.1:8000/models');
        const models = res?.data?.text_models?.slice(0, 3)?.filter(Boolean) || CURATED_TEXT_MODELS;
        const defaultModel = res?.data?.default_text_model || models[0] || CURATED_TEXT_MODELS[0];
        setModelOptions(models);
        setSelectedModel(defaultModel);
      } catch (error) {
        setModelOptions(CURATED_TEXT_MODELS);
        setSelectedModel(CURATED_TEXT_MODELS[0]);
      }
    };

    loadModels();
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(scrollToBottom, [messages]);

  const clearChat = () => {
    setMessages([{ role: 'bot', text: 'System Online. Memory wiped. Ready for new query.' }]);
    setBackendHistory([]);
    setSummary("");
  };

  const clearDatabase = async () => {
    const confirmDelete = window.confirm("🚨 WARNING: Are you sure you want to delete ALL uploaded PDFs from the database? This cannot be undone.");
    if (!confirmDelete) return;

    try {
      await axios.delete('http://127.0.0.1:8000/clear');
      setUploadedFiles([]);
      setMessages([{ role: 'bot', text: '>> SYSTEM PURGE COMPLETE: All document vectors have been erased from the PostgreSQL database.' }]);
    } catch (error) {
      alert("Error clearing database. Is the backend running?");
    }
  };

  const handleUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    setUploading(true);
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      await axios.post('http://127.0.0.1:8000/upload', formData);
      const newFileNames = files.map(f => f.name);
      setUploadedFiles(prev => [...prev, ...newFileNames]);

      const fileCountText = files.length === 1 ? '1 file' : `${files.length} files`;
      setMessages(prev => [...prev, { role: 'bot', text: `>> Ingestion Complete: Processed ${fileCountText}.` }]);
    } catch (error) {
      const detail =
        error?.response?.data?.detail ||
        error?.message ||
        'Upload failed while processing the PDF.';
      setMessages(prev => [...prev, { role: 'bot', text: `>> Upload Error: ${detail}` }]);
    }
    setUploading(false);
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg = input;

    setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setInput('');
    setLoading(true);

    const updatedHistory = [...backendHistory, { role: 'user', content: userMsg }];

    let activeHistory = updatedHistory;
    let toSummarize = [];

    if (updatedHistory.length > 4) {
      toSummarize = updatedHistory.slice(0, updatedHistory.length - 4);
      activeHistory = updatedHistory.slice(-4);
    }

    try {
      const res = await axios.post('http://127.0.0.1:8000/chat', {
        question: userMsg,
        history: activeHistory,
        summarize_these: toSummarize,
        current_summary: summary,
        selected_model: selectedModel,
        temperature: parseFloat(temperature),
        max_tokens: parseInt(maxTokens),
        top_p: parseFloat(topP),
        frequency_penalty: parseFloat(frequencyPenalty),
        presence_penalty: parseFloat(presencePenalty)
      });

      const botAnswer = res.data.answer;

      setSummary(res.data.new_summary);
      setBackendHistory([...activeHistory, { role: 'assistant', content: botAnswer }]);
      setMessages(prev => [...prev, { role: 'bot', text: botAnswer }]);

    } catch (error) {
      setMessages(prev => [...prev, { role: 'bot', text: ">> Error: Could not retrieve response." }]);
    }
    setLoading(false);
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans">

      <div className="w-[420px] bg-slate-900 border-r border-slate-800 flex flex-col p-8 shadow-2xl z-10 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700">

        <div className="mb-8 flex justify-between items-start">
          <div>
            <div className="flex items-center gap-3 text-indigo-400 mb-2">
              <Cpu className="w-6 h-6" />
              <span className="font-mono text-sm tracking-widest uppercase text-indigo-300">Document Intelligence System</span>
            </div>
            <h1 className="text-2xl font-bold text-white tracking-tight whitespace-nowrap">
              PDF Document AI Assistant
            </h1>
          </div>
        </div>

        <div className="mb-6">
          <label className={`
            group flex flex-col items-center justify-center w-full p-6 
            border-2 border-dashed rounded-xl cursor-pointer transition-all duration-300
            ${uploading ? 'border-indigo-500 bg-indigo-500/10' : 'border-slate-700 hover:border-indigo-400 hover:bg-slate-800'}
          `}>
            {uploading ? <Zap className="w-8 h-8 text-indigo-400 animate-pulse" /> : <Upload className="w-8 h-8 text-slate-400 group-hover:text-indigo-300" />}
            <span className="mt-3 text-sm font-medium text-slate-300 group-hover:text-white transition-colors">
              {uploading ? "Ingesting Data..." : "Upload PDF Source(s)"}
            </span>
            <input type="file" className="hidden" accept=".pdf" multiple onChange={handleUpload} />
          </label>
        </div>

        {uploadedFiles.length > 0 && (
          <div className="mb-8 bg-slate-950/50 rounded-xl border border-slate-800 p-3">
            <div className="text-xs text-slate-500 font-bold mb-2 uppercase tracking-wider pl-1">
              Uploaded PDFs ({uploadedFiles.length})
            </div>
            <div className="space-y-1 mb-3">
              {uploadedFiles.map((file, idx) => (
                <div key={idx} className="flex items-center gap-2 text-xs text-indigo-300 font-mono py-1.5 px-2 rounded hover:bg-slate-800 break-all">
                  <FileText size={12} className="shrink-0" />
                  <span>{file}</span>
                  <button
                    onClick={() => handleDeletePdf(file)}
                    className="ml-auto px-2 py-1 text-red-400 hover:text-red-600 bg-transparent border-none cursor-pointer rounded transition-all"
                    title="Delete PDF"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              ))}
            </div>

            <button
              onClick={clearDatabase}
              className="w-full py-2 flex items-center justify-center gap-2 bg-red-900/20 hover:bg-red-600/30 border border-red-900/50 hover:border-red-500 text-red-400 rounded-lg transition-all font-bold text-[10px] tracking-widest uppercase"
            >
              <Trash2 size={12} /> Purge Database
            </button>
          </div>
        )}

        <div className="mb-6 bg-slate-800/50 p-5 rounded-xl border border-slate-700 space-y-5">
          <div className="flex items-center gap-2 text-indigo-300 font-bold text-sm uppercase tracking-wider border-b border-slate-700 pb-2">
            <Sliders size={16} /> Model Parameters
          </div>

          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>AI Model</span>
              <span className="text-indigo-300 font-mono text-[10px] truncate max-w-[200px] text-right">{MODEL_LABELS[selectedModel] || selectedModel}</span>
            </div>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 text-slate-200 text-sm px-3 py-2 rounded-lg focus:outline-none focus:ring-1 focus:ring-indigo-500/50"
            >
              {modelOptions.map((model) => (
                <option key={model} value={model}>{MODEL_LABELS[model] || model}</option>
              ))}
            </select>
          </div>

          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>Max Tokens (Response Length)</span>
              <span className="text-indigo-300 font-mono">{maxTokens}</span>
            </div>
            <input type="range" min="100" max="4000" step="100" value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500" />
          </div>

          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>Temperature (Creativity)</span>
              <span className="text-indigo-300 font-mono">{temperature}</span>
            </div>
            <input type="range" min="0" max="1" step="0.1" value={temperature} onChange={(e) => setTemperature(e.target.value)} className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500" />
          </div>

          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>Top-P (Vocab Pool)</span>
              <span className="text-indigo-300 font-mono">{topP}</span>
            </div>
            <input type="range" min="0" max="1" step="0.05" value={topP} onChange={(e) => setTopP(e.target.value)} className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500" />
          </div>

          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>Frequency Penalty (Repetition)</span>
              <span className="text-indigo-300 font-mono">{frequencyPenalty}</span>
            </div>
            <input type="range" min="0" max="2" step="0.1" value={frequencyPenalty} onChange={(e) => setFrequencyPenalty(e.target.value)} className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500" />
          </div>

          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>Presence Penalty (Novelty)</span>
              <span className="text-indigo-300 font-mono">{presencePenalty}</span>
            </div>
            <input type="range" min="0" max="2" step="0.1" value={presencePenalty} onChange={(e) => setPresencePenalty(e.target.value)} className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500" />
          </div>
        </div>

        <button onClick={clearChat} className="w-full mb-6 py-3 flex items-center justify-center gap-2 bg-slate-900 hover:bg-red-500/10 border border-slate-700 hover:border-red-500/50 text-slate-400 hover:text-red-400 rounded-xl transition-all font-semibold text-sm">
          <Trash2 size={16} /> WIPE CHAT SCREEN
        </button>

        <div className="mt-auto pt-4 border-t border-slate-700">
          <div className="flex items-center gap-2 text-slate-500 text-xs font-medium">
            <User size={12} />
            <span>Developed by Andy Ting</span>
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col bg-slate-950">
        <div className="flex-1 overflow-y-auto p-10 space-y-8 scrollbar-thin scrollbar-thumb-slate-800">
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
              <div className={`
                max-w-[80%] p-6 rounded-2xl text-base leading-7 shadow-lg
                ${msg.role === 'user'
                  ? 'bg-indigo-600 text-white rounded-tr-none'
                  : 'bg-slate-900 border border-slate-700 text-slate-200 rounded-tl-none'}
              `}>
                {msg.role === 'bot' && (
                  <div className="flex items-center gap-2 mb-3 text-indigo-400 text-xs font-bold tracking-widest uppercase">
                    <Terminal size={14} /> AI Assistant Response
                  </div>
                )}
                <div className="whitespace-pre-wrap font-medium">{msg.text}</div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
          {loading && (
            <div className="flex items-center gap-3 text-indigo-400 text-base ml-10 font-mono animate-pulse">
              <MessageSquare className="w-5 h-5" /> <span>Analyzing data...</span>
            </div>
          )}
        </div>

        <div className="p-8 bg-slate-950 border-t border-slate-900">
          <div className="max-w-5xl mx-auto relative flex items-center gap-4">
            <input
              type="text"
              className="w-full bg-slate-900 border border-slate-800 text-white text-lg p-5 pl-8 rounded-2xl focus:outline-none focus:ring-1 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all shadow-xl placeholder:text-slate-600"
              placeholder="Type your query here..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            />
            <button
              onClick={sendMessage}
              disabled={loading}
              className="absolute right-3 p-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl transition-all shadow-lg disabled:opacity-50"
            >
              <Send size={24} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
