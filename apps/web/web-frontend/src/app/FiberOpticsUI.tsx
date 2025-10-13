import React, { useState } from "react";

const mockBatches = Array.from({ length: 12 }).map((_, i) => ({
  id: `batch-${i + 1}`,
  name: `Batch ${i + 1}`,
  totalSamples: 12 + i,
}));

const mockSamples = (batchId, page = 1, perPage = 8) => {
  const start = (page - 1) * perPage;
  return Array.from({ length: 48 }).map((_, idx) => ({
    id: `${batchId}-S-${idx + 1}`,
    classification: ["A", "B", "C"][idx % 3],
    luster: Math.round(Math.random() * 10),
    roughness: Math.round(Math.random() * 10),
    tensile: (Math.random() * 50 + 10).toFixed(2),
    img: null,
    date_created: new Date(Date.now() - idx * 86400000).toISOString().slice(0, 10),
  })).slice(start, start + perPage);
};

export default function App() {
  const [page, setPage] = useState("login"); // 'login' | 'dashboard'
  const [showSignUp, setShowSignUp] = useState(false);
  const [user, setUser] = useState(null);

  const handleLogin = (credentials) => {
    // TODO: replace with real auth
    setUser({ name: "Insp. Castillo", role: "Inspector" });
    setPage("dashboard");
  };

  const handleSignUp = (data) => {
    // TODO: sign up logic
    setShowSignUp(false);
    // optionally log the user in or show success toast
  };

  return (
    <div className="min-h-screen bg-[#1f1f1f] text-gray-100 font-sans">
      {page === "login" && (
        <LoginPage
          onLogin={handleLogin}
          onOpenSignUp={() => setShowSignUp(true)}
        />
      )}

      {showSignUp && (
        <SignUpModal
          onClose={() => setShowSignUp(false)}
          onSignUp={handleSignUp}
          onSwitchToSignIn={() => setShowSignUp(false)}
        />
      )}

      {page === "dashboard" && user && (
        <Dashboard user={user} onSignOut={() => setPage("login") } />
      )}
    </div>
  );
}

function LoginPage({ onLogin, onOpenSignUp }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  return (
    <div className="grid grid-cols-2 min-h-screen">
      {/* Left: large image area */}
      <div
        className="bg-cover bg-center"
        style={{ backgroundImage: `url('/assets/abaca-left.jpg')` }}
      />

      {/* Right: login card */}
      <div className="flex items-center justify-center">
        <div className="w-3/5 max-w-md">
          <div className="flex flex-col items-center gap-4">
            <div className="w-24 h-24 rounded-lg bg-emerald-800/80 flex items-center justify-center">
              {/* app icon placeholder */}
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none">
                <path d="M12 2l2 6 4-1-3 4 2 6-5-3-5 3 2-6-3-4 4 1 2-6z" fill="#cbd5e1" />
              </svg>
            </div>

            <h1 className="text-2xl tracking-wider">Fiber Optics</h1>
            <p className="text-sm text-gray-300">Abaca Classification Tracking System</p>
            <h2 className="text-xl mt-2">LOG-IN</h2>

            <div className="w-full mt-3 space-y-3">
              <label className="text-xs text-gray-400">Username or Email</label>
              <input
                className="w-full rounded-lg px-4 py-2 text-black"
                placeholder="Example@email.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />

              <label className="text-xs text-gray-400">Password</label>
              <input
                className="w-full rounded-lg px-4 py-2 text-black"
                placeholder="At least 8 characters"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />

              <div className="flex justify-end text-sm">
                <a href="#" className="text-blue-400">Forgot Password?</a>
              </div>

              <button
                className="w-full rounded-lg py-2 bg-emerald-800/70"
                onClick={() => onLogin({ email, password })}
              >
                SIGN IN
              </button>

              <div className="text-center text-sm mt-4">
                Don't have an account? <button className="text-blue-400" onClick={onOpenSignUp}>Sign up</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SignUpModal({ onClose, onSignUp, onSwitchToSignIn }) {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");

  const handleSubmit = () => {
    if (password !== confirm) return alert("Passwords do not match");
    onSignUp({ username, email, password });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-[#252525] rounded-lg w-11/12 max-w-2xl p-6">
        <div className="flex justify-between items-center">
          <h3 className="text-lg">Sign up</h3>
          <button onClick={onClose} className="text-gray-400">✕</button>
        </div>

        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="col-span-1 md:col-span-1">
            <label className="text-xs text-gray-400">Username</label>
            <input value={username} onChange={e => setUsername(e.target.value)} className="w-full rounded px-3 py-2 text-black" />

            <label className="text-xs text-gray-400 mt-2 block">Email</label>
            <input value={email} onChange={e => setEmail(e.target.value)} className="w-full rounded px-3 py-2 text-black" />
          </div>

          <div className="col-span-1 md:col-span-1">
            <label className="text-xs text-gray-400">Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} className="w-full rounded px-3 py-2 text-black" />

            <label className="text-xs text-gray-400 mt-2 block">Confirm Password</label>
            <input type="password" value={confirm} onChange={e => setConfirm(e.target.value)} className="w-full rounded px-3 py-2 text-black" />
          </div>
        </div>

        <div className="mt-6 flex justify-between items-center">
          <div>
            Already have an account? <button onClick={onSwitchToSignIn} className="text-blue-400">Sign in</button>
          </div>

          <div className="flex gap-3">
            <button onClick={onClose} className="px-4 py-2 rounded border border-gray-600">Cancel</button>
            <button onClick={handleSubmit} className="px-4 py-2 rounded bg-emerald-800/70">Sign up</button>
          </div>
        </div>
      </div>
    </div>
  );
}

function Dashboard({ user, onSignOut }) {
  const [batches, setBatches] = useState(mockBatches);
  const [selectedBatchId, setSelectedBatchId] = useState(batches[0].id);
  const [pageNum, setPageNum] = useState(1);
  const [perPage, setPerPage] = useState(8);

  const samples = mockSamples(selectedBatchId, pageNum, perPage);

  return (
    <div className="min-h-screen p-6">
      <header className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded bg-emerald-800/70 flex items-center justify-center">ICON</div>
          <div>
            <div className="text-sm">Signed in as</div>
            <div className="font-semibold">{user.name}</div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button className="px-3 py-2 rounded border" onClick={() => { /* refresh */ }}>Refresh</button>
          <button className="px-3 py-2 rounded bg-gray-800/60" onClick={onSignOut}>Sign out</button>
        </div>
      </header>

      <div className="grid grid-cols-4 gap-6">
        {/* Left: batch list */}
        <div className="col-span-1 bg-[#262626] p-4 rounded">
          <h4 className="mb-3">Batches</h4>

          <div className="mb-3 flex gap-2">
            <input className="flex-1 rounded px-2 py-1 text-black" placeholder="New batch name" />
            <button className="px-3 py-1 rounded bg-emerald-800/70">Add</button>
          </div>

          <div className="space-y-2 max-h-[60vh] overflow-auto">
            {batches.map(b => (
              <button
                key={b.id}
                className={`w-full text-left px-3 py-2 rounded ${b.id === selectedBatchId ? 'bg-blue-800/40' : 'bg-transparent'}`}
                onClick={() => { setSelectedBatchId(b.id); setPageNum(1); }}
              >
                {b.name}
              </button>
            ))}
          </div>
        </div>

        {/* Right: main content */}
        <div className="col-span-3">
          <div className="bg-[#262626] p-4 rounded mb-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <input value={batches.find(b=>b.id===selectedBatchId)?.name || ''} className="px-2 py-1 rounded text-black" />
                <button className="px-3 py-1 rounded border">Update</button>
                <button className="px-3 py-1 rounded border">Delete</button>
                <button className="px-3 py-1 rounded border">Reload samples & stats</button>
              </div>

              <div className="flex gap-2">
                <button className="px-3 py-1 rounded">Export page</button>
                <button className="px-3 py-1 rounded">Export all</button>
                <button className="px-3 py-1 rounded">Export selected</button>
                <button className="px-3 py-1 rounded">Delete selected</button>
              </div>
            </div>
          </div>

          <div className="bg-[#262626] p-4 rounded mb-4">
            <SamplesTable samples={samples} />

            <div className="mt-3 flex items-center justify-between">
              <div>Showing page {pageNum}</div>
              <div className="flex items-center gap-2">
                <button className="px-3 py-1 rounded border" onClick={() => setPageNum(p => Math.max(1, p-1))}>Prev</button>
                <button className="px-3 py-1 rounded border" onClick={() => setPageNum(p => p+1)}>Next</button>
                <select value={perPage} onChange={e=>setPerPage(Number(e.target.value))} className="px-2 py-1 rounded text-black">
                  <option value={5}>5</option>
                  <option value={8}>8</option>
                  <option value={12}>12</option>
                </select>
              </div>
            </div>
          </div>

          <div className="bg-[#262626] p-4 rounded">
            <BatchStats samples={samples} />
          </div>
        </div>
      </div>
    </div>
  );
}

function SamplesTable({ samples }) {
  return (
    <table className="w-full border-collapse">
      <thead>
        <tr className="text-left border-b border-gray-600">
          <th className="py-2">Sample ID</th>
          <th>Class</th>
          <th>Luster</th>
          <th>Roughness</th>
          <th>Tensile</th>
          <th>Date</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {samples.map(s => (
          <tr key={s.id} className="border-b border-gray-700">
            <td className="py-2">{s.id}</td>
            <td>{s.classification}</td>
            <td>{s.luster}</td>
            <td>{s.roughness}</td>
            <td>{s.tensile}</td>
            <td>{s.date_created}</td>
            <td className="text-right"> <button className="px-2 py-1 rounded border">Edit</button> </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function BatchStats({ samples }) {
  const total = samples.length;
  const counts = samples.reduce((acc, s) => { acc[s.classification] = (acc[s.classification] || 0) + 1; return acc; }, {});

  return (
    <div className="flex gap-6 items-center">
      <div>
        <div className="text-sm text-gray-300">Total samples</div>
        <div className="text-3xl font-bold">{total}</div>
      </div>

      <div className="flex gap-4">
        {Object.entries(counts).map(([k,v]) => (
          <div key={k} className="text-center">
            <div className="text-sm text-gray-300">Class {k}</div>
            <div className="font-semibold">{v}</div>
          </div>
        ))}
      </div>

      <div className="flex-1">
        {/* Placeholder for pie chart - replace with recharts or chart.js in production */}
        <div className="h-24 rounded bg-gray-800/30 flex items-center justify-center">Pie chart</div>
      </div>
    </div>
  );
}
