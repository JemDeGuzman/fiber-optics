"use client";
import React, { useEffect, useState } from "react";
import SampleTable, { SampleRow } from "@/components/SampleTable";
import ClassificationPie from "@/components/ClassificationPie";

interface Batch {
  id: number;
  name: string;
  createdAt: string;
}

interface BatchStats {
  total: number;
  ratio: Record<string, number>;
}

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:4000";

export default function Dashboard(): React.JSX.Element {
  const [currentUser, setCurrentUser] = useState<{ id:number; email:string; name?:string } | null>(null);
  const [batches, setBatches] = useState<Batch[]>([]);
  const [selectedBatch, setSelectedBatch] = useState<number | null>(null);

  // pagination state
  const [page, setPage] = useState<number>(1);
  const [limit, setLimit] = useState<number>(25);
  const [totalSamples, setTotalSamples] = useState<number>(0);

  const [samples, setSamples] = useState<SampleRow[]>([]);
  const [stats, setStats] = useState<BatchStats | null>(null);
  const [newBatchName, setNewBatchName] = useState<string>("");
  const [editBatchName, setEditBatchName] = useState<string>("");

  // selected samples for export/delete
  const [selectedSampleIds, setSelectedSampleIds] = useState<number[]>([]);

  // fetch batches
  const fetchBatches = async () => {
    try {
      const res = await fetch(`${API}/api/batches`);
      const data = await res.json();
      setBatches(data.batches || []);
    } catch (err) {
      console.error("fetchBatches error", err);
    }
  };

  const getAuthHeaders = (): Record<string,string> => {
    const token = typeof window !== "undefined" ? localStorage.getItem("token") : null;
    return token ? { Authorization: `Bearer ${token}` } : {};
  };


  // fetch samples + stats with pagination
  const fetchSamplesAndStats = async (batchId: number, p = page, l = limit) => {
    try {
      const [sRes, stRes] = await Promise.all([
        fetch(`${API}/api/batches/${batchId}/samples?page=${p}&limit=${l}`),
        fetch(`${API}/api/batches/${batchId}/stats`)
      ]);
      const sJson = await sRes.json();
      const stJson = await stRes.json();
      setSamples(sJson.samples || []);
      setTotalSamples(sJson.total || 0);
      setPage(sJson.page || p);
      setLimit(sJson.limit || l);
      setStats(stJson || null);
    } catch (err) {
      console.error("fetchSamplesAndStats error", err);
    }
  };

  useEffect(() => {
    fetchBatches();
  }, []);

  useEffect(() => {
    if (selectedBatch) {
      // reset pagination when batch changes
      setPage(1);
      fetchSamplesAndStats(selectedBatch, 1, limit);
      const batchObj = batches.find(b => b.id === selectedBatch);
      setEditBatchName(batchObj?.name ?? "");
    } else {
      setSamples([]);
      setStats(null);
      setEditBatchName("");
    }
  }, [selectedBatch, batches]);

  useEffect(() => {
    const loadMe = async () => {
      try {
        const headers = { "Content-Type": "application/json", ...getAuthHeaders() };
        const res = await fetch(`${API}/api/auth/me`, { headers });
        if (!res.ok) {
          // not logged in or token invalid
          setCurrentUser(null);
          return;
        }
        const data = await res.json();
        setCurrentUser(data.user ?? null);
      } catch (err) {
        console.error("loadMe error", err);
        setCurrentUser(null);
      }
    };
    loadMe();
  }, []);

  // create batch
  const handleAddBatch = async () => {
    if (!newBatchName) return;
    const res = await fetch(`${API}/api/batches`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: newBatchName })
    });
    const data = await res.json();
    if (res.ok && data.batch) {
      setBatches(prev => [data.batch, ...prev]);
      setNewBatchName("");
    } else {
      alert("Failed to add batch: " + (data.error ?? JSON.stringify(data)));
    }
  };

  // update batch name
  const handleUpdateBatch = async () => {
    if (!selectedBatch) return;
    const res = await fetch(`${API}/api/batches/${selectedBatch}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: editBatchName })
    });
    const data = await res.json();
    if (res.ok && data.batch) {
      setBatches(prev => prev.map(b => b.id === selectedBatch ? data.batch : b));
      alert("Batch updated");
    } else {
      alert("Failed to update batch: " + (data.error ?? JSON.stringify(data)));
    }
  };

  // delete batch
  const handleDeleteBatch = async () => {
    if (!selectedBatch) return;
    if (!confirm("Delete batch and all samples?")) return;
    const res = await fetch(`${API}/api/batches/${selectedBatch}`, {
      method: "DELETE"
    });
    if (res.ok) {
      setBatches(prev => prev.filter(b => b.id !== selectedBatch));
      setSelectedBatch(null);
      alert("Deleted");
    } else {
      const data = await res.json();
      alert("Delete failed: " + (data.error ?? JSON.stringify(data)));
    }
  };

  // sample update
  const handleUpdateSample = async (id: number, patch: Partial<SampleRow>) => {
    const res = await fetch(`${API}/api/samples/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch)
    });
    const data = await res.json();
    if (res.ok && data.sample) {
      setSamples(prev => prev.map(s => s.id === id ? data.sample : s));
    } else {
      alert("Update sample failed: " + (data.error ?? JSON.stringify(data)));
    }
  };

  // Deletes selected samples in bulk via backend endpoint:
  // POST /api/samples/deleteMany { ids: number[] }
  const deleteSelectedSamples = async (): Promise<void> => {
    if (!selectedSampleIds || selectedSampleIds.length === 0) {
      alert("No samples selected");
      return;
    }
    if (!confirm(`Delete ${selectedSampleIds.length} sample(s)? This cannot be undone.`)) return;

    try {
      const resp = await fetch(`${API}/api/samples/deleteMany`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: selectedSampleIds }),
      });

      if (!resp.ok) {
        const text = await resp.text();
        console.error("deleteMany failed:", resp.status, text);
        alert("Delete failed: " + (text || resp.statusText));
        return;
      }

      const body = await resp.json();
      // optional: show count returned by backend
      if (body && typeof body.deleted === "number") {
        console.log(`Deleted ${body.deleted} samples`);
      }

      // refresh the current page of samples and stats
      if (selectedBatch) {
        // keep page & limit as-is
        await fetchSamplesAndStats(selectedBatch, page, limit);
      }
      // clear selection of deleted ids
      setSelectedSampleIds([]);
      alert("Selected samples deleted");
    } catch (err) {
      console.error("deleteSelectedSamples error", err);
      alert("Failed to delete selected samples. See console for details.");
    }
  };


  // pagination helpers
  const totalPages = Math.max(1, Math.ceil(totalSamples / limit));
  const gotoPage = (p: number) => {
    if (!selectedBatch) return;
    const next = Math.max(1, Math.min(totalPages, p));
    setPage(next);
    fetchSamplesAndStats(selectedBatch, next, limit);
    setSelectedSampleIds([]);
  };
  const changeLimit = (newLimit: number) => {
    setLimit(newLimit);
    if (selectedBatch) {
      setPage(1);
      fetchSamplesAndStats(selectedBatch, 1, newLimit);
      setSelectedSampleIds([]);
    }
  };

  // CSV export functions
  const exportPageCsv = async () => {
    if (!selectedBatch) return;
    const url = `${API}/api/batches/${selectedBatch}/export?page=${page}&limit=${limit}`;
    const resp = await fetch(url);
    if (!resp.ok) { alert("Export failed"); return; }
    const blob = await resp.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `batch-${selectedBatch}-page-${page}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const exportAllCsv = async () => {
    if (!selectedBatch) return;
    const url = `${API}/api/batches/${selectedBatch}/export`;
    const resp = await fetch(url);
    if (!resp.ok) { alert("Export failed"); return; }
    const blob = await resp.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `batch-${selectedBatch}-all.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const exportSelectedCsv = async () => {
    if (!selectedBatch || selectedSampleIds.length === 0) return;
    const resp = await fetch(`${API}/api/batches/${selectedBatch}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: selectedSampleIds })
    });
    if (!resp.ok) { alert("Selected export failed"); return; }
    const blob = await resp.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `batch-${selectedBatch}-selected.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div style={{ padding: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h1>Welcome to Fiber Optics!</h1>
        <div>
          {currentUser ? (
            <>
              <span style={{ marginRight: 12 }}>Signed in as <strong>{currentUser.name ?? currentUser.email}</strong></span>
              <button onClick={() => {
                localStorage.removeItem("token");
                setCurrentUser(null);
                // optionally redirect to login page
                window.location.href = "/login";
              }}>Logout</button>
            </>
          ) : (
            <a href="/login">Login</a>
          )}
        </div>
      </div>

      <section style={{ marginBottom: 10 }}>
        <h2>Create Batch</h2>
        <input placeholder="Batch name" value={newBatchName} onChange={e => setNewBatchName(e.target.value)} />
        <button onClick={handleAddBatch}>Add Batch</button>
        <button onClick={fetchBatches} style={{ marginLeft: 8 }}>Refresh</button>
      </section>

      <section style={{ display: "flex", gap: 24 }}>
        <div style={{ flex: 1 }}>
          <h3>Batches</h3>
          <ul>
            {batches.map(b => (
              <li key={b.id} style={{ marginBottom: 6 }}>
                <button onClick={() => setSelectedBatch(b.id)} style={{ fontWeight: selectedBatch === b.id ? "bold" : "normal" }}>
                  {b.name}
                </button>
              </li>
            ))}
          </ul>
        </div>

        <div style={{ flex: 3 , marginTop: -110}}>
          {selectedBatch ? (
            <>
              <h3>Batch: {selectedBatch}</h3>

              <div style={{ marginBottom: 8 }}>
                <input value={editBatchName} onChange={e => setEditBatchName(e.target.value)} />
                <button onClick={handleUpdateBatch}>Update name</button>
                <button onClick={handleDeleteBatch} style={{ marginLeft: 8 }}>Delete batch</button>
              </div>

              <div style={{ marginBottom: 16 }}>
                <button onClick={() => fetchSamplesAndStats(selectedBatch)}>Reload samples & stats</button>
              </div>

              <div style={{ marginBottom: 12 }}>
                <button onClick={exportPageCsv}>Export current page CSV</button>
                <button onClick={exportAllCsv} style={{ marginLeft: 8 }}>Export all CSV</button>
                <button onClick={exportSelectedCsv} style={{ marginLeft: 8 }} disabled={selectedSampleIds.length === 0}>Export selected CSV</button>
                <button onClick={deleteSelectedSamples} disabled={selectedSampleIds.length === 0}>Delete selected</button>

              </div>

              <div style={{ marginBottom: 8 }}>
                <span>Selected: {selectedSampleIds.length}</span>
                <button onClick={() => setSelectedSampleIds([])} style={{ marginLeft: 8 }} disabled={selectedSampleIds.length === 0}>
                  Clear selection
                </button>
              </div>

              <div style={{ display: "flex", gap: 16 }}>
                <div style={{ flex: 2 }}>
                  <h4>Samples</h4>
                  <SampleTable
                    samples={samples}
                    onUpdate={handleUpdateSample}
                    selectedIds={selectedSampleIds}
                    onSelectionChange={(ids) => setSelectedSampleIds(ids)}
                  />
                  <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 8 }}>
                    <button onClick={() => gotoPage(page - 1)} disabled={page <= 1}>Prev</button>
                    <span>Page {page} of {totalPages} (total {totalSamples})</span>
                    <button onClick={() => gotoPage(page + 1)} disabled={page >= totalPages}>Next</button>

                    <label style={{ marginLeft: 12 }}>
                      Show
                      <select value={limit} onChange={(e) => changeLimit(Number(e.target.value))} style={{ marginLeft: 6 }}>
                        <option value={10}>10</option>
                        <option value={25}>25</option>
                        <option value={50}>50</option>
                        <option value={100}>100</option>
                      </select>
                      per page
                    </label>
                  </div>
                </div>

                <div style={{ flex: 1 }}>
                  <h4>Stats</h4>
                  {stats ? (
                    <>
                      <p>Total: {stats.total}</p>
                      <ClassificationPie ratio={stats.ratio} />
                    </>
                  ) : <p>Loading stats...</p>}
                </div>
              </div>
            </>
          ) : (
            <div>Select a batch to view samples & stats</div>
          )}
        </div>
      </section>
    </div>
  );
}
