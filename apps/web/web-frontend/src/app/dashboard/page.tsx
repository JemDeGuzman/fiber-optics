"use client";
import React, { useEffect, useRef, useState } from "react";
import SampleTable, { SampleRow } from "@/components/SampleTable";
import ClassificationPie from "@/components/ClassificationPie";
import styled, { createGlobalStyle } from "styled-components";

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

/* ===========================
   GLOBAL STYLES / RESET
=========================== */
const GlobalReset = createGlobalStyle`
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: sans-serif;
  }
`;

const Wrapper = styled.div`
  padding: 20px;
  background-color: #1f1f1f;
  min-height: 100vh;
  color: #EBE1BD; /* Label color */
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
`;

const LogoTitleWrapper = styled.div`
  display: flex;
  align-items: center;
  gap: 12px; // space between logo and title
`;

const Logo = styled.img`
  width: 48px;  // adjust as needed
  height: 48px;
  border-radius: 12px;
  object-fit: contain;
`;

const Toolbar = styled.div`
  display: flex;
  justify-content: space-between;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #EBE1BD; /* Border between toolbar and table */
`;

const ToolbarSection = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
`;

const Button = styled.button`
  background-color: #3A4946;
  color: #EBE1BD;
  border: none;
  padding: 6px 12px;
  border-radius: 6px;
  cursor: pointer;
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const Input = styled.input`
  background-color: #262626;
  color: #EBE1BD;
  border: 1px solid #3A4946;
  border-radius: 6px;
  padding: 6px 10px;
`;

const Select = styled.select`
  background-color: #262626;
  color: #EBE1BD;
  border: 1px solid #3A4946;
  border-radius: 6px;
  padding: 6px 10px;
`;

const TableStatsWrapper = styled.div`
  display: flex;
  gap: 16px;
  margin-top: 16px;
  border-top: 1px solid #EBE1BD; /* Border between table and stats */
  padding-top: 16px;
`;

const TableWrapper = styled.div`
  flex: 2;
`;

const StatsWrapper = styled.div`
  flex: 1;
`;

/* Modal styles for scanner */
const ModalOverlay = styled.div`
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 999;
`;

const ModalContent = styled.div`
  background: #1f1f1f;
  padding: 20px;
  border-radius: 12px;
  color: #EBE1BD;
  width: 90%;
  max-width: 480px;
  text-align: center;
  box-shadow: 0 10px 30px rgba(0,0,0,0.6);
`;

export default function Dashboard(): React.JSX.Element {
  const [currentUser, setCurrentUser] = useState<{ id:number; email:string; name?:string } | null>(null);
  const [batches, setBatches] = useState<Batch[]>([]);
  const [selectedBatch, setSelectedBatch] = useState<number | null>(null);

  const [page, setPage] = useState<number>(1);
  const [limit, setLimit] = useState<number>(25);
  const [totalSamples, setTotalSamples] = useState<number>(0);

  const [samples, setSamples] = useState<SampleRow[]>([]);
  const [stats, setStats] = useState<BatchStats | null>(null);
  const [newBatchName, setNewBatchName] = useState<string>("");
  const [editBatchName, setEditBatchName] = useState<string>("");
  const [selectedSampleIds, setSelectedSampleIds] = useState<number[]>([]);

  // scanner state
  const [showScanner, setShowScanner] = useState(false);
  const [scanMessage, setScanMessage] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const scannerRef = useRef<any | null>(null);
  const readerElemId = "html5qr-reader";

  const getAuthHeaders = (): Record<string,string> => {
    const token = typeof window !== "undefined" ? localStorage.getItem("token") : null;
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const fetchBatches = async () => {
    try {
      const res = await fetch(`${API}/api/batches`);
      const data = await res.json();
      setBatches(data.batches || []);
    } catch (err) {
      console.error("fetchBatches error", err);
    }
  };

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

  useEffect(() => { fetchBatches(); }, []);
  useEffect(() => {
    if (selectedBatch) {
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
        if (!res.ok) { setCurrentUser(null); return; }
        const data = await res.json();
        setCurrentUser(data.user ?? null);
      } catch (err) {
        console.error("loadMe error", err);
        setCurrentUser(null);
      }
    };
    loadMe();
  }, []);

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

  const handleDeleteBatch = async () => {
    if (!selectedBatch) return;
    if (!confirm("Delete batch and all samples?")) return;
    const res = await fetch(`${API}/api/batches/${selectedBatch}`, { method: "DELETE" });
    if (res.ok) {
      setBatches(prev => prev.filter(b => b.id !== selectedBatch));
      setSelectedBatch(null);
      alert("Deleted");
    } else {
      const data = await res.json();
      alert("Delete failed: " + (data.error ?? JSON.stringify(data)));
    }
  };

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

  const deleteSelectedSamples = async (): Promise<void> => {
    if (!selectedSampleIds || selectedSampleIds.length === 0) {
      alert("No samples selected"); return;
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
      if (body && typeof body.deleted === "number") console.log(`Deleted ${body.deleted} samples`);
      if (selectedBatch) await fetchSamplesAndStats(selectedBatch, page, limit);
      setSelectedSampleIds([]);
      alert("Selected samples deleted");
    } catch (err) {
      console.error("deleteSelectedSamples error", err);
      alert("Failed to delete selected samples. See console for details.");
    }
  };

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

  const exportPageCsv = async () => { if (!selectedBatch) return; const url = `${API}/api/batches/${selectedBatch}/export?page=${page}&limit=${limit}`; const resp = await fetch(url); if (!resp.ok) { alert("Export failed"); return; } const blob = await resp.blob(); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `batch-${selectedBatch}-page-${page}.csv`; a.click(); URL.revokeObjectURL(a.href); };
  const exportAllCsv = async () => { if (!selectedBatch) return; const url = `${API}/api/batches/${selectedBatch}/export`; const resp = await fetch(url); if (!resp.ok) { alert("Export failed"); return; } const blob = await resp.blob(); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `batch-${selectedBatch}-all.csv`; a.click(); URL.revokeObjectURL(a.href); };
  const exportSelectedCsv = async () => { if (!selectedBatch || selectedSampleIds.length === 0) return; const resp = await fetch(`${API}/api/batches/${selectedBatch}/export`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids: selectedSampleIds }) }); if (!resp.ok) { alert("Selected export failed"); return; } const blob = await resp.blob(); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `batch-${selectedBatch}-selected.csv`; a.click(); URL.revokeObjectURL(a.href); };

  // ---------- QR Scanner handling ----------
  // Attempt to POST the scanned payload to the Pi capture-server /connect endpoint.
  // We try to auto-detect a Pi host inside the scanned JSON (common keys), otherwise prompt user.
  const tryLinkPi = async (payload: any) => {
    // payload: parsed JSON from QR
    // prefer explicit pi host fields if present
    const candidateKeys = [
      "piHost","piBase","captureServer","capture_host","capture_url",
      "piUrl","pi_url","host","server","capture","backendUrl","backend","url"
    ];

    let targetBase: string | null = null;

    // if payload contains exact "piHost" use it first
    for (const k of candidateKeys) {
      const v = payload[k];
      if (!v) continue;
      const s = String(v).trim();
      // skip empty strings
      if (!s) continue;
      // if it looks like a URL (http/https) accept it
      if (/^https?:\/\//i.test(s)) {
        targetBase = s.replace(/\/$/, "");
        break;
      }
      // if it's just an IP/host:port, coerce to http
      if (/^[\d.]+(:\d+)?$/.test(s) || /^[a-z0-9.-]+(:\d+)?$/i.test(s)) {
        targetBase = (s.startsWith("http") ? s : `http://${s}`).replace(/\/$/, "");
        break;
      }
    }

    // As a last-resort, if the QR contains a "backendUrl" we still might use that as info,
    // but we need the Pi endpoint to POST to; so prompt the user for Pi's address.
    if (!targetBase) {
      // prompt the user (fall back to manual entry)
      const manual = window.prompt("Pi host not found in QR. Enter capture-server base URL (e.g. http://192.168.1.123:3001):");
      if (!manual) {
        setScanMessage("No Pi host provided - cancelled.");
        return false;
      }
      targetBase = manual.replace(/\/$/, "");
    }

    const connectUrl = `${targetBase}/connect`;
    setScanMessage(`Connecting to ${connectUrl}...`);
    try {
      const resp = await fetch(connectUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const isJson = (resp.headers.get("content-type") || "").includes("application/json");
      const body = isJson ? await resp.json() : await resp.text();

      if (!resp.ok) {
        console.error("connect failed", resp.status, body);
        setScanMessage(`Connect failed: ${resp.status} ${typeof body === "string" ? body : JSON.stringify(body)}`);
        return false;
      }

      setScanMessage("Connected successfully ✅");
      // Optionally: refresh batches to reflect any new connection state
      await fetchBatches();
      // close scanner after a short delay so user sees success
      setTimeout(() => setShowScanner(false), 900);
      return true;
    } catch (err) {
      console.error("connect error", err);
      setScanMessage("Connect request failed: " + String(err));
      return false;
    }
  };

  // ---------- html5-qrcode integration ----------
  useEffect(() => {
    // start scanner when modal opens
    let mounted = true;
    let instance: any = null;

    const startScanner = async () => {
      if (!showScanner) return;
      setScanMessage("Initializing camera...");
      try {
        const mod = await import("html5-qrcode");
        const { Html5Qrcode } = mod;
        if (!mounted) return;
        // create instance targeting div id
        instance = new Html5Qrcode(readerElemId, /* verbose= */ false);
        scannerRef.current = instance;

        const config = { fps: 10, qrbox: { width: 280, height: 280 } };

        await instance.start(
          // camera config - prefer environment (rear) camera
          { facingMode: "environment" } as any,
          config,
          async (decodedText: string, decodedResult: any) => {
            if (!decodedText) return;
            setScanMessage("QR scanned, processing...");
            setScanning(true);
            try {
              // parse payload
              let parsed: any;
              try {
                parsed = JSON.parse(decodedText);
              } catch (e) {
                setScanMessage("Scanned value is not JSON: " + decodedText.slice(0, 200));
                setScanning(false);
                return;
              }

              const ok = await tryLinkPi(parsed);
              setScanning(false);
              if (ok) {
                setScanMessage("Linked!");
                // stop scanner - tryLinkPi will close modal shortly
                try { await instance.stop(); } catch (e) { /* ignore */ }
              }
            } catch (err) {
              console.error("scan callback error", err);
              setScanMessage("Scan handling error: " + String(err));
              setScanning(false);
            }
          },
          (errorMessage: string) => {
            // camera frame decode errors - ignore or log
            // console.debug("QR decode frame error", errorMessage);
          }
        );

        setScanMessage("Point your camera at the Pi's QR code");
      } catch (err: any) {
        console.error("html5-qrcode start error", err);
        setScanMessage("Camera init error: " + (err?.message ?? String(err)));
      }
    };

    startScanner();

    return () => {
      mounted = false;
      if (scannerRef.current) {
        try {
          scannerRef.current.stop().catch(() => {});
        } catch (e) {}
        scannerRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showScanner]);

  return (
    <Wrapper>
      <GlobalReset />
      <Header>
        <LogoTitleWrapper>
          <Logo src="/assets/splash-logo.png" alt="FO" />
          <h1>Welcome to Fiber Optics!</h1>
        </LogoTitleWrapper>

        <div>
          {currentUser ? (
            <>
              <span style={{ marginRight: 12 }}>
                Signed in as <strong>{currentUser.name ?? currentUser.email}</strong>
              </span>
              <Button onClick={() => {
                localStorage.removeItem("token");
                setCurrentUser(null);
                window.location.href = "/login";
              }}>Logout</Button>
            </>
          ) : <a href="/login">Login</a>}
        </div>
      </Header>

      {/* ===================== TOOLBAR ===================== */}
      <Toolbar>
        {/* Left section: Batch Creation & Selection */}
        <ToolbarSection>
          <Input placeholder="New batch" value={newBatchName} onChange={e => setNewBatchName(e.target.value)} />
          <Button onClick={handleAddBatch}>Add</Button>
          <Button onClick={fetchBatches}>Reload</Button>
          <Select value={selectedBatch ?? ""} onChange={e => setSelectedBatch(Number(e.target.value))}>
            <option value="" disabled>Select Batch</option>
            {batches.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
          </Select>
        </ToolbarSection>

        {/* Middle section: Batch Editing */}
        <ToolbarSection>
          <Input value={editBatchName} onChange={e => setEditBatchName(e.target.value)} />
          <Button onClick={handleUpdateBatch} disabled={!selectedBatch}>Update</Button>
          <Button onClick={handleDeleteBatch} disabled={!selectedBatch}>Delete</Button>
          <Button onClick={() => selectedBatch && fetchSamplesAndStats(selectedBatch)} disabled={!selectedBatch}>Reload Samples</Button>
        </ToolbarSection>

        {/* Right section: Data Export + Scanner */}
        <ToolbarSection>
          <Button onClick={exportPageCsv} disabled={!selectedBatch}>Export Page</Button>
          <Button onClick={exportAllCsv} disabled={!selectedBatch}>Export All</Button>
          <Button onClick={exportSelectedCsv} disabled={!selectedBatch || selectedSampleIds.length === 0}>Export Selected</Button>
          <Button onClick={deleteSelectedSamples} disabled={selectedSampleIds.length === 0}>Delete Selected</Button>

          {/* Scanner trigger */}
          <Button onClick={() => { setScanMessage(null); setShowScanner(true); }}>Scan QR</Button>
        </ToolbarSection>
      </Toolbar>

      {/* Show selected sample count */}
      <p style={{ marginTop: 8 }}>Selected: {selectedSampleIds.length}</p>

      {/* ===================== TABLE + STATS ===================== */}
      <TableStatsWrapper>
        <TableWrapper>
          <SampleTable
            samples={samples}
            onUpdate={handleUpdateSample}
            selectedIds={selectedSampleIds}
            onSelectionChange={(ids) => setSelectedSampleIds(ids)}
          />
          <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 8 }}>
            <Button onClick={() => gotoPage(page - 1)} disabled={page <= 1}>Prev</Button>
            <span>Page {page} of {totalPages} (total {totalSamples})</span>
            <Button onClick={() => gotoPage(page + 1)} disabled={page >= totalPages}>Next</Button>
            <label style={{ marginLeft: 12 }}>
              Show
              <Select value={limit} onChange={(e) => changeLimit(Number(e.target.value))} style={{ marginLeft: 6 }}>
                <option value={10}>10</option>
                <option value={25}>25</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </Select>
              per page
            </label>
          </div>
        </TableWrapper>

        <StatsWrapper>
          <h4>Stats</h4>
          {stats ? (
            <>
              <p>Total: {stats.total}</p>
              <ClassificationPie ratio={stats.ratio} />
            </>
          ) : <p>Loading stats...</p>}
        </StatsWrapper>
      </TableStatsWrapper>

      {/* Scanner modal */}
      {showScanner && (
        <ModalOverlay onClick={() => setShowScanner(false)}>
          <ModalContent onClick={(e) => e.stopPropagation()}>
            <h3 style={{ marginBottom: 8 }}>Scan Pi QR to connect</h3>
            <p style={{ color: "#C3C8C7", marginBottom: 12 }}>{scanMessage ?? "Point your camera at the Pi's QR code"}</p>

            <div id={readerElemId} style={{ width: "100%", maxWidth: 400, margin: "0 auto" }} />

            <div style={{ display: "flex", gap: 8, justifyContent: "center", marginTop: 12 }}>
              <Button onClick={async () => {
                // manual retry: stop if running then re-open scanner
                try {
                  if (scannerRef.current) {
                    await scannerRef.current.stop();
                    scannerRef.current = null;
                  }
                } catch (e) { /* ignore */ }
                setScanMessage("Retrying...");
                // small delay to allow stop to complete
                setTimeout(() => setShowScanner(true), 250);
              }}>Retry</Button>

              <Button onClick={() => {
                // close and cleanup
                (async () => {
                  try { if (scannerRef.current) await scannerRef.current.stop(); } catch (e) {}
                  scannerRef.current = null;
                  setShowScanner(false);
                })();
              }}>Close</Button>
            </div>
          </ModalContent>
        </ModalOverlay>
      )}
    </Wrapper>
  );
}