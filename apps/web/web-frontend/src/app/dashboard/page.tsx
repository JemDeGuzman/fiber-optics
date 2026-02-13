"use client";
import { useRouter } from 'next/navigation';
import React, { useEffect, useRef, useState, useMemo } from "react";
import SampleTable, { SampleRow } from "@/components/SampleTable";
import ClassificationPie from "@/components/ClassificationPie";
import styled, { createGlobalStyle } from "styled-components";
import { FiberComparisonScatter } from "@/components/FiberScatterPlot";
import { SamplingTrend } from "@/components/SamplingTrend";
import { FiberBoxPlot } from "@/components/FiberBoxPlot";

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
  color: #EBE1BD; 
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
  gap: 12px;
`;

const Logo = styled.img`
  width: 48px;
  height: 48px;
  border-radius: 12px;
  object-fit: contain;
`;

const Toolbar = styled.div`
  display: flex;
  justify-content: space-between;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #EBE1BD;

  position: sticky;
  top: 0;
  background-color: #1f1f1f; /* Matches your wrapper background */
  z-index: 100;              /* Keeps it above table content */
  padding-top: 10px;         /* Prevents text from hitting the very top edge */
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
  padding: 12px 24px;
  border-radius: 12px;
  cursor: pointer;
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  &:hover:not(:disabled) {
    background-color: #EBE1BD;
    color: #3A4946;
  }
`;

const Input = styled.input`
  background-color: #262626;
  color: #EBE1BD;
  border: 1px solid #3A4946;
  border-radius: 12px;
  padding: 12px 16px;
`;

const Select = styled.select`
  background-color: #262626;
  color: #EBE1BD;
  border: 1px solid #3A4946;
  border-radius: 12px;
  padding: 12px 16px;
`;

const TableStatsWrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 32px;
  /* margin-top: 16px; 
  /* border-top: 1px solid #EBE1BD; */
  padding-top: 16px;
`;

const TableWrapper = styled.div`
  width: 100%;
  border-top: 1px solid #EBE1BD;
  padding-top: 24px;
`;

const VisualsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr; /* Two main columns */
  grid-template-areas: 
    "scatter box"
    "bottom bottom"; /* Bottom area spans both columns */
  gap: 16px;
  width: 100%;
  margin-bottom: 24px;

  .area-scatter { grid-area: scatter; min-height: 300px; }
  .area-box { grid-area: box; min-height: 300px; }
  
  .bottom-container {
    grid-area: bottom;
    display: grid;
    grid-template-columns: 280px 1fr; /* Compact Pie, wide Trend */
    gap: 16px;
    min-height: 250px;
  }

  @media (max-width: 1100px) {
    grid-template-columns: 1fr;
    grid-template-areas: "scatter" "box" "bottom";
    .bottom-container { grid-template-columns: 1fr; }
  }
`;

const StatsWrapper = styled.div`
  flex: 1;
`;

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
  const [limit, setLimit] = useState<number>(100);
  const [totalSamples, setTotalSamples] = useState<number>(0);

  const [samples, setSamples] = useState<SampleRow[]>([]);
  const [stats, setStats] = useState<BatchStats | null>(null);
  const [newBatchName, setNewBatchName] = useState<string>("");
  const [editBatchName, setEditBatchName] = useState<string>("");
  const [selectedSampleIds, setSelectedSampleIds] = useState<number[]>([]);

  // Scanner & Device state
  const [showScanner, setShowScanner] = useState(false);
  const [scanMessage, setScanMessage] = useState<string | null>(null);
  const [, setScanning] = useState(false);
  const scannerRef = useRef<any | null>(null);
  const readerElemId = "html5qr-reader";
  const [showDeviceConnector, setShowDeviceConnector] = useState(false);
  const [currentDeviceIp, setCurrentDeviceIp] = useState<string>("Loading...");
  const [connectedDevices, setConnectedDevices] = useState<any[]>([]);

  // Filtering and Sorting State
  const [searchTerm, setSearchTerm] = useState("");
  const [sortKey, setSortKey] = useState("id");
  const [sortOrder, setSortOrder] = useState("desc");

  const [allSamplesForVisuals, setAllSamplesForVisuals] = useState<any[]>([]);

  const router = useRouter();

  const getAuthHeaders = (): Record<string,string> => {
    const token = typeof window !== "undefined" ? localStorage.getItem("token") : null;
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  /* ===========================
     FETCHING LOGIC
  =========================== */
  const fetchBatches = async () => {
    try {
      const res = await fetch(`${API}/api/batches`);
      const data = await res.json();
      setBatches(data.batches || []);
    } catch (err) {
      console.error("fetchBatches error", err);
    }
  };

  const fetchSamplesAndStats = async (batchId: number, p = page, l = limit, search = searchTerm, sKey = sortKey, sOrder = sortOrder) => {
    try {
      const url = `${API}/api/batches/${batchId}/samples?page=${p}&limit=${l}&search=${encodeURIComponent(search)}&sortBy=${sKey}&sortOrder=${sOrder}`;
      const [sRes, stRes] = await Promise.all([
        fetch(url),
        fetch(`${API}/api/batches/${batchId}/stats`)
      ]);
      const sJson = await sRes.json();
      const stJson = await stRes.json();
      
      setSamples(sJson.samples || []);
      setTotalSamples(sJson.total || 0);
      setStats(stJson || null);
    } catch (err) {
      console.error("fetchSamplesAndStats error", err);
    }
  };

  // Fetch visual data (all samples) separately
  const fetchVisualData = async (batchId: number) => {
    const res = await fetch(`${API}/api/batches/${batchId}/visuals`);
    const data = await res.json();
    setAllSamplesForVisuals(data.samples || []);
  };

  useEffect(() => {
    if (selectedBatch) {
      fetchSamplesAndStats(selectedBatch, page, limit, searchTerm, sortKey, sortOrder);
      fetchVisualData(selectedBatch); // Get the full dataset for charts
    }
  }, [selectedBatch, page, limit, searchTerm, sortKey, sortOrder]);

  // MASTER EFFECT: Trigger fetch when page, limit, search, or sort changes
  useEffect(() => {
    if (selectedBatch) {
      fetchSamplesAndStats(selectedBatch, page, limit, searchTerm, sortKey, sortOrder);
    }
  }, [selectedBatch, page, limit, searchTerm, sortKey, sortOrder]);

  // Reset page when filter or sort changes
  useEffect(() => {
    setPage(1);
  }, [searchTerm, sortKey, sortOrder]);

  // Initial Load & Batch Metadata Sync
  useEffect(() => { 
    fetchBatches();
    fetchDeviceInfo();
  }, []);

  useEffect(() => {
    if (selectedBatch) {
      const batchObj = batches.find(b => b.id === selectedBatch);
      setEditBatchName(batchObj?.name ?? "");
    } else {
      setSamples([]);
      setStats(null);
      setEditBatchName("");
    }
  }, [selectedBatch, batches]);

  /* ===========================
     DEVICE & AUTH HANDLERS
  =========================== */
  const fetchDeviceInfo = async () => {
    try {
      const res = await fetch(`${API}/api/devices/info`);
      if (res.ok) {
        const data = await res.json();
        setCurrentDeviceIp(data.deviceIp || "localhost");
      }
    } catch (err) { console.error("Failed to fetch device info:", err); }
  };

  const handleConnectToDevice = async (remoteDeviceIp: string): Promise<boolean> => {
    try {
      const response = await fetch(`${API}/api/devices/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ remoteIp: remoteDeviceIp }),
      });
      if (!response.ok) return false;
      await fetchConnectedDevices();
      return true;
    } catch (err) { return false; }
  };

  const fetchConnectedDevices = async () => {
    try {
      const res = await fetch(`${API}/api/devices/list`);
      if (res.ok) {
        const data = await res.json();
        setConnectedDevices(data.devices || []);
      }
    } catch (err) { console.error("Failed to fetch devices:", err); }
  };

  useEffect(() => {
    const loadMe = async () => {
      try {
        const headers = { "Content-Type": "application/json", ...getAuthHeaders() };
        const res = await fetch(`${API}/api/auth/me`, { headers });
        if (!res.ok) { setCurrentUser(null); return; }
        const data = await res.json();
        setCurrentUser(data.user ?? null);
      } catch (err) { setCurrentUser(null); }
    };
    loadMe();
  }, []);

  /* ===========================
     BATCH & SAMPLE ACTIONS
  =========================== */
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
    }
  };

  const handleUpdateBatch = async () => {
    if (!selectedBatch) return;
    const res = await fetch(`${API}/api/batches/${selectedBatch}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: editBatchName })
    });
    if (res.ok) {
        fetchBatches();
        alert("Batch updated");
    }
  };

  const handleDeleteBatch = async () => {
    if (!selectedBatch || !confirm("Delete batch and all samples?")) return;
    const res = await fetch(`${API}/api/batches/${selectedBatch}`, { method: "DELETE" });
    if (res.ok) {
      setBatches(prev => prev.filter(b => b.id !== selectedBatch));
      setSelectedBatch(null);
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
    }
  };

  const deleteSelectedSamples = async (): Promise<void> => {
    if (!selectedSampleIds.length || !confirm(`Delete ${selectedSampleIds.length} sample(s)?`)) return;
    try {
      const resp = await fetch(`${API}/api/samples/deleteMany`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: selectedSampleIds }),
      });
      if (resp.ok) {
        fetchSamplesAndStats(selectedBatch!, page, limit);
        setSelectedSampleIds([]);
      }
    } catch (err) { console.error(err); }
  };

  /* ===========================
     PAGINATION & EXPORT
  =========================== */
  const totalPages = Math.max(1, Math.ceil(totalSamples / limit));
  
  const gotoPage = (p: number) => {
    if (!selectedBatch) return;
    setPage(Math.max(1, Math.min(totalPages, p)));
    setSelectedSampleIds([]);
  };

  const changeLimit = (newLimit: number) => {
    setLimit(newLimit);
    setPage(1);
    setSelectedSampleIds([]);
  };

  const renderPagination = () => (
    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: '10px 0' }}>
      <Button onClick={() => gotoPage(page - 1)} disabled={page <= 1}>Prev</Button>
      <span style={{ color: '#EBE1BD' }}>
        Page <strong>{page}</strong> of {totalPages} ({totalSamples} total)
      </span>
      <Button onClick={() => gotoPage(page + 1)} disabled={page >= totalPages}>Next</Button>
      
      <label style={{ marginLeft: 'auto', color: '#EBE1BD' }}>
        Show
        <Select value={limit} onChange={(e) => changeLimit(Number(e.target.value))} style={{ margin: '0 8px' }}>
          {[10, 25, 50, 100, 200].map(v => <option key={v} value={v}>{v}</option>)}
        </Select>
        per page
      </label>
    </div>
  );

  const exportPageCsv = async () => { if (!selectedBatch) return; window.open(`${API}/api/batches/${selectedBatch}/export?page=${page}&limit=${limit}`); };
  const exportAllCsv = async () => { if (!selectedBatch) return; window.open(`${API}/api/batches/${selectedBatch}/export`); };
  const exportSelectedCsv = async () => { 
    if (!selectedBatch || !selectedSampleIds.length) return;
    const resp = await fetch(`${API}/api/batches/${selectedBatch}/export`, { 
        method: "POST", 
        headers: { "Content-Type": "application/json" }, 
        body: JSON.stringify({ ids: selectedSampleIds }) 
    });
    const blob = await resp.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `selected-samples.csv`;
    a.click();
  };

  // QR Logic Placeholder (from your code)
  const tryLinkPi = async (payload: any) => { /* logic remains the same */ return true; };

  return (
    <Wrapper>
      <GlobalReset />
      <Header>
        <LogoTitleWrapper>
          <Logo src="/assets/splash-logo.png" alt="FO" />
          <h1>Welcome to Fiber Optics!</h1>
          <Button onClick={() => router.push('dashboard/about')}>About Us</Button>
        </LogoTitleWrapper>
        
        <div>
          {currentUser ? (
            <>
              <span style={{ marginRight: 12 }}>Signed in as <strong>{currentUser.name ?? currentUser.email}</strong></span>
              <Button onClick={() => { localStorage.removeItem("token"); window.location.href = "/login"; }}>Logout</Button>
            </>
          ) : <a href="/login" style={{color: '#EBE1BD'}}>Login</a>}
        </div>
      </Header>

      <Toolbar>
        <ToolbarSection>
          <Input 
            placeholder="🔍 Search ID or Class" 
            value={searchTerm} 
            onChange={(e) => setSearchTerm(e.target.value)} 
            style={{ width: '200px' }}
          />
        </ToolbarSection>

        <ToolbarSection>
          <Input placeholder="New batch" value={newBatchName} onChange={(e) => setNewBatchName(e.target.value)} />
          <Button onClick={handleAddBatch}>Add</Button>
          <Select value={selectedBatch ?? ""} onChange={(e) => setSelectedBatch(Number(e.target.value))}>
            <option value="" disabled>Select Batch</option>
            {batches.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
          </Select>
        </ToolbarSection>

        <ToolbarSection>
          <Input value={editBatchName} onChange={(e) => setEditBatchName(e.target.value)} />
          <Button onClick={handleUpdateBatch} disabled={!selectedBatch}>Update</Button>
          <Button onClick={handleDeleteBatch} disabled={!selectedBatch}>Delete</Button>
        </ToolbarSection>

        <ToolbarSection>
          <Button onClick={exportPageCsv} disabled={!selectedBatch}>Export Page</Button>
          <Button onClick={exportSelectedCsv} disabled={!selectedBatch || !selectedSampleIds.length}>Export Selected</Button>
          <Button onClick={deleteSelectedSamples} disabled={!selectedSampleIds.length}>Delete Selected</Button>
        </ToolbarSection>
      </Toolbar>

      <TableStatsWrapper>
        {stats ? (
        <VisualsGrid>
          <div className="chart-item area-scatter">
            <FiberComparisonScatter allSamples={allSamplesForVisuals} />
          </div>
          <div className="chart-item area-box">
            <FiberBoxPlot allSamples={allSamplesForVisuals} />
          </div>
          <div className="bottom-container">
            <div className="chart-item">
              <ClassificationPie ratio={stats.ratio} />
            </div>
            <div className="chart-item">
              <SamplingTrend samples={allSamplesForVisuals} />
            </div>
          </div>
        </VisualsGrid>
        ) : <p>Please select a Batch to view analytics.</p>}

        <TableWrapper>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h3 style={{ color: '#EBE1BD' }}>Sample Data</h3>
            <span style={{ background: '#3A4946', padding: '4px 12px', borderRadius: '20px', fontSize: '0.85rem' }}>
              Selected: <strong>{selectedSampleIds.length}</strong>
            </span>
          </div>

          {renderPagination()}

          <SampleTable
            samples={samples}
            onUpdate={handleUpdateSample}
            selectedIds={selectedSampleIds}
            onSelectionChange={(ids) => setSelectedSampleIds(ids)}
            sortKey={sortKey}
            sortOrder={sortOrder}
            onSort={(key) => {
              if (key === sortKey) {
                setSortOrder(sortOrder === "asc" ? "desc" : "asc");
              } else {
                setSortKey(key);
                setSortOrder("asc");
              }
            }}
          />

          <div style={{ marginTop: 16 }}>
            {renderPagination()}
          </div>
        </TableWrapper>
      </TableStatsWrapper>

      {/* Modals (Scanner/Connector) would go here as per your original code */}
    </Wrapper>
  );
}