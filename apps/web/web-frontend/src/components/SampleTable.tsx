"use client";
import React, { useMemo } from "react";
import styled from "styled-components";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

export interface SampleRow {
  id: number;
  classification: string;
  luster_value: number | null;
  roughness: number | null;
  tensile_strength: number | null;
  image_capture: string | null;
  createdAt: string;
  images?: ImageCapture[]; // Optional array of related images
}

export interface ImageCapture {
  id: number;
  fileName: string;
  imageUrl: string;
}

interface Props {
  samples: SampleRow[];
  onUpdate: (id: number, patch: Partial<SampleRow>) => Promise<void>;
  selectedIds?: number[];
  onSelectionChange?: (ids: number[]) => void;
  sortKey: string;
  sortOrder: string;
  onSort: (key: string) => void;
}

/* ===========================
   STYLED COMPONENTS
=========================== */
const TableContainer = styled.div`
  width: 100%;
  overflow-x: visible;
`;

const TableEl = styled.table`
  width: 100%;
  border-collapse: collapse;
  color: #EBE1BD;
`;

const Th = styled.th`
  text-align: left;
  padding: 8px;
  border-bottom: 1px solid #555;
  user-select: none;
  
  /* Sticky Header Logic */
  position: sticky;
  top: 55px;
  background-color: #1f1f1f; /* Must have a background to hide rows sliding under it */
  z-index: 10;
`;

const ModalOverlay = styled.div`
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0, 0, 0, 0.85);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
`;

const ModalContent = styled.div`
  background: #1f1f1f;
  padding: 20px;
  border-radius: 12px;
  max-width: 90vw;
  max-height: 90vh;
  overflow-y: auto;
  border: 1px solid #3A4946;
`;

const ImageGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-top: 15px;
`;

const PreviewImage = styled.img`
  width: 100%;
  border-radius: 8px;
  border: 1px solid #555;
`;

const Td = styled.td`
  padding: 8px;
  border-bottom: 1px solid #333;
`;

const Checkbox = styled.input`
  width: 16px;
  height: 16px;
`;

const Input = styled.input`
  padding: 4px 6px;
  border-radius: 6px;
  border: 1px solid #3A4946;
  background-color: #262626;
  color: #EBE1BD;
  width: 100%;
`;

const Button = styled.button<{ small?: boolean }>`
  padding: ${({ small }) => (small ? "4px 8px" : "6px 10px")};
  border-radius: 6px;
  border: none;
  background-color: #3A4946;
  color: #EBE1BD;
  cursor: pointer;
  margin-right: 4px;

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const StyledLink = styled.a`
  color: #8fb3a9; /* A slightly different color to indicate it's a link */
  text-decoration: underline;
  font-size: 0.9rem;
  cursor: pointer;

  &:hover {
    color: #ebe1bd;
  }
`;

const SortIcon = styled.span<{ active: boolean; direction: 'asc' | 'desc' }>`
  display: inline-block;
  margin-left: 8px;
  width: 0;
  height: 0;
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  
  /* Arrow pointing up or down */
  ${props => props.direction === 'asc' 
    ? `border-bottom: 5px solid ${props.active ? '#EBE1BD' : '#3A4946'};` 
    : `border-top: 5px solid ${props.active ? '#EBE1BD' : '#3A4946'};`}
    
  transition: border-color 0.2s;
`;

/* ===========================
   COMPONENT
=========================== */
export default function SampleTable({samples, onUpdate, selectedIds = [], onSelectionChange, sortKey, sortOrder, onSort}: Props) {
  const [editingId, setEditingId] = React.useState<number | null>(null);
  const [edited, setEdited] = React.useState<Partial<SampleRow>>({});
  const [sortConfig, setSortConfig] = React.useState<{ key: keyof SampleRow, direction: 'asc' | 'desc' } | null>(null);
  const [viewingSample, setViewingSample] = React.useState<SampleRow | null>(null);

  const isSelected = (id: number) => selectedIds.includes(id);

  const toggleSelect = (id: number) => {
    const next = isSelected(id)
      ? selectedIds.filter(i => i !== id)
      : [...selectedIds, id];
    onSelectionChange?.(next);
  };

  const allSelected = useMemo(
    () => samples.length > 0 && samples.every(s => isSelected(s.id)),
    [samples, selectedIds]
  );

  const toggleSelectAllOnPage = (e?: React.ChangeEvent<HTMLInputElement>) => {
    if (allSelected) {
      const next = selectedIds.filter(i => !samples.some(s => s.id === i));
      onSelectionChange?.(next);
    } else {
      const pageIds = samples.map(s => s.id);
      const merged = Array.from(new Set([...selectedIds, ...pageIds]));
      onSelectionChange?.(merged);
    }
  };

  const startEdit = (s: SampleRow) => {
    setEditingId(s.id);
    setEdited({
      classification: s.classification,
      luster_value: s.luster_value,
      roughness: s.roughness,
      tensile_strength: s.tensile_strength,
      image_capture: s.image_capture,
    });
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEdited({});
  };

  const saveEdit = async () => {
    if (editingId == null) return;
    await onUpdate(editingId, edited);
    setEditingId(null);
    setEdited({});
  };

  const renderSortIcon = (key: string) => {
    if (sortKey !== key) return <span style={{ color: '#3A4946', marginLeft: '4px' }}>↕</span>;
    return sortOrder === 'asc' ? 
      <span style={{ color: '#EBE1BD', marginLeft: '4px' }}>▲</span> : 
      <span style={{ color: '#EBE1BD', marginLeft: '4px' }}>▼</span>;
  };

  // Sorting Logic
  const sortedSamples = React.useMemo(() => {
    let sortableItems = [...samples];
    if (sortConfig !== null) {
      sortableItems.sort((a, b) => {
        const aVal = a[sortConfig.key] ?? 0;
        const bVal = b[sortConfig.key] ?? 0;
        if (aVal < bVal) return sortConfig.direction === 'asc' ? -1 : 1;
        if (aVal > bVal) return sortConfig.direction === 'asc' ? 1 : -1;
        return 0;
      });
    }
    return sortableItems;
  }, [samples, sortConfig]);

  return (
    <TableContainer>
      <TableEl>
        <thead>
          <tr>
            <Th><Checkbox type="checkbox" checked={allSelected} onChange={toggleSelectAllOnPage} /></Th>
            <Th onClick={() => onSort('id')} style={{ cursor: 'pointer' }}>ID {renderSortIcon('id')}</Th>
            <Th onClick={() => onSort('classification')} style={{ cursor: 'pointer' }}>Classification {renderSortIcon('classification')} </Th>
            <Th onClick={() => onSort('luster_value')} style={{cursor:'pointer'}}>Luster {renderSortIcon('luster_value')} </Th>
            <Th onClick={() => onSort('roughness')} style={{cursor:'pointer'}}>Roughness {renderSortIcon('roughness')}</Th>
            <Th onClick={() => onSort('tensile_strength')} style={{cursor:'pointer'}}>Tensile {renderSortIcon('tensile_strength')}</Th>
            <Th>Image</Th>
            <Th onClick={() => onSort('createdAt')} style={{ cursor: 'pointer' }}>Created {renderSortIcon('createdAt')}</Th>
            <Th></Th>
          </tr>
        </thead>
        <tbody>
          {sortedSamples.map(s => (
            <tr key={s.id}>
              <Td><Checkbox type="checkbox" checked={isSelected(s.id)} onChange={() => toggleSelect(s.id)} /></Td>
              <Td>{s.id}</Td>

              <Td>
                {editingId === s.id ? (
                  <Input value={(edited.classification as string) ?? ""} onChange={e => setEdited({ ...edited, classification: e.target.value })} />
                ) : s.classification}
              </Td>

              <Td>
                {editingId === s.id ? (
                  <Input type="number" value={edited.luster_value ?? ""} onChange={e => setEdited({ ...edited, luster_value: e.target.value === "" ? null : Number(e.target.value) })} />
                ) : (s.luster_value !== null ? s.luster_value.toFixed(4) : "-")}
              </Td>

              <Td>
                {editingId === s.id ? (
                  <Input type="number" value={edited.roughness ?? ""} onChange={e => setEdited({ ...edited, roughness: e.target.value === "" ? null : Number(e.target.value) })} />
                ) : (s.roughness !== null ? s.roughness.toFixed(4) : "-")}
              </Td>

              <Td>
                {editingId === s.id ? (
                  <Input type="number" value={edited.tensile_strength ?? ""} onChange={e => setEdited({ ...edited, tensile_strength: e.target.value === "" ? null : Number(e.target.value) })} />
                ) : (s.tensile_strength !== null ? s.tensile_strength.toFixed(4) : "-")}
              </Td>

              {/* IMAGE COLUMN - UPDATED FOR MODAL */}
              <Td>
                {(() => {
                  const hasImages = Array.isArray(s.images) && s.images.length > 0;
                  
                  if (!hasImages && !s.image_capture) return "No Image";

                  return (
                    <Button 
                      small 
                      onClick={() => setViewingSample(s)}
                    >
                      View Images ({s.images?.length || 1})
                    </Button>
                  );
                })()}
              </Td>

              <Td>{new Date(s.createdAt).toLocaleString()}</Td>

              <Td>
                {editingId === s.id ? (
                  <>
                    <Button small onClick={saveEdit}>Save</Button>
                    <Button small onClick={cancelEdit}>Cancel</Button>
                  </>
                ) : (
                  <Button small onClick={() => startEdit(s)}>Edit</Button>
                )}
              </Td>
            </tr>
          ))}
        </tbody>
      </TableEl>
      {viewingSample && (
        <ModalOverlay onClick={() => setViewingSample(null)}>
          <ModalContent onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
              <h2 style={{ color: '#EBE1BD', margin: 0 }}>Sample #{viewingSample.id} - All Captures</h2>
              <Button onClick={() => setViewingSample(null)}>Close</Button>
            </div>
            
            <ImageGrid>
              {/* Map through all images in the relation */}
              {viewingSample.images && viewingSample.images.length > 0 ? (
                viewingSample.images.map((img) => (
                  <div key={img.id}>
                    <PreviewImage src={img.imageUrl} alt={img.fileName} />
                    <p style={{ fontSize: '0.8rem', color: '#8fb3a9', marginTop: '5px' }}>{img.fileName}</p>
                  </div>
                ))
              ) : (
                /* Fallback for old records using just image_capture */
                <div>
                  <PreviewImage src={viewingSample.image_capture || ''} alt="Capture" />
                  <p style={{ fontSize: '0.8rem', color: '#8fb3a9' }}>Primary Capture</p>
                </div>
              )}
            </ImageGrid>
          </ModalContent>
        </ModalOverlay>
      )}
    </TableContainer>
  );
}
