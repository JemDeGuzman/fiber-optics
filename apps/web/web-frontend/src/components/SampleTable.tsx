"use client";
import React, { useMemo } from "react";
import styled from "styled-components";

export interface SampleRow {
  id: number;
  classification: string;
  luster_value: number | null;
  roughness: number | null;
  tensile_strength: number | null;
  image_capture: string | null;
  createdAt: string;
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
  overflow-x: auto;
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

/* ===========================
   COMPONENT
=========================== */
export default function SampleTable({
  samples, 
  onUpdate, 
  selectedIds = [], 
  onSelectionChange, 
  sortKey, 
  sortOrder, 
  onSort
}: Props) {
  const [editingId, setEditingId] = React.useState<number | null>(null);
  const [edited, setEdited] = React.useState<Partial<SampleRow>>({});

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

  const toggleSelectAllOnPage = () => {
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
    setEdited({ ...s });
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

  // Helper to render the sort arrows based on props from page.tsx
  const renderSortIcon = (key: string) => {
    if (sortKey !== key) return <span style={{ color: '#3A4946', marginLeft: '4px' }}>↕</span>;
    return sortOrder === 'asc' ? 
      <span style={{ color: '#EBE1BD', marginLeft: '4px' }}>▲</span> : 
      <span style={{ color: '#EBE1BD', marginLeft: '4px' }}>▼</span>;
  };

  return (
    <TableContainer>
      <TableEl>
        <thead>
          <tr>
            <Th><Checkbox type="checkbox" checked={allSelected} onChange={toggleSelectAllOnPage} /></Th>
            <Th onClick={() => onSort('id')} style={{ cursor: 'pointer' }}>ID {renderSortIcon('id')}</Th>
            <Th onClick={() => onSort('classification')} style={{ cursor: 'pointer' }}>Classification {renderSortIcon('classification')} </Th>
            <Th onClick={() => onSort('luster_value')} style={{cursor:'pointer'}}>Luster {renderSortIcon('luster_value')}</Th>
            <Th onClick={() => onSort('roughness')} style={{cursor:'pointer'}}>Roughness {renderSortIcon('roughness')}</Th>
            <Th onClick={() => onSort('tensile_strength')} style={{cursor:'pointer'}}>Tensile {renderSortIcon('tensile_strength')}</Th>
            <Th>Image</Th>
            <Th>Created</Th>
            <Th>Actions</Th>
          </tr>
        </thead>
        <tbody>
          {/* We map 'samples' directly because page.tsx handles the sorting/filtering via API */}
          {samples.map(s => (
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
                ) : String(s.luster_value ?? "-")}
              </Td>

              <Td>
                {editingId === s.id ? (
                  <Input type="number" value={edited.roughness ?? ""} onChange={e => setEdited({ ...edited, roughness: e.target.value === "" ? null : Number(e.target.value) })} />
                ) : String(s.roughness ?? "-")}
              </Td>

              <Td>
                {editingId === s.id ? (
                  <Input type="number" value={edited.tensile_strength ?? ""} onChange={e => setEdited({ ...edited, tensile_strength: e.target.value === "" ? null : Number(e.target.value) })} />
                ) : String(s.tensile_strength ?? "-")}
              </Td>

              <Td>
                {s.image_capture ? (
                  <Button as="a" href={s.image_capture} target="_blank" rel="noopener noreferrer" small style={{ textDecoration: 'none' }}>
                    View
                  </Button>
                ) : "N/A"}
              </Td>

              <Td>{new Date(s.createdAt).toLocaleDateString()}</Td>

              <Td>
                {editingId === s.id ? (
                  <>
                    <Button small onClick={saveEdit}>Save</Button>
                    <Button small onClick={cancelEdit}>X</Button>
                  </>
                ) : (
                  <Button small onClick={() => startEdit(s)}>Edit</Button>
                )}
              </Td>
            </tr>
          ))}
          {samples.length === 0 && (
            <tr>
              <Td colSpan={9} style={{ textAlign: 'center', padding: '20px' }}>No samples found for this search/batch.</Td>
            </tr>
          )}
        </tbody>
      </TableEl>
    </TableContainer>
  );
}