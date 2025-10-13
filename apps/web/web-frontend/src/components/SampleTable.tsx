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

  return (
    <TableContainer>
      <TableEl>
        <thead>
          <tr>
            <Th><Checkbox type="checkbox" checked={allSelected} onChange={toggleSelectAllOnPage} /></Th>
            <Th>ID</Th>
            <Th>Classification</Th>
            <Th>Luster</Th>
            <Th>Roughness</Th>
            <Th>Tensile</Th>
            <Th>Image</Th>
            <Th>Created</Th>
            <Th></Th>
          </tr>
        </thead>
        <tbody>
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
                  <Input value={edited.luster_value ?? ""} onChange={e => setEdited({ ...edited, luster_value: e.target.value === "" ? null : Number(e.target.value) })} />
                ) : String(s.luster_value ?? "")}
              </Td>

              <Td>
                {editingId === s.id ? (
                  <Input value={edited.roughness ?? ""} onChange={e => setEdited({ ...edited, roughness: e.target.value === "" ? null : Number(e.target.value) })} />
                ) : String(s.roughness ?? "")}
              </Td>

              <Td>
                {editingId === s.id ? (
                  <Input value={edited.tensile_strength ?? ""} onChange={e => setEdited({ ...edited, tensile_strength: e.target.value === "" ? null : Number(e.target.value) })} />
                ) : String(s.tensile_strength ?? "")}
              </Td>

              <Td>
                {editingId === s.id ? (
                  <Input value={edited.image_capture ?? ""} onChange={e => setEdited({ ...edited, image_capture: e.target.value })} />
                ) : s.image_capture}
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
    </TableContainer>
  );
}
