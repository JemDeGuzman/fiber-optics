// web-frontend/src/components/SampleTable.tsx
"use client";
import React, { useMemo } from "react";

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
  selectedIds?: number[]; // controlled selection across pages
  onSelectionChange?: (ids: number[]) => void;
}

export default function SampleTable({
  samples,
  onUpdate,
  selectedIds = [],
  onSelectionChange,
}: Props) {
  const [editingId, setEditingId] = React.useState<number | null>(null);
  const [edited, setEdited] = React.useState<Partial<SampleRow>>({});

  // local helper to check if id is selected (driven by selectedIds prop)
  const isSelected = (id: number) => selectedIds.includes(id);

  // toggle selection for a single id
  const toggleSelect = (id: number) => {
    const next = isSelected(id)
      ? selectedIds.filter(i => i !== id)
      : [...selectedIds, id];
    onSelectionChange?.(next);
  };

  // toggle select all on current page
  const allSelected = useMemo(
    () => samples.length > 0 && samples.every(s => isSelected(s.id)),
    [samples, selectedIds]
  );

  const toggleSelectAllOnPage = () => {
    if (allSelected) {
      // remove page ids
      const next = selectedIds.filter(i => !samples.some(s => s.id === i));
      onSelectionChange?.(next);
    } else {
      // add all page ids (avoid duplicates)
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
    <div>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ textAlign: "left", borderBottom: "1px solid #ddd" }}>
            <th><input type="checkbox" checked={allSelected} onChange={toggleSelectAllOnPage} /></th>
            <th>ID</th>
            <th>Classification</th>
            <th>Luster</th>
            <th>Roughness</th>
            <th>Tensile</th>
            <th>Image</th>
            <th>Created</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {samples.map(s => (
            <tr key={s.id} style={{ borderBottom: "1px solid #f0f0f0" }}>
              <td style={{ padding: 8 }}>
                <input
                  type="checkbox"
                  checked={isSelected(s.id)}
                  onChange={() => toggleSelect(s.id)}
                />
              </td>

              <td style={{ padding: 8 }}>{s.id}</td>

              <td style={{ padding: 8 }}>
                {editingId === s.id ? (
                  <input value={(edited.classification as string) ?? ""} onChange={(e)=> setEdited({...edited, classification: e.target.value})} />
                ) : s.classification}
              </td>

              <td style={{ padding: 8 }}>
                {editingId === s.id ? (
                  <input value={edited.luster_value ?? ""} onChange={(e)=> setEdited({...edited, luster_value: e.target.value === "" ? null : Number(e.target.value)})} />
                ) : String(s.luster_value ?? "")}
              </td>

              <td style={{ padding: 8 }}>
                {editingId === s.id ? (
                  <input value={edited.roughness ?? ""} onChange={(e)=> setEdited({...edited, roughness: e.target.value === "" ? null : Number(e.target.value)})} />
                ) : String(s.roughness ?? "")}
              </td>

              <td style={{ padding: 8 }}>
                {editingId === s.id ? (
                  <input value={edited.tensile_strength ?? ""} onChange={(e)=> setEdited({...edited, tensile_strength: e.target.value === "" ? null : Number(e.target.value)})} />
                ) : String(s.tensile_strength ?? "")}
              </td>

              <td style={{ padding: 8 }}>
                {editingId === s.id ? (
                  <input value={edited.image_capture ?? ""} onChange={(e)=> setEdited({...edited, image_capture: e.target.value})} />
                ) : s.image_capture}
              </td>

              <td style={{ padding: 8 }}>{new Date(s.createdAt).toLocaleString()}</td>

              <td style={{ padding: 8 }}>
                {editingId === s.id ? (
                  <>
                    <button onClick={saveEdit}>Save</button>
                    <button onClick={cancelEdit}>Cancel</button>
                  </>
                ) : (
                  <>
                    <button onClick={() => startEdit(s)}>Edit</button>
                  </>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
