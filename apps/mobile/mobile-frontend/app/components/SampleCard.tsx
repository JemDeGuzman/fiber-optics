// src/components/SampleCard.tsx
import React from "react";
import { View, Text, TouchableOpacity, Switch } from "react-native";
import styled from "styled-components/native";

const Card = styled.View`
  background-color: #262626;
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 12px;
`;

const Row = styled.View`
  flex-direction: row;
  justify-content: space-between;
  margin-bottom: 6px;
`;

const Label = styled.Text`
  color: #C3C8C7;
  font-weight: bold;
`;

const Value = styled.Text`
  color: #EBE1BD;
`;

const EditButton = styled.TouchableOpacity`
  background-color: #3A4946;
  padding: 6px 12px;
  border-radius: 6px;
  align-items: center;
`;

const EditButtonText = styled.Text`
  color: #EBE1BD;
  font-weight: bold;
`;

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
  sample: SampleRow;
  selected?: boolean;
  onSelect?: (id: number) => void;
  onEdit?: (sample: SampleRow) => void;   // ⬅ added
}

export default function SampleCard({ sample, selected = false, onSelect, onEdit }: Props) {
  return (
    <Card>
      <Row>
        <Label>ID:</Label>
        <Value>{sample.id}</Value>
      </Row>

      <Row>
        <Label>Classification:</Label>
        <Value>{sample.classification}</Value>
      </Row>

      <Row>
        <Label>Luster:</Label>
        <Value>{sample.luster_value ?? "-"}</Value>
      </Row>

      <Row>
        <Label>Roughness:</Label>
        <Value>{sample.roughness ?? "-"}</Value>
      </Row>

      <Row>
        <Label>Tensile:</Label>
        <Value>{sample.tensile_strength ?? "-"}</Value>
      </Row>

      <Row>
        <Label>Created:</Label>
        <Value>{new Date(sample.createdAt).toLocaleString()}</Value>
      </Row>

      <Row style={{ justifyContent: "flex-start", alignItems: "center" }}>
        <Switch value={selected} onValueChange={() => onSelect?.(sample.id)} />
        <EditButton style={{ marginLeft: 12 }} onPress={() => onEdit?.(sample)}>
          <EditButtonText>Edit</EditButtonText>
        </EditButton>
      </Row>
    </Card>
  );
}
