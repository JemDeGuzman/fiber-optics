"use client";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import styled from "styled-components";

const BarWrapper = styled.div`
  width: 100%;
  height: 350px;
  background-color: #262626;
  border-radius: 12px;
  padding: 16px;
`;

export default function ClassificationBar({ ratio }: { ratio: Record<string, number> }) {
  const data = Object.entries(ratio).map(([key, value]) => ({ name: key, count: value }));

  return (
    <BarWrapper>
      <h3 style={{ color: "#EBE1BD", marginBottom: "10px" }}>Samples per Class</h3>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
          <XAxis dataKey="name" stroke="#EBE1BD" fontSize={12} />
          <YAxis stroke="#EBE1BD" fontSize={12} />
          <Tooltip cursor={{fill: '#2f2f2f'}} contentStyle={{ backgroundColor: "#1f1f1f", border: "1px solid #3A4946" }} />
          <Bar dataKey="count" fill="#8FB3A9" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </BarWrapper>
  );
}