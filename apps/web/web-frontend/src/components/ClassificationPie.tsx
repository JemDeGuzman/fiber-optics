"use client";
import React from "react";
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from "recharts";
import styled from "styled-components";

const COLORS = ["#EBE1BD", "#B5B39C", "#C3C8C7", "#8FB3A9", "#3A4946", "#D4AF37"];

const PieWrapper = styled.div`
  width: 100%;
  height: 350px;
  background-color: #262626;
  border-radius: 12px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  align-items: center;
`;

export default function ClassificationPie({ ratio }: { ratio: Record<string, number> }) {
  const data = Object.entries(ratio).map(([key, value]) => ({ name: key, value }));

  return (
    <PieWrapper>
      <h3 style={{ color: "#EBE1BD", marginBottom: "10px" }}>Class Distribution</h3>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie 
            dataKey="value" 
            data={data} 
            nameKey="name" 
            cx="50%" 
            cy="50%" 
            outerRadius={80} 
            labelLine={false} // Removes messy lines
          >
            {data.map((_, index) => (
              <Cell key={index} fill={COLORS[index % COLORS.length]} stroke="#1f1f1f" />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{ backgroundColor: "#1f1f1f", border: "1px solid #3A4946", color: "#EBE1BD" }}
            itemStyle={{ color: "#EBE1BD" }}
          />
          <Legend verticalAlign="bottom" height={36} />
        </PieChart>
      </ResponsiveContainer>
    </PieWrapper>
  );
}