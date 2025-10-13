"use client";
import React from "react";
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from "recharts";
import styled from "styled-components";

const COLORS = ["#EBE1BD", "#B5B39C", "#C3C8C7", "#0088FE", "#00C49F", "#FFBB28"];

interface Props {
  ratio: Record<string, number>;
}

/* ===========================
   STYLED COMPONENTS
=========================== */
const PieWrapper = styled.div`
  width: 100%;
  height: 300px;
  background-color: #262626;
  border-radius: 12px;
  padding: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
`;

export default function ClassificationPie({ ratio }: Props) {
  const data = Object.entries(ratio).map(([key, value]) => ({ name: key, value }));

  return (
    <PieWrapper>
      <ResponsiveContainer>
        <PieChart>
          <Pie dataKey="value" data={data} nameKey="name" outerRadius={100} label>
            {data.map((_, index) => (
              <Cell key={index} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f1f1f",
              border: "none",
              color: "#e5e5e5",
            }}
          />
          <Legend
            wrapperStyle={{
              color: "#e5e5e5",
            }}
          />
        </PieChart>
      </ResponsiveContainer>
    </PieWrapper>
  );
}
