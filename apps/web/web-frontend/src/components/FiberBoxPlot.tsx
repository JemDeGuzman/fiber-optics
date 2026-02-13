"use client";
import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Bar,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  Scatter,
  RectangleProps
} from "recharts";

/* ===========================
   CUSTOM SHAPES (The Whisker Logic)
=========================== */
const HorizonBar = (props: RectangleProps) => {
  const { x, y, width } = props;
  if (x == null || y == null || width == null) return null;
  return <line x1={x} y1={y} x2={x + width} y2={y} stroke={"#EBE1BD"} strokeWidth={2} />;
};

const DotBar = (props: RectangleProps) => {
  const { x, y, width, height } = props;
  if (x == null || y == null || width == null || height == null) return null;
  return (
    <line
      x1={x + width / 2}
      y1={y + height}
      x2={x + width / 2}
      y2={y}
      stroke={"#EBE1BD"}
      strokeWidth={2}
      strokeDasharray={"3 3"}
    />
  );
};

/* ===========================
   COMPONENT
=========================== */
export const FiberBoxPlot = ({ allSamples }: { allSamples: any[] }) => {
  const data = useMemo(() => {
    const metrics = [
      { key: "luster_value", label: "Luster" },
      { key: "roughness", label: "Roughness" },
      { key: "tensile_strength", label: "Tensile" }
    ];

    return metrics.map((m) => {
      const vals = allSamples
        .map((s) => s[m.key])
        .filter((v) => v !== null && v !== undefined)
        .sort((a, b) => a - b);

      if (vals.length === 0) return { name: m.label, min: 0, bottomWhisker: 0, bottomBox: 0, topBox: 0, topWhisker: 0, average: 0, size: 0 };

      const min = vals[0];
      const q1 = vals[Math.floor(vals.length * 0.25)];
      const median = vals[Math.floor(vals.length * 0.5)];
      const q3 = vals[Math.floor(vals.length * 0.75)];
      const max = vals[vals.length - 1];
      const avg = vals.reduce((a, b) => a + b, 0) / vals.length;

      return {
        name: m.label,
        min: min,
        bottomWhisker: q1 - min,
        bottomBox: median - q1,
        topBox: q3 - median,
        topWhisker: max - q3,
        average: avg,
        size: 100 // Dot size for average
      };
    });
  }, [allSamples]);

  return (
    <div style={{ width: "100%", height: "100%", background: "#262626", padding: "20px", borderRadius: "12px", border: "1px solid #3A4946" }}>
      <h4 style={{ color: "#EBE1BD", marginBottom: "5px" }}>Distribution Analysis</h4>
      <ResponsiveContainer width="100%" height={440}>
        <ComposedChart data={data} margin={{ top: 15, right: 10, left: -20}}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3A4946" vertical={false} />
          <XAxis dataKey="name" stroke="#EBE1BD" fontSize={12} tickLine={false} />
          <YAxis stroke="#EBE1BD" fontSize={10} tickLine={false} />
          <ZAxis type="number" dataKey="size" range={[0, 100]} />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1f1f1f', border: '1px solid #3A4946' }}
            itemStyle={{ color: '#8fb3a9' }}
          />
          
          {/* Stacked Bars to create the Box */}
          <Bar stackId={"a"} dataKey={"min"} fill={"none"} /> 
          <Bar stackId={"a"} dataKey={"bar"} shape={<HorizonBar />} /> {/* Bottom Line */}
          <Bar stackId={"a"} dataKey={"bottomWhisker"} shape={<DotBar />} />
          <Bar stackId={"a"} dataKey={"bottomBox"} fill={"#3A4946"} stroke="#8fb3a9" />
          <Bar stackId={"a"} dataKey={"topBox"} fill={"#3A4946"} stroke="#8fb3a9" />
          <Bar stackId={"a"} dataKey={"topWhisker"} shape={<DotBar />} />
          <Bar stackId={"a"} dataKey={"bar"} shape={<HorizonBar />} /> {/* Top Line */}
          
          <Scatter dataKey="average" fill={"#EBE1BD"} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};