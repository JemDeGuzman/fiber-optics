import React, { useState, useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Label } from 'recharts';

const COLORS = ['#7CB342', '#EBE1BD', '#B5B39C', '#4A90E2', '#D32F2F', '#9C27B0', '#F57C00'];

export const FiberComparisonScatter = ({ allSamples }: { allSamples: any[] }) => {
  // 1. Group the data dynamically
  const groupedData = useMemo(() => {
    return allSamples.reduce((acc, sample) => {
      const className = sample.classification || 'Unknown';
      if (!acc[className]) acc[className] = [];
      
      if (sample.luster_value !== null && sample.tensile_strength !== null) {
        acc[className].push({
          x: Number(sample.luster_value),
          y: Number(sample.tensile_strength),
          z: Number(sample.roughness || 0),
          name: className
        });
      }
      return acc;
    }, {} as Record<string, any[]>);
  }, [allSamples]);

  const categories = Object.keys(groupedData);

  // 2. State to track which classes are visible
  const [visibleCategories, setVisibleCategories] = useState<string[]>(categories);

  // Toggle function
  const toggleCategory = (cat: string) => {
    setVisibleCategories(prev => 
      prev.includes(cat) ? prev.filter(c => c !== cat) : [...prev, cat]
    );
  };

  return (
    <div style={{ width: '100%', background: '#262626', padding: '20px', borderRadius: '12px', border: "1px solid #3A4946" }}>
      <h4 style={{ color: '#EBE1BD', marginBottom: '15px' }}>Multi-Class Fiber Comparison</h4>

      {/* 3. FILTER TOGGLE BAR */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginBottom: '20px' }}>
        {categories.map((cat, index) => {
          const isVisible = visibleCategories.includes(cat);
          const color = COLORS[index % COLORS.length];
          return (
            <button
              key={cat}
              onClick={() => toggleCategory(cat)}
              style={{
                background: isVisible ? color : 'transparent',
                color: isVisible ? '#1a1a1a' : color,
                border: `1px solid ${color}`,
                padding: '5px 12px',
                borderRadius: '20px',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: 'bold',
                transition: 'all 0.2s',
                opacity: isVisible ? 1 : 0.5
              }}
            >
              {cat.toUpperCase()} {isVisible ? '✓' : '+'}
            </button>
          );
        })}
      </div>

      {/* 4. THE CHART */}
      <div style={{ width: '100%', height: 400 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}> {/* Increased margins for labels */}
            <CartesianGrid stroke="#3A4946" strokeDasharray="3 3" />
            
            <XAxis type="number" dataKey="x" name="Luster" stroke="#888" fontSize={12}>
                {/* 👈 Added X-Axis Label */}
                <Label 
                  value="Luster (AI Map Unit)" 
                  offset={0} 
                  position="insideBottom" 
                  style={{ textAnchor: 'middle', fill: '#888', fontSize: '12px', fontWeight: 'bold', transform: 'translateY(20px)'}} 
                />
            </XAxis>

            <YAxis type="number" dataKey="y" name="Tensile" stroke="#888" fontSize={12} unit="N">
                {/* 👈 Added Y-Axis Label */}
                <Label 
                  value="Tensile Strength (Newton N)" 
                  angle={-90} 
                  position="insideLeft" 
                  style={{ textAnchor: 'middle', fill: '#888', fontSize: '12px', fontWeight: 'bold' }} 
                />
            </YAxis>

            <ZAxis type="number" dataKey="z" range={[50, 400]} name="Roughness" />
            
            <Tooltip 
              cursor={{ strokeDasharray: '3 3' }}
              contentStyle={{ backgroundColor: "#1f1f1f", border: "1px solid #3A4946", borderRadius: '8px' }}
              itemStyle={{ color: "#FFFFFF" }} 
            />
            
            <Legend verticalAlign="top" align="right" height={36} />
            
            {/* Render only visible categories */}
            {categories.map((category, index) => (
              visibleCategories.includes(category) && (
                <Scatter
                  key={category}
                  name={category}
                  data={groupedData[category]}
                  fill={COLORS[index % COLORS.length]}
                  shape={index % 2 === 0 ? "circle" : "diamond"}
                />
              )
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};