import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export const FiberScatterPlot = ({ samples }: { samples: any[] }) => {
  // Filter out null values so the chart doesn't break
  const data = samples
    .filter(s => s.luster_value !== null && s.roughness !== null)
    .map(s => ({
      x: s.luster_value,
      y: s.roughness,
      name: s.classification
    }));

  return (
    <div style={{ width: '100%', height: 300, background: '#262626', padding: '10px', borderRadius: '8px' }}>
      <h4 style={{ color: '#EBE1BD', marginBottom: '10px' }}>Luster vs. Tensile Correlation</h4>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3A4946" />
          <XAxis type="number" dataKey="x" name="Luster" unit="" stroke="#EBE1BD" />
          <YAxis type="number" dataKey="y" name="Tensile" unit="N" stroke="#EBE1BD" />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          <Scatter name="Samples" data={data} fill="#8fb3a9" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};

export const FiberComparisonScatter = ({ allSamples }: { allSamples: any[] }) => {
  // Separate data by class
  const abacaData = allSamples
    .filter(s => s.classification?.toLowerCase() === 'abaca')
    .map(s => ({ x: s.luster_value, y: s.tensile_strength, z: s.roughness, name: 'Abaca' }));

  const daratexData = allSamples
    .filter(s => s.classification?.toLowerCase() === 'daratex')
    .map(s => ({ x: s.luster_value, y: s.tensile_strength, z: s.roughness, name: 'Daratex' }));

  return (
    <div style={{ width: '100%', height: 510, background: '#262626', padding: '20px', borderRadius: '12px',  border: "1px solid #3A4946" }}>
      <h4 style={{ color: '#EBE1BD', marginBottom: '10px' }}>Abaca vs. Daratex: Luster, Tensile & Roughness</h4>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid stroke="#3A4946" />
          <XAxis type="number" dataKey="x" name="Luster" stroke="#EBE1BD" unit="" />
          <YAxis type="number" dataKey="y" name="Tensile" stroke="#EBE1BD" unit="N" />
          <ZAxis type="number" dataKey="z" range={[50, 400]} name="Roughness" />
          <Tooltip 
            cursor={{ strokeDasharray: '3 3' }} 
            contentStyle={{ backgroundColor: "#1f1f1f", border: "1px solid #3A4946", color: "#EBE1BD" }}
            itemStyle={{ color: "#EBE1BD" }}/>
          <Legend />
          <Scatter name="Abaca" data={abacaData} fill="#B5B39C" shape="circle" />
          <Scatter name="Daratex" data={daratexData} fill="#EBE1BD" shape="diamond" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};