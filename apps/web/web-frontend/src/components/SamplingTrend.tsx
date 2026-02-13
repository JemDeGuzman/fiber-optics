import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export const SamplingTrend = ({ samples }: { samples: any[] }) => {
  // Group by date
  const counts = samples.reduce((acc: any, s) => {
    const date = new Date(s.createdAt).toLocaleDateString();
    acc[date] = (acc[date] || 0) + 1;
    return acc;
  }, {});

  const data = Object.keys(counts).map(date => ({
    date,
    count: counts[date]
  })).sort((a,b) => new Date(a.date).getTime() - new Date(b.date).getTime());

  return (
    <div style={{ width: '100%', height: 350, background: '#262626', padding: '10px', borderRadius: '8px' }}>
      <h4 style={{ color: '#EBE1BD'}}>Daily Upload Volume</h4>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3A4946" />
          <XAxis 
            dataKey="date" 
            stroke="#EBE1BD" 
            tick={{ fill: '#EBE1BD', fontSize: 15 }}
            interval="preserveStartEnd" // Only shows start/end if crowded
            minTickGap={30}             // Ensures 30px space between date labels
        />
          <YAxis stroke="#EBE1BD" />
          <Tooltip
            contentStyle={{ backgroundColor: "#1f1f1f", border: "1px solid #3A4946", color: "#EBE1BD" }}
            itemStyle={{ color: "#EBE1BD" }}
          />
          <Line type="monotone" dataKey="count" stroke="#3A4946" strokeWidth={2} dot={{ fill: '#3A4946' }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};