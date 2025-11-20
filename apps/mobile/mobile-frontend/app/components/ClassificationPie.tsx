// src/components/ClassificationPie.tsx
import React from "react";
import { View, Text } from "react-native";
import styled from "styled-components/native";
import { PieChart } from "react-native-gifted-charts";

type Sample = { classification: string };

const LegendRow = styled.View`
  flex-direction: row;
  align-items: center;
  margin-bottom: 6px;
`;

const ColorBox = styled.View`
  width: 16px;
  height: 16px;
  border-radius: 4px;
  margin-right: 10px;
`;

export default function ClassificationPie({ samples }: { samples: Sample[] }) {
  // Count items per classification
  const grouped = samples.reduce((acc, sample) => {
    acc[sample.classification] = (acc[sample.classification] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  // App color palette — stable and theme-consistent
  const colors = [
    "#3A4946",
    "#EBE1BD",
    "#C3C8C7",
    "#6B7A78",
    "#262626",
  ];

  // Build chart data
  const data = Object.keys(grouped).map((classification, idx) => ({
    value: grouped[classification],
    color: colors[idx % colors.length],
    label: `${classification} (${grouped[classification]})`,
    text: `${grouped[classification]}`, // number display
  }));

  return (
    <View style={{ marginBottom: 20 }}>
      <Text
        style={{
          color: "#EBE1BD",
          fontSize: 20,
          fontWeight: "600",
          marginBottom: 16,
          textAlign: "center",
        }}
      >
        Classification Summary
      </Text>

      <PieChart
        data={data}
        radius={110}
        textColor="#000000ff"
        textSize={14}
        showText
        textBackgroundColor="transparent"
        focusOnPress
      />

      {/* Legend */}
      <View style={{ marginTop: 16 }}>
        {data.map((item, index) => (
          <LegendRow key={index}>
            <ColorBox style={{ backgroundColor: item.color }} />
            <Text style={{ color: "#C3C8C7", fontSize: 14 }}>
              {item.label}
            </Text>
          </LegendRow>
        ))}
      </View>
    </View>
  );
}
