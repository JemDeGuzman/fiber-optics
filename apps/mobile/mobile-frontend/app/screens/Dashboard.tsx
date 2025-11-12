// src/screens/Dashboard.tsx
import React from "react";
import { View, Text, Button } from "react-native";
import styled from "styled-components/native";

export default function Dashboard({ navigation }: any) {
  return (
    <Root>
      <Text style={{ color: "#fff", fontSize: 18, marginBottom: 12 }}>
        Dashboard (placeholder)
      </Text>
      <Button title="Log out" onPress={() => navigation.replace("Login")} />
    </Root>
  );
}

const Root = styled.View`
  flex: 1;
  background: #0f0f10;
  align-items: center;
  justify-content: center;
`;
