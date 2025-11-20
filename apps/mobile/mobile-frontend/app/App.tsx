// App.tsx
import React from "react";
import { StatusBar } from "react-native";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { ThemeProvider } from "styled-components/native";

// <-- IMPORTANT: use exact, case-sensitive paths to your screen files
import LoginScreen from "./screens/loginScreen";
import Dashboard from "./screens/Dashboard";

export type RootStackParamList = {
  Login: undefined;
  Dashboard: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

const theme = {
  colors: {
    background: "#0f0f10",
    text: "#eef6f9",
  },
};

export default function App() {
  return (
    <SafeAreaProvider>
      <ThemeProvider theme={theme}>
        <StatusBar barStyle="light-content" />
          <Stack.Navigator initialRouteName="Login" screenOptions={{ headerShown: false }}>
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="Dashboard" component={Dashboard} />
          </Stack.Navigator>
      </ThemeProvider>
    </SafeAreaProvider>
  );
}
