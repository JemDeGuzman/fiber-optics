import { Stack } from "expo-router";
import { Platform } from "react-native";

export default function AuthLayout() {
  return (
    <Stack screenOptions={{ headerShown: false }}>
      <Stack.Screen name="login" />
      <Stack.Screen
        name="register"
        options={{
          presentation: "modal",
          title: "Register",
          headerShown: true,
          headerTitleAlign: "center",
        }}
      />
    </Stack>
  );
}