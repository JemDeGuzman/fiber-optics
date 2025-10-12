import FontAwesome from "@expo/vector-icons/FontAwesome";
import { useFonts } from "expo-font";
import * as SplashScreen from "expo-splash-screen";
import { useEffect, useState } from "react";
import "react-native-reanimated";
import { AppThemeProvider } from "@/components/themeContext";
import { useColorScheme } from "@/components/useColorScheme";
import { View, ActivityIndicator } from "react-native";
import { Slot, Redirect } from "expo-router"; // ✅ import Redirect

export { ErrorBoundary } from "expo-router";

export const unstable_settings = {
  initialRouteName: "(auth)",
};

SplashScreen.preventAutoHideAsync();

export default function RootLayout() {
  const [loaded, error] = useFonts({
    SpaceMono: require("../assets/fonts/SpaceMono-Regular.ttf"),
    ...FontAwesome.font,
  });

  useEffect(() => {
    if (error) throw error;
  }, [error]);

  useEffect(() => {
    if (loaded) SplashScreen.hideAsync();
  }, [loaded]);

  if (!loaded) return null;

  return <RootLayoutNav />;
}

function RootLayoutNav() {
  const colorScheme = useColorScheme();
  const [isLoading, setIsLoading] = useState(true);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // simulate checking token
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
      setIsLoggedIn(false); // toggle this for testing
    }, 1000);
    return () => clearTimeout(timer);
  }, []);

  if (isLoading) {
    return (
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  return (
    <AppThemeProvider>
      {/* ✅ Redirect user based on auth state */}
      {isLoggedIn ? (
        <Redirect href="/(tabs)" />
      ) : (
        <Redirect href="/(auth)/login" />
      )}

      {/* Slot ensures nested layouts still render correctly */}
      <Slot />
    </AppThemeProvider>
  );
}
