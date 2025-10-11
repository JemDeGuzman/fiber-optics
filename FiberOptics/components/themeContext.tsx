// contexts/theme-context.tsx
import React, { createContext, useContext, useState } from "react";
import { DarkTheme, DefaultTheme, ThemeProvider } from "@react-navigation/native";

const ThemeContext = createContext({
  theme: DefaultTheme,
  mode: "light" as "light" | "dark",
  toggleTheme: (mode?: "light" | "dark") => {},
  colors: DefaultTheme.colors,
});

export const useAppTheme = () => useContext(ThemeContext);

export const AppThemeProvider = ({ children }: { children: React.ReactNode }) => {
  const [mode, setMode] = useState<"light" | "dark">("light");

  const toggleTheme = (newMode?: "light" | "dark") => {
    setMode(newMode ?? (mode === "light" ? "dark" : "light"));
  };

  const theme = mode === "dark" ? DarkTheme : DefaultTheme;

  return (
    <ThemeContext.Provider value={{ theme, mode, toggleTheme, colors: theme.colors }}>
      <ThemeProvider value={theme}>{children}</ThemeProvider>
    </ThemeContext.Provider>
  );
};
