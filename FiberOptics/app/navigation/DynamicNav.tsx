import React from "react";
import { useWindowDimensions, Platform } from "react-native";
import { NavigationContainer } from "@react-navigation/native";
import BottomTabsNavigator from "./MobileNav";
import TopTabsNavigator from "./WebNav";
import Header from "../../components/Header";

export default function ResponsiveNavigator() {
  const { width } = useWindowDimensions();
  const isMobile = width < 768 || Platform.OS !== "web";

  return (
    <NavigationContainer>
      {isMobile ? (
        <BottomTabsNavigator />
      ) : (
        <>
          <Header />
          <TopTabsNavigator />
        </>
      )}
    </NavigationContainer>
  );
}
