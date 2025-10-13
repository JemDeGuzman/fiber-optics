import React from "react";
import { createMaterialTopTabNavigator } from "@react-navigation/material-top-tabs";
// import type { MaterialTopTabBarProps } from "@react-navigation/material-top-tabs";
// import { Text, View, TouchableOpacity } from "react-native";
import HomeScreen from "../../app/(tabs)/index";
import FilesScreen from "../../app/(tabs)/Files";
import UserScreen from "../../app/(tabs)/User";
import { Stack } from "expo-router";

const TopTabs = createMaterialTopTabNavigator();

export default function TopTabsNavigator() {
  return (
    <TopTabs.Navigator
      screenOptions={{
        
        // tabBarStyle: { display: "none" }, // hides the tab bar entirely
        tabBarIndicatorStyle: { backgroundColor: "#007aff" },
        tabBarStyle: { backgroundColor: "#fff" },
        tabBarActiveTintColor: "#007aff",
        tabBarLabelStyle: { fontWeight: "600" },
      }}
    >
        <Stack.Screen options={{ headerShown: false }} />
      <TopTabs.Screen
        name="FiberOptics"
        component={HomeScreen}
        options={{ tabBarLabel: "FiberOptics" }}
      />
      <TopTabs.Screen
        name="Files"
        component={FilesScreen}
        options={{ tabBarLabel: "Files" }}
      />
      <TopTabs.Screen
        name="Home"
        component={HomeScreen}
        options={{ tabBarLabel: "Home" }}
      />
      <TopTabs.Screen
        name="User"
        component={UserScreen}
        options={{ tabBarLabel: "User" }}
      />
    </TopTabs.Navigator>
  );
}
