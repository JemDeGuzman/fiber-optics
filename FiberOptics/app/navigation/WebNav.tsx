import React from "react";
import { createMaterialTopTabNavigator } from "@react-navigation/material-top-tabs";
import CurrentSampleScreen from "../../app/(tabs)/index";
import PreviousSamplesScreen from "../../app/(tabs)/PrevSamples";
import SettingsScreen from "../../app/(tabs)/User";

const TopTabs = createMaterialTopTabNavigator();

export default function TopTabsNavigator() {
  return (
    <TopTabs.Navigator
      screenOptions={{
        tabBarIndicatorStyle: { backgroundColor: "#007aff" },
        tabBarStyle: { backgroundColor: "#fff" },
        tabBarActiveTintColor: "#007aff",
        tabBarLabelStyle: { fontWeight: "600" },
      }}
    >
      <TopTabs.Screen name="Current Sample" component={CurrentSampleScreen} />
      <TopTabs.Screen name="Previous Samples" component={PreviousSamplesScreen} />
      <TopTabs.Screen name="User Settings" component={SettingsScreen} />
    </TopTabs.Navigator>
  );
}
