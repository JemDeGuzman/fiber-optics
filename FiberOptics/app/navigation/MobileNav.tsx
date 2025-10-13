import React from "react";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { Ionicons } from "@expo/vector-icons";
import CurrentSampleScreen from "../../app/(tabs)/index";
import PreviousSamplesScreen from "../../app/(tabs)/Files";
import SettingsScreen from "../../app/(tabs)/User";

const BottomTabs = createBottomTabNavigator();

export default function BottomTabsNavigator() {
  return (
    <BottomTabs.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarShowLabel: false,
        tabBarStyle: {
          position: "absolute",
          bottom: 15,
          left: 20,
          right: 20,
          borderRadius: 25,
          backgroundColor: "#fff",
          height: 60,
          elevation: 10,
          shadowColor: "#000",
          shadowOpacity: 0.1,
          shadowRadius: 10,
        },
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: keyof typeof Ionicons.glyphMap = "ellipse";

          if (route.name === "Current Sample") iconName = focused ? "home" : "home-outline";
          else if (route.name === "Previous Samples") iconName = focused ? "folder" : "folder-outline";
          else if (route.name === "User Settings") iconName = focused ? "person" : "person-outline";

          return <Ionicons name={iconName} size={26} color={focused ? "#007aff" : "#aaa"} />;
        },
      })}
    >
      <BottomTabs.Screen name="Previous Samples" component={PreviousSamplesScreen} />
      <BottomTabs.Screen name="Current Sample" component={CurrentSampleScreen} />
      <BottomTabs.Screen name="User Settings" component={SettingsScreen} />
    </BottomTabs.Navigator>
  );
}
