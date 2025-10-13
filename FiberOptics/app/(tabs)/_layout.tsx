import { Stack, Tabs } from "expo-router";
import { Platform, useWindowDimensions } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import Header from "../../components/Header";

export default function TabsLayout() {
  const { width, height } = useWindowDimensions();
  const isLandscape = width > height; // true = desktop / landscape
  const isMobile = !isLandscape || Platform.OS !== "web";

  return (
    <>
      {isLandscape && <Header />} {/* show header only in desktop/landscape */}

      <Tabs
      initialRouteName="index"
      screenOptions={({ route }) => ({
          headerShown: false,
          tabBarPosition: isMobile ? "bottom" : "top",
          tabBarShowLabel: true,
          tabBarIndicatorStyle: { backgroundColor: "#007aff" },
          tabBarStyle: {
            backgroundColor: "#fff",
            ...(isMobile
              ? {
                  position: "absolute",
                  bottom: 15,
                  left: 20,
                  right: 20,
                  borderRadius: 25,
                  height: 60,
                  shadowColor: "#000",
                  shadowOpacity: 0.1,
                  shadowRadius: 10,
                  elevation: 10,
                }
              : {}),
          },
          tabBarIcon: ({ focused, color }) => {
            let iconName: keyof typeof Ionicons.glyphMap = "home-outline";
            if (route.name === "previous") iconName = "folder-outline";
            else if (route.name === "settings") iconName = "person-outline";
            else iconName = "home-outline";
            return (
              <Ionicons
                name={iconName}
                size={22}
                color={focused ? "#007aff" : "#999"}
              />
            );
          },
        })}
      >
        <Tabs.Screen
          name="Files"
          options={{ headerShown: false, title: "Previous Samples" }}
        />
        <Tabs.Screen
          name="index"
          options={{ headerShown: false, title: "Current Sample" }}
        />
        <Tabs.Screen
          name="User"
          options={{ headerShown: false, title: "User Settings" }}
        />
      </Tabs>
    </>
  );
}
