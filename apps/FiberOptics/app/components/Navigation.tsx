import React from "react";
import { View, Text, TouchableOpacity } from "react-native";

export default function TopNav({ navigation }: any) {
  return (
    <View
      style={{
        flexDirection: "row",
        padding: 12,
        justifyContent: "space-around",
      }}
    >
      <TouchableOpacity onPress={() => navigation.navigate("Home")}>
        <Text>Home</Text>
      </TouchableOpacity>
      <TouchableOpacity onPress={() => navigation.navigate("Settings")}>
        <Text>Settings</Text>
      </TouchableOpacity>
      <TouchableOpacity onPress={() => navigation.navigate("Login")}>
        <Text>Login</Text>
      </TouchableOpacity>
      <TouchableOpacity onPress={() => navigation.navigate("Register")}>
        <Text>Register</Text>
      </TouchableOpacity>
    </View>
  );
}
