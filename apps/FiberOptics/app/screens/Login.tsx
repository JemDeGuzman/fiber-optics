import React, { useState } from "react";
import { View, Text, TextInput, TouchableOpacity, Alert } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { loginUser } from "../index";
import TopNav from "../components/Navigation";

export default function LoginScreen({ navigation }: any) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setLoading(true);
    try {
      const json = await loginUser(email, password);
      if (json.token) {
        await AsyncStorage.setItem("token", json.token);
        Alert.alert("Success", "Logged in");
        navigation.navigate("Home");
      } else {
        Alert.alert("Login failed", json.message || "Check credentials");
      }
    } catch (e) {
      Alert.alert("Error", "Network error");
    }
    setLoading(false);
  };

  return (
    <View style={{ padding: 16 }}>
      <TopNav navigation={navigation} />
      <Text style={{ fontSize: 18, marginVertical: 12 }}>Login</Text>
      <Text>Email</Text>
      <TextInput
        value={email}
        onChangeText={setEmail}
        autoCapitalize="none"
        style={{ borderWidth: 1, padding: 8, marginVertical: 8 }}
      />
      <Text>Password</Text>
      <TextInput
        value={password}
        onChangeText={setPassword}
        secureTextEntry
        style={{ borderWidth: 1, padding: 8, marginVertical: 8 }}
      />
      <TouchableOpacity
        onPress={submit}
        disabled={loading}
        style={{ padding: 12, borderWidth: 1, alignItems: "center" }}
      >
        <Text>{loading ? "Signing in..." : "Sign in"}</Text>
      </TouchableOpacity>
      <TouchableOpacity
        onPress={() => navigation.navigate("Register")}
        style={{ marginTop: 12 }}
      >
        <Text>Don't have an account? Register</Text>
      </TouchableOpacity>
    </View>
  );
}
