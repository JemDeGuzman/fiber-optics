import React, { useState } from "react";
import { View, Text, TextInput, TouchableOpacity, Alert } from "react-native";
import { registerUser } from "../index";
import TopNav from "../components/Navigation";

export default function RegisterScreen({ navigation }: any) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setLoading(true);
    try {
      const json = await registerUser(name, email, password);
      if (json.message && !json.error) {
        Alert.alert("Registered", "Please login with your new account");
        navigation.navigate("Login");
      } else {
        Alert.alert("Registration failed", json.message || "Check input");
      }
    } catch (e) {
      Alert.alert("Error", "Network error");
    }
    setLoading(false);
  };

  return (
    <View style={{ padding: 16 }}>
      <TopNav navigation={navigation} />
      <Text style={{ fontSize: 18, marginVertical: 12 }}>Register</Text>
      <Text>Name</Text>
      <TextInput
        value={name}
        onChangeText={setName}
        style={{ borderWidth: 1, padding: 8, marginVertical: 8 }}
      />
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
        <Text>{loading ? "Registering..." : "Create account"}</Text>
      </TouchableOpacity>
    </View>
  );
}
