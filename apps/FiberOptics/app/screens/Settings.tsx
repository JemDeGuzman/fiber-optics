import React, { useEffect, useState } from "react";
import { View, Text, TextInput, TouchableOpacity, Alert } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { getSettings, updateSettings } from "../index";
import TopNav from "../components/Navigation";

export default function SettingsScreen({ navigation }: any) {
  const [settings, setSettings] = useState({ theme: "light" });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const load = async () => {
      const token = await AsyncStorage.getItem("token");
      if (!token) return;
      const data = await getSettings(token);
      setSettings(data);
    };
    load();
  }, []);

  const save = async () => {
    setLoading(true);
    const token = await AsyncStorage.getItem("token");
    const res = await updateSettings(token!, settings);
    if (!res.error) Alert.alert("Saved", "Settings updated");
    else Alert.alert("Failed", "Could not save settings");
    setLoading(false);
  };

  const signOut = async () => {
    await AsyncStorage.removeItem("token");
    Alert.alert("Signed out");
    navigation.navigate("Home");
  };

  return (
    <View style={{ padding: 16 }}>
      <TopNav navigation={navigation} />
      <Text style={{ fontSize: 18, marginVertical: 12 }}>Settings</Text>
      <Text>Theme</Text>
      <TextInput
        value={settings.theme}
        onChangeText={(v) => setSettings({ ...settings, theme: v })}
        style={{ borderWidth: 1, padding: 8, marginVertical: 8 }}
      />
      <TouchableOpacity
        onPress={save}
        disabled={loading}
        style={{ padding: 12, borderWidth: 1, alignItems: "center" }}
      >
        <Text>{loading ? "Saving..." : "Save settings"}</Text>
      </TouchableOpacity>
      <TouchableOpacity
        onPress={signOut}
        style={{
          marginTop: 12,
          padding: 12,
          borderWidth: 1,
          alignItems: "center",
        }}
      >
        <Text>Sign out</Text>
      </TouchableOpacity>
    </View>
  );
}
