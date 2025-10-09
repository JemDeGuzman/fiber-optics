import React, { useEffect, useState } from "react";
import { View, Text } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { getUser } from "../index";
import TopNav from "../components/Navigation";

export default function HomeScreen({ navigation }: any) {
  const [user, setUser] = useState<any>(null);

  useEffect(() => {
    const loadUser = async () => {
      const token = await AsyncStorage.getItem("token");
      if (!token) return;
      try {
        const data = await getUser(token);
        setUser(data);
      } catch (e) {
        console.warn("Failed to fetch user:", e);
      }
    };
    loadUser();
  }, []);

  return (
    <View style={{ padding: 16 }}>
      <TopNav navigation={navigation} />
      <Text style={{ fontSize: 18, marginTop: 12 }}>
        Welcome to the minimal app
      </Text>
      <Text style={{ marginTop: 8 }}>
        {user ? `Signed in as: ${user.email}` : "Not signed in."}
      </Text>
    </View>
  );
}
