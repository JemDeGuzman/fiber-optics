import { Text, View } from "@/components/Themed";
import { Button, Platform, StyleSheet, TextInput } from "react-native";

export default function TabOneScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>User Login</Text>
      <Text>Please enter your login credentials below.</Text>
      <View
        style={styles.separator}
        lightColor="#eee"
        darkColor="rgba(255,255,255,0.1)"
      />
      <View style={styles.form}> 
        <TextInput placeholder="User Email" />
        <TextInput placeholder="Password" />
      </View>
      <Button title="Login" onPress={() => {}} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  title: {
    fontSize: 20,
    fontWeight: "bold",
    paddingBottom: 20,
  },
  separator: {
    marginVertical: 30,
    height: 1,
    width: "80%",
  },
  form: {
    width: "60%",
    marginBottom: 20,
    gap: 10,
    alignItems: "center",
    borderColor: "gray",
    borderWidth: 1,
    padding: 10,
    borderRadius: 5,
  },
});
