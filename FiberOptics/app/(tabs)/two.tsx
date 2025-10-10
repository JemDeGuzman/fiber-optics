import { StyleSheet, TouchableOpacity } from "react-native";

import EditScreenInfo from "@/components/EditScreenInfo";
import { Text, View } from "@/components/Themed";

export default function TabTwoScreen() {
  return (
    <View style={styles.outerForm}>
{/* left half is for user profile settings
-user profile [name, email, password, profile picture]
-logout button
-edit profile button */}
      <View style={styles.innerForm}>
        <View style={styles.centerForm}>
          <Text style={styles.title}>User Profile</Text>
        </View>
        <View style={styles.profileForm}>
          <Text>Name: John Doe</Text>
          <Text>Email: john.doe@example.com</Text>
          <TouchableOpacity activeOpacity={0.5} style={styles.button}>
            <Text style={styles.Text}>Edit</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.spacedForm}>
          <TouchableOpacity activeOpacity={0.5} style={styles.button}>
            <Text style={styles.Text}>Switch Accounts</Text>
          </TouchableOpacity>
          <TouchableOpacity activeOpacity={0.5} style={styles.button}>
            <Text style={styles.Text}>Logout</Text>
          </TouchableOpacity>
        </View>
      </View>
{/* right half is for web/app settings
-light/dark mode
-notification settings
-link to device
-language */}
      <View style={styles.innerForm}>
        <View style={styles.centerForm}>
          <Text style={styles.title}>App Settings</Text>
        </View>
      </View>
    </View>
    // <View style={styles.container}>
    //   <Text style={styles.title}>Tab Two</Text>
    //   <View style={styles.separator} lightColor="#eee" darkColor="rgba(255,255,255,0.1)" />
    //   <EditScreenInfo path="app/(tabs)/two.tsx" />
    // </View>
  );
}

const styles = StyleSheet.create({
  outerForm: {
    flex: 1,
    flexDirection: "row",
    padding: 20,
  },
  innerForm: {
    height: "70%",
    flex: 1,
    flexDirection: "column",
    alignItems: "center",
    // alignContent: "flex-start",
    // justifyContent: "center",
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 10,
    margin: 10,
    padding: 10,
  },
  spacedForm: {
    width: "100%",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 10,
    margin: 10,
    padding: 10,
  },
  centerForm: {
    width: "100%",
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    // borderWidth: 1,
    // borderColor: "#ccc",
    // borderRadius: 10,
    margin: 10,
    padding: 10,
  },
  profileForm: {
    width: "100%",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 10,
    margin: 10,
    padding: 10,
    paddingHorizontal: 15,
  },
  title: {
    fontSize: 20,
    fontWeight: "bold",
  },
  separator: {
    marginVertical: 30,
    height: 1,
    width: "80%",
  },
  button: {
    backgroundColor: "Grey",
    paddingVertical: 10,
    paddingHorizontal: 30,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "Black",
    elevation: 3, // adds shadow on Android
  },
  Text: {
    fontSize: 16,
    fontWeight: "600",
  },
});
