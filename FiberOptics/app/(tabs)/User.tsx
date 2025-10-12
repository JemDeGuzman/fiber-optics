import { StyleSheet, TouchableOpacity } from "react-native";
import { Text, View } from "@/components/Themed";
import { useAppTheme } from "@/components/themeContext";

export default function TabTwoScreen() {
  const { mode, toggleTheme, colors } = useAppTheme();

  return (
    <View style={[styles.outerForm, { backgroundColor: colors.background }]}>
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
      <View style={[styles.innerForm, { backgroundColor: colors.background }]}>
        <View style={[styles.centerForm, { backgroundColor: colors.background }]}>
          <Text style={[styles.title, { color: colors.text }]}>App Settings</Text>
        </View>
        <View style={[styles.spacedForm, { backgroundColor: colors.background }]}>
          <Text style={{ color: colors.text }}>Notifications</Text>
          <TouchableOpacity activeOpacity={0.5} style={styles.button}>
            <Text style={[styles.Text, { color: colors.text }]}>{"▼"}</Text>
          </TouchableOpacity>
        </View>
        <View style={[styles.spacedForm, { backgroundColor: colors.background }]}>
          <Text style={{ color: colors.text }}>Link to Device</Text>
          <TouchableOpacity activeOpacity={0.5} style={styles.button}>
            <Text style={[styles.Text, { color: colors.text }]}>{"▼"}</Text>
          </TouchableOpacity>
        </View>
        <View style={[styles.centerForm, { backgroundColor: colors.background }]}>
          <TouchableOpacity
            activeOpacity={0.5}
            style={[styles.button, { backgroundColor: colors.primary }]}
            onPress={() => toggleTheme("light")}
          >
            <Text style={[styles.Text, { color: colors.text }]}>Light Mode</Text>
          </TouchableOpacity>
          <TouchableOpacity
            activeOpacity={0.5}
            style={[styles.button, { backgroundColor: colors.primary }]}
            onPress={() => toggleTheme("dark")}
          >
            <Text style={[styles.Text, { color: colors.text }]}>Dark Mode</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
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
    marginHorizontal: 10,
    borderRadius: 8,
    borderWidth: 1,
    elevation: 3, // adds shadow on Android
  },
  Text: {
    fontSize: 16,
    fontWeight: "600",
  },
});
