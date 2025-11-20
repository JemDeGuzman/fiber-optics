// src/screens/loginScreen.tsx
import React, { useState } from "react";
import { KeyboardAvoidingView, Platform, Alert, Image as RNImage } from "react-native";
import styled from "styled-components/native";
import { LinearGradient } from 'expo-linear-gradient';
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import AsyncStorage from "@react-native-async-storage/async-storage";
import api from "../api/axiosInstance";
import SignupModal from "../components/SignupModal";

type RootStackParamList = {
  Login: undefined;
  Dashboard: undefined;
};

type Props = NativeStackScreenProps<RootStackParamList, "Login">;

export default function LoginScreen({ navigation }: Props) {
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [showSignup, setShowSignup] = useState<boolean>(false);

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert("Validation", "Please enter email and password.");
      return;
    }
    setLoading(true);
    try {
      const res = await api.post("/api/auth/login", { email, password });
      const data = res.data as { token?: string; error?: string };
      if (res.status === 200 && data.token) {
        await AsyncStorage.setItem("token", data.token);
        navigation.replace("Dashboard");
      } else {
        Alert.alert("Login failed", data.error ?? "Unknown error");
      }
    } catch (err: any) {
      console.error("Login error:", err?.message ?? err);
      Alert.alert("Network error", "Unable to connect to server. Please check API status.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Root pointerEvents="box-none">
      <RightPanel pointerEvents="box-none">
        <BrandTitle>Fiber Optics</BrandTitle>
        <BrandSubtitle>Abaca Classification Tracking System</BrandSubtitle>
        <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : undefined} style={{ width: "100%" }}>
          <Card>
            <Header>
              <LogoWrapper>
                <RNImage source={require("../../assets/splash-logo.png")} style={{ width: 128, height: 128 }} />
              </LogoWrapper>
              <Heading>LOG IN</Heading>
              <SubHeading>Inspector access</SubHeading>
            </Header>

            <Form>
              <Label>Email</Label>
              <TextInput
                value={email}
                onChangeText={setEmail}
                placeholder="you@example.com"
                keyboardType="email-address"
                autoCapitalize="none"
                autoComplete="email"
                placeholderTextColor="rgba(230,246,245,0.5)"
              />

              <Label>Password</Label>
              <TextInput
                value={password}
                onChangeText={setPassword}
                placeholder="At least 8 characters"
                secureTextEntry
                autoComplete="password"
                placeholderTextColor="rgba(230,246,245,0.5)"
              />

              <Actions>
                <Spacer />
                <ActionRight>
                  <GradientContainer>
                    <PrimaryButton onPress={handleLogin} disabled={loading}>
                      <ButtonContent>{loading ? "Signing in..." : "Sign in"}</ButtonContent>
                    </PrimaryButton>
                  </GradientContainer>
                </ActionRight>
              </Actions>
            </Form>

            <Footer>
              <FooterText>Don't have an account?</FooterText>
              <LinkButton
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                onPress={() => {
                  console.log("Signup link pressed!");
                  setShowSignup(true);
                }}
              >
                <FooterText style={{ color: "#EBE1BD", fontWeight: "700" }}>Sign up</FooterText>
              </LinkButton>
            </Footer>
          </Card>
        </KeyboardAvoidingView>
      </RightPanel>

      {showSignup && (
        <SignupModal visible={showSignup} onClose={() => setShowSignup(false)} />
      )}
    </Root>
  );
}

/* Styled components using styled-components/native */
const Root = styled.View`
  flex: 1;
  flex-direction: row;
  background: #0f0f10;
`;


const BrandTitle = styled.Text`
  color: #dbeafe;
  font-size: 20px;
  font-weight: 700;
`;

const BrandSubtitle = styled.Text`
  color: #9fb4c2;
  margin-top: 6px;
  margin-bottom: 12px;
`;

const RightPanel = styled.View`
  flex: 1;
  justify-content: center;
  align-items: center;
  padding: 30px;
`;

const Card = styled.View`
  width: 360px;
  background: rgba(255,255,255,0.02);
  border-radius: 12px;
  padding: 20px;
  border: 1px solid rgba(255,255,255,0.04);
  box-shadow: 0px 8px 24px rgba(2,6,23,0.6);
`;

const Header = styled.View`
  align-items: center;
  margin-bottom: 12px;
`;

const LogoWrapper = styled.View`
  width: 128px;
  height: 128px;
  border-radius: 24px;
  overflow: hidden;
  align-items: center;
  justify-content: center;
`;

const Heading = styled.Text`
  color: #C3C8C7;
  font-weight: 700;
  margin-top: 8px;
`;

const SubHeading = styled.Text`
  color: #9fb4c2;
  margin-top: 2px;
  font-size: 13px;
`;

const Form = styled.View`
  margin-top: 6px;
`;

const Label = styled.Text`
  color: #C3C8C7;
  font-size: 12px;
  margin-bottom: 6px;
`;

const TextInput = styled.TextInput`
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.06);
  background: rgba(255,255,255,0.02);
  color: #eef6f9;
  font-size: 16px;
  margin-bottom: 8px;
`;

const Actions = styled.View`
  width: 100%;
  margin-top: 6px;
  flex-direction: row;
  align-items: center;
`;

const Spacer = styled.View`
  flex: 1;
`;

const ActionRight = styled.View`
  flex-basis: 40%;
`;

const ButtonContent = styled.Text`
  color: #0b1112;
  font-weight: 700;
`;

const GradientContainer = styled(LinearGradient).attrs({
  colors: ['#C3C8C7', '#EBE1BD'],
  start: { x: 0, y: 0.5 }, // Horizontal start point
  end: { x: 1, y: 0.5 },   // Horizontal end point
})`
  padding: 10px;
  border-radius: 5px;
  /* Add shadow/elevation if needed */
`;

const PrimaryButton = styled.TouchableOpacity.attrs({
  activeOpacity: 0.8,
})`
  align-items: center;
  justify-content: center;
  padding-vertical: 6px;
  width: 100%;
  border-radius: 5px;
  ${(p: { disabled?: boolean }) => p.disabled && "opacity: 0.6;"}
  z-index: 1;
`;


/* Because TouchableOpacity doesn't accept children style exports, render the text inline */
const Footer = styled.View`
  margin-top: 18px;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  z-index: 10;
  elevation: 10;
`;

const FooterText = styled.Text`
  color: #C3C8C7;
`;

const LinkButton = styled.TouchableOpacity.attrs({
  activeOpacity: 0.7,
})`
  padding-left: 8px;
  z-index: 20;
  elevation: 20;
`;