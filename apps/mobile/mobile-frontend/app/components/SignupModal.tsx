// src/components/SignupModal.tsx
import React, { useEffect, useRef, useState } from "react";
import {
  Modal,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
  Alert,
  TextInput
} from "react-native";
import styled from "styled-components/native";
import type { TextInput as RNTextInput } from "react-native";
import { z } from "zod";
import api from "../api/axiosInstance";

interface Props {
  visible: boolean;
  onClose: () => void;
}

const signupSchema = z
  .object({
    name: z.string().min(2, "Name must be at least 2 characters"),
    email: z.string().email("Invalid email address"),
    password: z.string().min(8, "Password must be at least 8 characters").max(128),
    passwordConfirm: z.string(),
  })
  .refine((d) => d.password === d.passwordConfirm, {
    message: "Passwords do not match",
    path: ["passwordConfirm"],
  });

export default function SignupModal({ visible, onClose }: Props) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [passwordConfirm, setPasswordConfirm] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const nameRef = useRef<RNTextInput | null>(null);

  useEffect(() => {
    if (visible) {
      // focus the name input when the modal becomes visible
      nameRef.current?.focus?.();
      setErrors({});
    } else {
      setName("");
      setEmail("");
      setPassword("");
      setPasswordConfirm("");
      setErrors({});
    }
  }, [visible]);

  const handleSignup = async () => {
    setErrors({});
    const result = signupSchema.safeParse({
      name,
      email,
      password,
      passwordConfirm,
    });

    if (!result.success) {
      const errObj: Record<string, string> = {};
      for (const issue of result.error.issues) {
        const path = issue.path[0] as string | undefined;
        const key = path ?? "form";
        if (!errObj[key]) errObj[key] = issue.message;
      }
      setErrors(errObj);
      return;
    }

    setSubmitting(true);
    try {
      const res = await api.post("/api/auth/signup", {
        name,
        email,
        password,
        passwordConfirm,
      });
      if (res.status === 200) {
        Alert.alert("Success", "Signup successful! You can now log in.");
        onClose();
      } else {
        setErrors({ form: res.data?.error ?? JSON.stringify(res.data) });
      }
    } catch (err) {
      console.error("Signup error:", err);
      setErrors({ form: "Unable to contact server. Please try again." });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal visible={visible} animationType="fade" transparent onRequestClose={onClose}>
      <Overlay>
        <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : undefined} style={{ flex: 1, width: "100%" }}>
          <Container contentContainerStyle={{ alignItems: "center", justifyContent: "center", padding: 20 }}>
            <Dialog>
              <Close onPress={onClose} accessibilityLabel="Close signup dialog">
                <CloseText>×</CloseText>
              </Close>
              <Title>Create an account</Title>

              <Field>
                <Input
                  ref={nameRef as any}
                  value={name}
                  onChangeText={setName}
                  placeholder="Lebron"
                  placeholderTextColor="#93a3ad"
                  editable={!submitting}
                />
                {errors.name && <ErrorText>{errors.name}</ErrorText>}
              </Field>

              <Field>
                <Label>Email</Label>
                <Input
                  value={email}
                  onChangeText={setEmail}
                  placeholder="you@example.com"
                  keyboardType="email-address"
                  autoCapitalize="none"
                  placeholderTextColor="#93a3ad"
                  editable={!submitting}
                />
                {errors.email && <ErrorText>{errors.email}</ErrorText>}
              </Field>

              <TwoCol>
                <Col>
                  <Label>Password</Label>
                  <Input value={password} onChangeText={setPassword} secureTextEntry placeholder="At least 8 characters" placeholderTextColor="#93a3ad" editable={!submitting} />
                  {errors.password && <ErrorText>{errors.password}</ErrorText>}
                </Col>
                <Col>
                  <Label>Confirm</Label>
                  <Input value={passwordConfirm} onChangeText={setPasswordConfirm} secureTextEntry placeholder="Repeat password" placeholderTextColor="#93a3ad" editable={!submitting} />
                  {errors.passwordConfirm && <ErrorText>{errors.passwordConfirm}</ErrorText>}
                </Col>
              </TwoCol>

              {errors.form && <FormError>{errors.form}</FormError>}

              <ActionRow>
                <Secondary disabled={submitting} onPress={onClose}><SecondaryText>Cancel</SecondaryText></Secondary>
                <Primary disabled={submitting} onPress={handleSignup}>
                  <PrimaryText>{submitting ? "Signing up..." : "Register"}</PrimaryText>
                </Primary>
              </ActionRow>

              <SigninRow>
                <SmallText>Already have an account?</SmallText>
                <TouchableOpacity disabled={submitting} onPress={onClose}>
                  <LinkText> Sign in</LinkText>
                </TouchableOpacity>
              </SigninRow>
            </Dialog>
          </Container>
        </KeyboardAvoidingView>
      </Overlay>
    </Modal>
  );
}

/* Styled */
const Overlay = styled.View`
  flex: 1;
  background: rgba(4,6,8,0.9);
  justify-content: flex-end;
`;

const Container = styled.ScrollView`
  width: 100%;
`;

const Dialog = styled.View`
  width: 100%;
  background: rgba(255,255,255,0.06);
  padding: 20px;
  border-top-left-radius: 20px;
  border-top-right-radius: 20px;
  border: 1px solid rgba(255,255,255,0.04);
  elevation: 6;
`;

const Close = styled.TouchableOpacity`
  position: absolute;
  right: 10px;
  top: 10px;
  padding: 6px;
`;

const CloseText = styled.Text`
  color: #dbeafe;
  font-size: 20px;
`;

const Title = styled.Text`
  color: #dbeafe;
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 8px;
`;

const Field = styled.View`
  margin-bottom: 10px;
`;

const Label = styled.Text`
  color: #9fb4c2;
  margin-bottom: 6px;
`;

const Input = styled.TextInput`
  padding: 10px;
  border-radius: 8px;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  color: #eef6f9;
`;

const TwoCol = styled.View`
  flex-direction: row;
  gap: 10px;
  margin-bottom: 8px;
`;

const Col = styled.View`
  flex: 1;
`;

const ErrorText = styled.Text`
  color: #b91c1c;
`;

const FormError = styled.Text`
  color: #b91c1c;
  margin-bottom: 8px;
`;

const ActionRow = styled.View`
  flex-direction: row;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 8px;
`;

const Primary = styled.TouchableOpacity<{ disabled?: boolean }>`
  background: rgba(195,200,199,1);
  padding: 10px 14px;
  border-radius: 8px;
  opacity: ${(p: { disabled: any; }) => (p.disabled ? 0.6 : 1)};
`;

const PrimaryText = styled.Text`
  font-weight: 700;
  color: #0b1112;
`;

const Secondary = styled.TouchableOpacity<{ disabled?: boolean }>`
  padding: 8px 12px;
  border-radius: 8px;
  border: 1px solid rgba(6,22,34,0.05);
  opacity: ${(p: { disabled: any; }) => (p.disabled ? 0.6 : 1)};
`;

const SecondaryText = styled.Text`
  color: #dbeafe;
`;

const SigninRow = styled.View`
  margin-top: 10px;
  flex-direction: row;
  justify-content: center;
  align-items: center;
`;

const SmallText = styled.Text`
  color: #C3C8C7;
`;

const LinkText = styled.Text`
  color: #EBE1BD;
  font-weight: 700;
`;