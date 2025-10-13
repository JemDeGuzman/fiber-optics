"use client";
import React, { useState, useRef, useEffect } from "react";
import styled from "styled-components";
import { z } from "zod";

interface SignupModalProps {
  onClose: () => void;
}

const signupSchema = z
  .object({
    name: z.string().min(2, "Name must be at least 2 characters"),
    email: z.string().email("Invalid email address"),
    password: z
      .string()
      .min(8, "Password must be at least 8 characters")
      .max(128),
    passwordConfirm: z.string(),
  })
  .refine((data) => data.password === data.passwordConfirm, {
    message: "Passwords do not match",
    path: ["passwordConfirm"],
  });

export default function SignupModal({ onClose }: SignupModalProps) {
  const [name, setName] = useState<string>("");
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [passwordConfirm, setPasswordConfirm] = useState<string>("");

  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState<boolean>(false);

  const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:4000";

  const nameRef = useRef<HTMLInputElement | null>(null);
  const dialogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // focus first input when modal opens
    nameRef.current?.focus();

    // simple escape-to-close
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  // optional: trap focus inside modal (simple)
  useEffect(() => {
    const el = dialogRef.current;
    if (!el) return;
    const focusable = el.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])'
    );
    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    const handleTab = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;
      if (!first || !last) return;
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleTab);
    return () => document.removeEventListener("keydown", handleTab);
  }, []);

  const handleSignup = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setErrors({});
    // Validate with zod
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
      const res = await fetch(`${API_URL}/api/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email,
          password,
          passwordConfirm,
          name,
        }),
      });

      const data = await res.json();

      if (res.ok) {
        alert("Signup successful! You can now log in.");
        onClose();
      } else {
        // backend error
        const backendError = data?.error ?? JSON.stringify(data);
        setErrors({ form: backendError });
      }
    } catch (err) {
      console.error("Signup error:", err);
      setErrors({ form: "Unable to contact server. Please try again." });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Overlay role="presentation" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <Dialog
        role="dialog"
        aria-modal="true"
        aria-labelledby="signup-title"
        ref={dialogRef}
      >
        <CloseButton
          aria-label="Close signup dialog"
          onClick={onClose}
          type="button"
        >
          ×
        </CloseButton>

        <Title id="signup-title">Create an account</Title>

        <Form onSubmit={handleSignup} noValidate>
          <Field>
            <Label htmlFor="signup-name">Name</Label>
            <Input
              id="signup-name"
              ref={nameRef}
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Lebron"
              disabled={submitting}
              aria-invalid={!!errors.name}
              aria-describedby={errors.name ? "err-name" : undefined}
            />
            {errors.name && <Error id="err-name">{errors.name}</Error>}
          </Field>

          <Field>
            <Label htmlFor="signup-email">Email</Label>
            <Input
              id="signup-email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              disabled={submitting}
              type="email"
              aria-invalid={!!errors.email}
              aria-describedby={errors.email ? "err-email" : undefined}
            />
            {errors.email && <Error id="err-email">{errors.email}</Error>}
          </Field>

          <TwoColumn>
            <Col>
              <Label htmlFor="signup-password">Password</Label>
              <Input
                id="signup-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="At least 8 characters"
                disabled={submitting}
                type="password"
                aria-invalid={!!errors.password}
                aria-describedby={errors.password ? "err-pass" : undefined}
              />
              {errors.password && <Error id="err-pass">{errors.password}</Error>}
            </Col>

            <Col>
              <Label htmlFor="signup-password-confirm">Confirm</Label>
              <Input
                id="signup-password-confirm"
                value={passwordConfirm}
                onChange={(e) => setPasswordConfirm(e.target.value)}
                placeholder="Repeat password"
                disabled={submitting}
                type="password"
                aria-invalid={!!errors.passwordConfirm}
                aria-describedby={errors.passwordConfirm ? "err-passconf" : undefined}
              />
              {errors.passwordConfirm && (
                <Error id="err-passconf">{errors.passwordConfirm}</Error>
              )}
            </Col>
          </TwoColumn>

          {errors.form && <FormError role="alert">{errors.form}</FormError>}

          <Actions>
            <Secondary type="button" onClick={onClose} disabled={submitting}>
              Cancel
            </Secondary>
            <Primary type="submit" disabled={submitting}>
              {submitting ? "Signing up..." : "Register"}
            </Primary>
          </Actions>

          <SigninRow>
            <SmallText>Already have an account?</SmallText>
            <LinkButton
              type="button"
              onClick={onClose}
              disabled={submitting}
            >
              Sign in
            </LinkButton>
          </SigninRow>
        </Form>
      </Dialog>
    </Overlay>
  );
}

/* ===========================
   Styled components
   =========================== */

const Overlay = styled.div`
  position: fixed;
  inset: 0;
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(4,6,8,0.6);
  //-webkit-backdrop-filter: blur(3px);
  //backdrop-filter: blur(3px);
  padding: 24px;
`;

const Dialog = styled.div`
  width: 100%;
  max-width: 520px;
  background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.06));
  color: #dbeafe;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(2,6,23,0.6);
  padding: 20px 20px 18px;
  position: relative;
  border: 1px solid rgba(255,255,255,0.04);
`;

const CloseButton = styled.button`
  position: absolute;
  right: 10px;
  top: 10px;
  border: none;
  background: transparent;
  font-size: 20px;
  line-height: 1;
  cursor: pointer;
  color: #dbeafe;
  padding: 6px;
  border-radius: 6px;

  &:hover { background: rgba(2,6,23,0.03); }
`;

const Title = styled.h2`
  margin: 0 0 8px 0;
  font-size: 1.125rem;
  color: #dbeafe;
`;

const Form = styled.form`
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const Field = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

const TwoColumn = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
`;

const Col = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

const Label = styled.label`
  font-size: 0.8rem;
  color: #9fb4c2;
`;

const Input = styled.input`
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.06);
  background: rgba(255,255,255,0.02);
  color: #eef6f9;
  outline: none;
  font-size: 0.95rem;

  &:focus {
    border-color: #3A4946;
    box-shadow: 0 4px 14px rgba(2,6,23,0.45);
  }

  &::placeholder {
    color: #93a3ad;
  }

  &:disabled {
    opacity: 0.7;
  }
`;

const Error = styled.div`
  color: #b91c1c;
  font-size: 0.825rem;
`;

const FormError = styled.div`
  color: #b91c1c;
  font-size: 0.9rem;
  margin-top: 6px;
`;

const Actions = styled.div`
  display: flex;
  gap: 10px;
  justify-content: flex-end;
  margin-top: 4px;
`;

const Primary = styled.button`
  background: linear-gradient(90deg, #C3C8C7, #EBE1BD);
  color: #0b1112;
  border: none;
  padding: 10px 14px;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  min-width: 110px;

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const Secondary = styled.button`
  background: transparent;
  color: #dbeafe;
  border: 1px solid rgba(6, 22, 34, 0.05);
  padding: 8px 12px;
  border-radius: 8px;
  cursor: pointer;

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const SigninRow = styled.div`
  margin-top: 10px;
  display: flex;
  gap: 8px;
  align-items: center;
  justify-content: center;
`;

const SmallText = styled.span`
  font-size: 0.85rem;
  color: #C3C8C7;
`;

const LinkButton = styled.button`
  background: none;
  border: none;
  color: #EBE1BD;
  font-weight: 600;
  cursor: pointer;
  padding: 4px 6px;
`;
