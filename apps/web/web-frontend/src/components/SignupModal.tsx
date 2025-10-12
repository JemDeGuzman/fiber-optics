"use client";
import React, { useState } from "react";
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

  const handleSignup = async (): Promise<void> => {
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
        // keep first error per field
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
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        background: "rgba(0,0,0,0.5)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 9999,
      }}
    >
      <div
        style={{
          background: "white",
          padding: 20,
          width: 380,
          borderRadius: 8,
          boxShadow: "0 6px 18px rgba(0,0,0,0.12)",
        }}
      >
        <h2 style={{ marginTop: 0 }}>Signup</h2>

        {/* Name */}
        <label style={{ display: "block", marginBottom: 8 }}>
          Name
          <input
            value={name}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setName(e.target.value)
            }
            style={{ width: "90%", padding: 8, marginTop: 6 }}
            placeholder="Lebron"
            disabled={submitting}
          />
        </label>
        {errors.name && (
          <div style={{ color: "crimson", marginBottom: 8 }}>{errors.name}</div>
        )}

        {/* Email */}
        <label style={{ display: "block", marginBottom: 8 }}>
          Email
          <input
            value={email}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setEmail(e.target.value)
            }
            style={{ width: "90%", padding: 8, marginTop: 6 }}
            placeholder="you@example.com"
            disabled={submitting}
          />
        </label>
        {errors.email && (
          <div style={{ color: "crimson", marginBottom: 8 }}>{errors.email}</div>
        )}

        {/* Password */}
        <label style={{ display: "block", marginBottom: 8 }}>
          Password
          <input
            type="password"
            value={password}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setPassword(e.target.value)
            }
            style={{ width: "90%", padding: 8, marginTop: 6 }}
            placeholder="at least 8 characters"
            disabled={submitting}
          />
        </label>
        {errors.password && (
          <div style={{ color: "crimson", marginBottom: 8 }}>
            {errors.password}
          </div>
        )}

        {/* Confirm Password */}
        <label style={{ display: "block", marginBottom: 8 }}>
          Confirm Password
          <input
            type="password"
            value={passwordConfirm}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setPasswordConfirm(e.target.value)
            }
            style={{ width: "90%", padding: 8, marginTop: 6 }}
            placeholder="repeat password"
            disabled={submitting}
          />
        </label>
        {errors.passwordConfirm && (
          <div style={{ color: "crimson", marginBottom: 8 }}>
            {errors.passwordConfirm}
          </div>
        )}

        {errors.form && (
          <div style={{ color: "crimson", marginTop: 8 }}>{errors.form}</div>
        )}

        <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
          <button
            onClick={handleSignup}
            disabled={submitting}
            style={{ padding: "8px 12px" }}
          >
            {submitting ? "Signing up..." : "Register"}
          </button>
          <button
            onClick={onClose}
            disabled={submitting}
            style={{ padding: "8px 12px" }}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}