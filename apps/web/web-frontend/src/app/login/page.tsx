"use client";
import React, { useState } from "react";
import { useRouter } from "next/navigation";
import SignupModal from "@/components/SignupModal";

// ✅ Use environment variable for API base URL
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

export default function Login(): React.JSX.Element {
  const router = useRouter();
  const [email, setEmail] = useState<string>(""); // backend expects "email", not "username"
  const [password, setPassword] = useState<string>("");
  const [showSignup, setShowSignup] = useState<boolean>(false);

  // ✅ Make sure we send "email" and "password" to backend
  const handleLogin = async (e: React.FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();
    try {
      const res = await fetch(`${API_URL}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      const data: { token?: string; error?: string } = await res.json();

      if (res.ok && data.token) {
        localStorage.setItem("token", data.token);
        router.push("/dashboard");
      } else {
        alert("Login failed: " + (data.error || "Unknown error"));
      }
    } catch (err) {
      console.error("Login error:", err);
      alert("Unable to connect to server. Please check API status.");
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Login</h1>
      <form onSubmit={handleLogin}>
        <input
          placeholder="Email"
          value={email}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEmail(e.target.value)}
        />
        <br />
        <input
          placeholder="Password"
          type="password"
          value={password}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPassword(e.target.value)}
        />
        <br />
        <button type="submit">Login</button>
        <button type="button" onClick={() => setShowSignup(true)}>
          Signup
        </button>
      </form>

      {showSignup && <SignupModal onClose={() => setShowSignup(false)} />}
    </div>
  );
}