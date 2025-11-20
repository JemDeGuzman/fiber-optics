"use client";
import React, { useState } from "react";
import { useRouter } from "next/navigation";
import styled, { createGlobalStyle } from "styled-components";
import SignupModal from "@/components/SignupModal";
import Image from "next/image";

// Environment variable for API base URL
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

export default function Login(): React.JSX.Element {
  const router = useRouter();
  const [email, setEmail] = useState<string>(""); // backend expects "email"
  const [password, setPassword] = useState<string>("");
  const [showSignup, setShowSignup] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);

  const handleLogin = async (e: React.FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();
    setLoading(true);
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
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <GlobalReset />
      <PageWrapper>
        <LeftPanel aria-hidden>
          <BrandImage role="img" />
          <LeftOverlay>
            <BrandTitle>Fiber Optics</BrandTitle>
            <BrandSubtitle>Abaca Classification Tracking System</BrandSubtitle>
          </LeftOverlay>
        </LeftPanel>

        <RightPanel>
          <Card role="region" aria-labelledby="login-heading">
            <Header>
              <LogoWrapper>
                <Image
                  src="/assets/splash-logo.png"
                  alt="FO"
                  width={128}
                  height={128}
                  priority
                />
              </LogoWrapper>
              <div>
                <Heading id="login-heading">LOG IN</Heading>
                <SubHeading>Inspector access</SubHeading>
              </div>
            </Header>

            <Form onSubmit={handleLogin} noValidate>
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                name="email"
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoComplete="email"
              />

              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                name="password"
                type="password"
                placeholder="At least 8 characters"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="current-password"
              />

              <Actions>
                <div /> {/* spacer to keep primary button right-aligned */}
                <div>
                  <PrimaryButton type="submit" disabled={loading}>
                    {loading ? "Signing in..." : "Sign in"}
                  </PrimaryButton>
                </div>
              </Actions>
            </Form>

            <Footer>
              <span>Don't have an account?</span>
              <LinkButton type="button" onClick={() => setShowSignup(true)}>
                Sign up
              </LinkButton>
            </Footer>
          </Card>
        </RightPanel>

        {showSignup && <SignupModal onClose={() => setShowSignup(false)} />}
      </PageWrapper>
    </>
  );
}

/* ---------------------------
   Global reset and styled-components
   --------------------------- */

const GlobalReset = createGlobalStyle`
  /* Simple reset to remove default body margin and ensure full-height layout */
  html, body, #__next {
    height: 100%;
  }
  body {
    margin: 0;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    background: #0f0f10;
    color: #eef2f5;
    font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
  }
  /* ensure no unexpected scrollbars from small body margins */
  * { box-sizing: border-box; }
`;

const PageWrapper = styled.div`
  height: 100vh; /* fill the viewport exactly */
  display: grid;
  grid-template-columns: 70% 30%; /* left image larger, right card compressed */
  overflow: hidden;
`;

/* Left side with large background image */
const LeftPanel = styled.aside`
  position: relative;
  display: flex;
  align-items: center;
  justify-content: left;
  overflow: hidden;
`;

const BrandImage = styled.div`
  position: absolute;
  inset: 0;
  background-image: url("/assets/fiber-bg.png");
  background-size: cover;
  background-position: center;
  opacity: 0.22;
  filter: grayscale(0.2);
`;

/* overlay text on left */
const LeftOverlay = styled.div`
  position: relative;
  z-index: 2;
  text-align: left;
  padding: 3rem 4rem;
  max-width: 420px;
`;

const BrandTitle = styled.h2`
  margin: 0;
  font-size: 1.8rem;
  letter-spacing: 1px;
  color: #dbeafe;
`;

const BrandSubtitle = styled.p`
  margin: 0.5rem 0 0;
  font-size: 1rem;
  color: #9fb4c2;
`;

/* Right side with compressed centered card */
const RightPanel = styled.main`
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2.5rem;
`;

/* Card */
const Card = styled.section`
  width: 360px; /* slightly narrower to appear more compressed */
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.04);
  box-shadow: 0 8px 24px rgba(2,6,23,0.6);
  border-radius: 12px;
  padding: 26px;
`;

/* Header arranged vertically (logo above headings) */
const Header = styled.div`
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-items: center;
  margin-bottom: 14px;
  text-align: center;
`;

const LogoWrapper = styled.div`
  width: 128px;
  height: 128px;
  border-radius: 24px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
`;

/* Text headings */
const Heading = styled.h1`
  margin: 0;
  font-size: 1.125rem;
  letter-spacing: 0.6px;
  color: #C3C8C7;
`;

const SubHeading = styled.p`
  margin: 4px 0 0;
  font-size: 0.85rem;
  color: #9fb4c2;
`;

/* Form */
const Form = styled.form`
  margin-top: 6px;
  display: flex;
  flex-direction: column;
  gap: 10px;
`;

const Label = styled.label`
  font-size: 0.75rem;
  color: #C3C8C7;
  margin-bottom: 6px;
`;

const Input = styled.input`
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.06);
  background: rgba(255,255,255,0.02);
  color: #eef6f9;
  outline: none;
  font-size: 0.95rem;
  transition: box-shadow 150ms ease, border-color 150ms ease;

  &:focus {
    border-color: #3A4946;
    box-shadow: 0 4px 14px rgba(2,6,23,0.45);
  }

  &::placeholder {
    color: rgba(230,246,245,0.5);
  }
`;

/* Action row */
const Actions = styled.div`
  width: 100%;
  justify-content: center;
  margin-top: 6px;
`;

/* Buttons */
const PrimaryButton = styled.button`
  width: 100%;
  background: linear-gradient(90deg, #C3C8C7, #EBE1BD);
  color: #0b1112;
  border: none;
  padding: 10px 12px;
  border-radius: 10px;
  font-weight: 600;
  cursor: pointer;
  min-width: 120px;
  transition: transform 120ms ease, opacity 120ms ease;

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  &:active {
    transform: translateY(1px);
  }
`;

/* footer area */
const Footer = styled.div`
  margin-top: 18px;
  display: flex;
  gap: 10px;
  align-items: center;
  justify-content: center;
  color: #C3C8C7;
  font-size: 0.95rem;
`;

const LinkButton = styled.button`
  background: none;
  border: none;
  color: #EBE1BD;
  cursor: pointer;
  font-weight: 600;
  padding: 4px 6px;
`;
