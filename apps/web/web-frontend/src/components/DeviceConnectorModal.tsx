"use client";
import React, { useState } from "react";
import styled from "styled-components";

interface DeviceConnectorModalProps {
  currentDeviceIp: string;
  onConnect: (remoteDeviceIp: string) => Promise<boolean>;
  onClose: () => void;
}

const ModalOverlay = styled.div`
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
`;

const ModalContent = styled.div`
  background: #1f1f1f;
  padding: 24px;
  border-radius: 12px;
  color: #EBE1BD;
  width: 90%;
  max-width: 500px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
`;

const Title = styled.h3`
  margin-bottom: 16px;
  font-size: 20px;
`;

const Section = styled.div`
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid #3A4946;

  &:last-child {
    border-bottom: none;
  }
`;

const Label = styled.label`
  display: block;
  margin-bottom: 8px;
  font-size: 14px;
  color: #C3C8C7;
`;

const Input = styled.input`
  width: 100%;
  padding: 10px 12px;
  background-color: #262626;
  color: #EBE1BD;
  border: 1px solid #3A4946;
  border-radius: 6px;
  font-size: 14px;

  &:focus {
    outline: none;
    border-color: #5A6B68;
    box-shadow: 0 0 0 2px rgba(90, 107, 104, 0.2);
  }
`;

const DeviceInfo = styled.div`
  background: #262626;
  padding: 12px;
  border-radius: 6px;
  font-size: 13px;
  color: #C3C8C7;
  word-break: break-all;
`;

const Button = styled.button`
  background-color: #3A4946;
  color: #EBE1BD;
  border: none;
  padding: 10px 16px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.2s;

  &:hover {
    background-color: #4A5A56;
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 8px;
  justify-content: flex-end;
`;

const Message = styled.div<{ type?: "success" | "error" | "info" }>`
  padding: 10px 12px;
  border-radius: 6px;
  font-size: 13px;
  margin-bottom: 12px;
  background-color: ${(props) => {
    switch (props.type) {
      case "success":
        return "#1e5631";
      case "error":
        return "#5d2a2a";
      case "info":
      default:
        return "#2a3f3a";
    }
  }};
  color: ${(props) => {
    switch (props.type) {
      case "success":
        return "#4ade80";
      case "error":
        return "#ef4444";
      case "info":
      default:
        return "#60a5fa";
    }
  }};
`;

export default function DeviceConnectorModal({
  currentDeviceIp,
  onConnect,
  onClose,
}: DeviceConnectorModalProps) {
  const [remoteIp, setRemoteIp] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{
    text: string;
    type: "success" | "error" | "info";
  } | null>(null);

  const handleConnect = async () => {
    if (!remoteIp.trim()) {
      setMessage({ text: "Please enter a device IP address", type: "error" });
      return;
    }

    setLoading(true);
    setMessage({ text: "Connecting to device...", type: "info" });

    try {
      const success = await onConnect(remoteIp);

      if (success) {
        setMessage({
          text: "Connected successfully!",
          type: "success",
        });
        setTimeout(() => {
          onClose();
        }, 1500);
      } else {
        setMessage({
          text: "Connection failed. Check the IP and try again.",
          type: "error",
        });
      }
    } catch (err) {
      setMessage({
        text: `Error: ${String(err)}`,
        type: "error",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !loading) {
      handleConnect();
    }
  };

  return (
    <ModalOverlay onClick={onClose}>
      <ModalContent onClick={(e) => e.stopPropagation()}>
        <Title>🔗 Connect to Device</Title>

        {message && (
          <Message type={message.type}>{message.text}</Message>
        )}

        <Section>
          <Label>This Device IP Address:</Label>
          <DeviceInfo>{currentDeviceIp}</DeviceInfo>
          <p style={{ fontSize: "12px", marginTop: 8, color: "#A0A59F" }}>
            Share this IP with the other device. They will enter it on their server.
          </p>
        </Section>

        <Section>
          <Label htmlFor="remoteIp">Enter Remote Device IP Address:</Label>
          <Input
            id="remoteIp"
            type="text"
            placeholder="e.g., 192.168.1.100"
            value={remoteIp}
            onChange={(e) => setRemoteIp(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={loading}
          />
          <p style={{ fontSize: "12px", marginTop: 8, color: "#A0A59F" }}>
            Enter the other device's IP (format: xxx.xxx.xxx.xxx). Both devices must register each other.
          </p>
        </Section>

        <ButtonGroup>
          <Button onClick={onClose} disabled={loading}>
            Cancel
          </Button>
          <Button onClick={handleConnect} disabled={loading}>
            {loading ? "Connecting..." : "Connect"}
          </Button>
        </ButtonGroup>
      </ModalContent>
    </ModalOverlay>
  );
}
