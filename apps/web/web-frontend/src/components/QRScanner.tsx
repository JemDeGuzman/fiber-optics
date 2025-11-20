// src/components/QRScanner.tsx
"use client";
import React, { useEffect, useRef, useState } from "react";
import { Html5Qrcode } from "html5-qrcode";

interface Props {
  captureServerBase?: string; // e.g. "http://192.168.1.123:3001" or "http://localhost:3001"
  onConnected?: (resp: any) => void;
  onError?: (err: any) => void;
}

export default function QRScanner({ captureServerBase = "http://localhost:3001", onConnected, onError }: Props) {
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const divRef = useRef<HTMLDivElement | null>(null);
  const html5Ref = useRef<Html5Qrcode | null>(null);

  useEffect(() => {
    return () => { stopScanner(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const startScanner = async () => {
    if (!divRef.current) {
      setMessage("No preview element");
      return;
    }
    try {
      const elementId = divRef.current.id || `qr_${Date.now()}`;
      divRef.current.id = elementId;
      const html5Qr = new Html5Qrcode(elementId, { verbose: false });
      html5Ref.current = html5Qr;

      const qrConfig = { fps: 10, qrbox: { width: 300, height: 300 }};

      await html5Qr.start(
        { facingMode: "environment" },
        qrConfig,
        async (decoded: string) => {
          // successful decode
          try {
            setMessage("Scanned, processing...");
            html5Qr.pause(true);
            setRunning(false);

            // parse payload
            let payload;
            try { payload = JSON.parse(decoded); } catch (e) {
              setMessage("Invalid QR: not JSON");
              html5Qr.resume();
              setRunning(true);
              return;
            }

            // POST to Pi capture-server /connect
            const url = `${captureServerBase.replace(/\/$/, "")}/connect`;
            const resp = await fetch(url, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload),
            });

            const contentType = resp.headers.get("content-type") || "";
            const body = contentType.includes("application/json") ? await resp.json() : await resp.text();

            if (!resp.ok) {
              const errText = typeof body === "string" ? body : JSON.stringify(body);
              setMessage("Connect failed: " + errText);
              onError?.(body);
              html5Qr.resume();
              setRunning(true);
              return;
            }

            setMessage("Connected successfully");
            onConnected?.(body);
            // optionally stop scanner permanently
            await html5Qr.stop();
            html5Ref.current = null;
          } catch (err) {
            console.error("scanner handler error", err);
            setMessage("Scanner processing error");
            onError?.(err);
            try { await html5Ref.current?.resume(); } catch (e) {}
            setRunning(true);
          }
        },
        (errorMessage) => {
          // decode failed for a frame; ignore or show small status
          // setMessage(errorMessage);
        }
      );
      setRunning(true);
      setMessage("Point your camera at the Pi screen QR");
    } catch (err) {
      console.error("startScanner error", err);
      setMessage("Could not start camera: " + String(err));
      onError?.(err);
    }
  };

  const stopScanner = async () => {
    try {
      if (html5Ref.current) {
        await html5Ref.current.stop();
        html5Ref.current.clear();
        html5Ref.current = null;
      }
    } catch (err) {
      console.warn("stop scanner error", err);
    }
    setRunning(false);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div ref={divRef} style={{ width: 340, height: 340, borderRadius: 8, overflow: "hidden", background: "#111" }} />
      <div style={{ display: "flex", gap: 8 }}>
        {!running ? (
          <button onClick={startScanner} style={{ background: "#3A4946", color: "#EBE1BD", padding: "8px 12px", borderRadius: 6 }}>
            Start Scanner
          </button>
        ) : (
          <button onClick={stopScanner} style={{ background: "#8b3a3a", color: "#EBE1BD", padding: "8px 12px", borderRadius: 6 }}>
            Stop Scanner
          </button>
        )}
        <div style={{ color: "#C3C8C7", alignSelf: "center" }}>{message}</div>
      </div>
    </div>
  );
}