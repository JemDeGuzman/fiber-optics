#!/usr/bin/env node

/**
 * Setup script for web frontend - detects local IP and updates .env.local
 */
const fs = require("fs");
const os = require("os");
const path = require("path");

function getLocalIp() {
  const interfaces = os.networkInterfaces();
  
  // Priority 1: WiFi/WLAN (most common for local development)
  const wifiPattern = /^wi-?fi|^wlan|^en0|^en1/i;
  for (const name of Object.keys(interfaces)) {
    if (wifiPattern.test(name)) {
      for (const iface of interfaces[name]) {
        if (iface.family === "IPv4" && !iface.internal) {
          return iface.address;
        }
      }
    }
  }
  
  // Priority 2: Other real ethernet interfaces (but skip obvious virtual ones)
  const ethernetPattern = /^ethernet|^eth/i;
  for (const name of Object.keys(interfaces)) {
    // Skip virtual ethernet adapters (VirtualBox, Hyper-V, Docker)
    if (/vethernet|vbox|docker|hyper-v/i.test(name)) {
      continue;
    }
    
    if (ethernetPattern.test(name)) {
      for (const iface of interfaces[name]) {
        if (iface.family === "IPv4" && !iface.internal) {
          return iface.address;
        }
      }
    }
  }
  
  // Priority 3: Any other non-virtual, non-internal IPv4
  for (const name of Object.keys(interfaces)) {
    // Skip known virtual interfaces
    if (/docker|veth|vboxnet|vmnet|vlan|vpn|tun|tap|wsl|radmin|teredo|pseudo/i.test(name)) {
      continue;
    }
    
    // Also skip addresses that look like virtual networks (VBox, WSL, etc)
    for (const iface of interfaces[name]) {
      if (iface.family === "IPv4" && !iface.internal && !/^172\.25|^172\.16|^10\.0|^192\.168\.56/.test(iface.address)) {
        return iface.address;
      }
    }
  }
  
  return "localhost";
}

const ip = getLocalIp();
const port = 4000;
const apiUrl = `http://${ip}:${port}`;
const envContent = `NEXT_PUBLIC_API_URL=${apiUrl}\n`;

const envPath = path.join(__dirname, ".env.local");

try {
  fs.writeFileSync(envPath, envContent);
  console.log(`✓ Web frontend .env.local updated`);
  console.log(`  API URL: ${apiUrl}`);
} catch (err) {
  console.error(`✗ Failed to update web .env.local file:`, err.message);
  process.exit(1);
}
