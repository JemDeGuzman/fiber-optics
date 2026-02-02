import os from "os";

export interface IPDetectionResult {
  ip: string;
  port: number;
  isLocalhost: boolean;
}

/**
 * Get the local machine's IP address (IPv4)
 * Prioritizes common physical network interfaces (WiFi, Ethernet) over virtual ones (VPN, Docker, WSL)
 * Falls back to 'localhost' if no suitable IP is found
 */
export function getLocalIp(): string {
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
  
  // Final fallback to localhost
  return "localhost";
}

/**
 * Get the full API URL based on detected local IP
 */
export function getApiUrl(port: number = 4000): string {
  const ip = getLocalIp();
  return `http://${ip}:${port}`;
}

/**
 * Get detection result with all details
 */
export function detectIpConfig(port: number = 4000): IPDetectionResult {
  const ip = getLocalIp();
  return {
    ip,
    port,
    isLocalhost: ip === "localhost",
  };
}
