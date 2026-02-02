#!/usr/bin/env node

/**
 * Diagnostic script to inspect network interfaces and IP detection
 */
const os = require("os");

console.log(`║ Network Interface Diagnostic\n`);

const interfaces = os.networkInterfaces();

console.log(`All Network Interfaces:\n`);

let detectedIp = null;
let detectedInterface = null;

Object.keys(interfaces).forEach((name) => {
  const ifaces = interfaces[name];
  console.log(`${name}:`);
  
  ifaces.forEach((iface, idx) => {
    const internal = iface.internal ? " [INTERNAL]" : "";
    const ipv = iface.family === "IPv4" ? " [IPv4]" : " [IPv6]";
    console.log(`  ${idx}: ${iface.address}${ipv}${internal}`);
  });
  console.log();
});

// Run detection logic
console.log(`IP Detection Process:\n`);

// First pass: WiFi/WLAN
console.log(`Checking WiFi/WLAN interfaces first...`);
const wifiPattern = /^wi-?fi|^wlan|^en0|^en1/i;
for (const name of Object.keys(interfaces)) {
  if (wifiPattern.test(name)) {
    console.log(`   ✓ Found: ${name}`);
    for (const iface of interfaces[name]) {
      if (iface.family === "IPv4" && !iface.internal) {
        console.log(`   ✓ Selected: ${iface.address} (IPv4, non-internal)`);
        detectedIp = iface.address;
        detectedInterface = name;
        break;
      }
    }
    if (detectedIp) break;
  }
}

// Second pass: Ethernet
if (!detectedIp) {
  console.log(`\nChecking Ethernet interfaces (excluding virtual)...`);
  const ethernetPattern = /^ethernet|^eth/i;
  for (const name of Object.keys(interfaces)) {
    if (/vethernet|vbox|docker|hyper-v/i.test(name)) {
      console.log(`Skipped: ${name} (virtual adapter)`);
      continue;
    }
    
    if (ethernetPattern.test(name)) {
      console.log(`Found: ${name}`);
      for (const iface of interfaces[name]) {
        if (iface.family === "IPv4" && !iface.internal) {
          console.log(`Selected: ${iface.address} (IPv4, non-internal)`);
          detectedIp = iface.address;
          detectedInterface = name;
          break;
        }
      }
      if (detectedIp) break;
    }
  }
}

// Third pass: fallback
if (!detectedIp) {
  console.log(`\nChecking other non-virtual interfaces...`);
  for (const name of Object.keys(interfaces)) {
    // Skip known virtual interfaces
    if (/docker|veth|vboxnet|vmnet|vlan|vpn|tun|tap|wsl|radmin|teredo|pseudo/i.test(name)) {
      console.log(`Skipped: ${name} (virtual interface)`);
      continue;
    }
    
    for (const iface of interfaces[name]) {
      // Skip virtual network addresses
      if (/^172\.25|^172\.16|^10\.0|^192\.168\.56/.test(iface.address)) {
        console.log(`Skipped: ${iface.address} (virtual network)`);
        continue;
      }
      
      if (iface.family === "IPv4" && !iface.internal) {
        console.log(`Found: ${name}`);
        console.log(`Selected: ${iface.address} (IPv4, non-internal)`);
        detectedIp = iface.address;
        detectedInterface = name;
        break;
      }
    }
    if (detectedIp) break;
  }
}

if (!detectedIp) {
  console.log(`\nNo suitable interface found, using fallback`);
  detectedIp = "localhost";
  detectedInterface = "fallback";
}

console.log(`\n═══════════════════════════════════════════════════\n`);
console.log(`   Final Detection Result:\n`);
console.log(`   Interface: ${detectedInterface}`);
console.log(`   IP Address: ${detectedIp}`);
console.log(`   API URL: http://${detectedIp}:4000\n`);

console.log(`═══════════════════════════════════════════════════\n`);
console.log(`   Network Info:\n`);
console.log(`   Hostname: ${os.hostname()}`);
console.log(`   Platform: ${os.platform()}`);
console.log(`   Type: ${os.type()}\n`);

console.log(`   Network Status:\n`);
if (detectedIp === "localhost") {
  console.log(`     Using localhost - make sure your network adapter is detected`);
} else if (/^26\./.test(detectedIp)) {
  console.log(`     Detected IP starts with 26. - This looks like a VPN IP`);
  console.log(`     Other local devices should use 192.168.x.x range instead`);
} else if (/^192\.168\.56/.test(detectedIp)) {
  console.log(`     Using VirtualBox Host-Only Adapter (192.168.56.x)`);
  console.log(`     This adapter doesn't reach other local devices`);
  console.log(`     Try connecting to WiFi (192.168.1.x) instead`);
} else if (/^172\.25/.test(detectedIp)) {
  console.log(`     Using WSL Virtual Ethernet adapter`);
  console.log(`     This is not accessible from other local devices`);
} else if (/^192\.168|^10\.|^172\.1[6-9]\.|^172\.2[0-9]\.|^172\.3[01]\./.test(detectedIp)) {
  console.log(`     Detected IP is a private LAN address - Good for local network!`);
  console.log(`     Other devices can reach your laptop at: http://${detectedIp}:4000`);
}

console.log();
