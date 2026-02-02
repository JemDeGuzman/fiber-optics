const fs = require('fs');
const os = require('os');

function getLocalIp() {
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name]) {
      // Look for IPv4 and skip internal/loopback addresses
      if (iface.family === 'IPv4' && !iface.internal) {
        return iface.address;
      }
    }
  }
  return 'localhost';
}

const ip = getLocalIp();
const port = 4000; // Your backend port
const envContent = `EXPO_PUBLIC_API_URL=http://${ip}:${port}\n`;

try {
  fs.writeFileSync('.env', envContent);
  console.log(`Success: .env updated with local IP: ${ip}`);
} catch (err) {
  console.error('Failed to update .env file', err);
}