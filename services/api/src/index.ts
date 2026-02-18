import dotenv from "dotenv";
import { getLocalIp, getApiUrl } from "./utils/ipDetection";

dotenv.config();

import app from "./app";

const PORT = parseInt(process.env.PORT || '8080', 10);
// In production, we care about the Railway Public URL, not the local IP
const PUBLIC_URL = process.env.RAILWAY_PUBLIC_DOMAIN || `http://localhost:${PORT}`;

app.listen(Number(PORT), '0.0.0.0', () => {
  console.log(`Server is definitely listening on 0.0.0.0:${PORT}`);
});

/*
app.listen(PORT, '0.0.0.0', () => {
  console.log(`\n╔════════════════════════════════════════════════════╗`);
  console.log(`║ API Server Started                                 ║`);
  console.log(`║ Public URL: ${PUBLIC_URL.padEnd(38)} ║`);
  console.log(`║ Internal Port: ${String(PORT).padEnd(35)} ║`);
  console.log(`╚════════════════════════════════════════════════════╝\n`);
});
*/