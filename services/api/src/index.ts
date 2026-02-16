import dotenv from "dotenv";
import { getLocalIp, getApiUrl } from "./utils/ipDetection";

dotenv.config();

import app from "./app";

const PORT = parseInt(process.env.PORT || process.env.API_PORT || '4000', 10);
const HOST = getLocalIp();
const API_URL = getApiUrl(PORT as number);

app.listen(PORT, '0.0.0.0', () => {
  console.log(`\n╔════════════════════════════════════════════════════╗`);
  console.log(`║ API Server Started                                 ║`);
  console.log(`║ URL: ${API_URL.padEnd(49 - "║ URL: ".length)}║`);
  console.log(`║ Port: ${String(PORT).padEnd(48 - "║ Port: ".length)}║`);
  console.log(`║ Host: ${HOST.padEnd(48 - "║ Host: ".length)}║`);
  console.log(`╚════════════════════════════════════════════════════╝\n`);
});
