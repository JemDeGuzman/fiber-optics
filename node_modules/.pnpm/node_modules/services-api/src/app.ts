import express from "express";
import cors from "cors";
import authRoutes from "./routes/auth";

const app = express();

app.use(cors());
app.use(express.json());

// simple health / root route for manual checks
app.get("/", (_req, res) => res.send("API is running!"));

// mount API routes
app.use("/api/auth", authRoutes);
app.get("/api/ping", (_req, res) => res.json({ ok: true, ts: Date.now() }));

export default app;
