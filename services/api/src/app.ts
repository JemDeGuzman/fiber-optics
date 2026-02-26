import express, { Request, Response, NextFunction } from "express";
import cors from "cors";
import path from "path"
import authRoutes from "./routes/auth";
import batchesRoutes from "./routes/batch";
import samplesRoutes from "./routes/samples";
import devicesRoutes from "./routes/devices";

const app = express();

app.use(cors({
  origin: (origin, callback) => {
    // This allows all origins while you debug. 
    // You can replace with your specific frontend URL later.
    callback(null, true);
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  optionsSuccessStatus: 200 // Some legacy browsers/environments need this
}));
app.use(express.json());

// simple health / root route for manual checks
app.get("/", (_req, res) => res.send("API is running!"));

app.use("/uploads", express.static("/app/services/api/uploads"));

// mount API routes
app.use("/api/auth", authRoutes);

app.use("/api/batches", batchesRoutes);

app.use("/api/samples", samplesRoutes);

app.use("/api/devices", devicesRoutes);

const UPLOAD_DIR = path.resolve(__dirname, "../../uploads");
//console.log("STATIC UPLOAD_DIR (express.static) ->", UPLOAD_DIR);
app.use("/uploads", express.static(UPLOAD_DIR));

app.get("/api/ping", (_req, res) => res.json({ ok: true, ts: Date.now() }));

function listApiRoutes(appInstance: express.Application) {
  console.log("==== Registered routes ====");
  appInstance._router.stack.forEach((middleware: any) => {
    if (middleware.route) { // Check if it's a route handler
      const path = middleware.route.path;
      const methods = Object.keys(middleware.route.methods).join(', ').toUpperCase();
      console.log(`- ${methods} ${path}`);
    } else if (middleware.name === 'router' && middleware.handle.stack) { // Check for nested routers
      middleware.handle.stack.forEach((handler: any) => {
        if (handler.route) {
          const path = handler.route.path;
          const methods = Object.keys(handler.route.methods).join(', ').toUpperCase();
          console.log(`- ${methods} ${path}`);
        }
      });
    }
  });
  console.log("===========================");
}

// listApiRoutes(app);

app.get('/health', (_req, res) => res.status(200).json({status: 'ok'}));

app.use((err: any, req: Request, res: Response, next: NextFunction) => {
  console.error("Uncaught Error:", err);
  
  // Explicitly set CORS headers for the error response
  res.header("Access-Control-Allow-Origin", req.headers.origin || "*");
  res.header("Access-Control-Allow-Credentials", "true");

  const status = err.status || 500;
  res.status(status).json({
    error: err.message || "Internal Server Error",
    details: err.errors || null
  });
});

export default app;
