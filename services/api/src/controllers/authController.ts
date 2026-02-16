// services/api/src/controllers/authController.ts
import { Request, Response, NextFunction} from "express";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import { PrismaClient } from '../generated/client';

const prisma = new PrismaClient();
const JWT_SECRET = process.env.JWT_SECRET || "4x0e3n0o9p7h1i5u0s7!5!7";

type SafeUser = { id: number; email: string; name?: string | null };

// middleware to attach user id onto req if token valid
export interface AuthenticatedRequest extends Request {
  userId?: number;
}

export function authenticateToken(req: AuthenticatedRequest, res: Response, next: NextFunction) {
  const authHeader = (req.headers.authorization || req.headers.Authorization) as string | undefined;
  const token = authHeader && authHeader.startsWith("Bearer ") ? authHeader.slice(7) : null;
  if (!token) return res.status(401).json({ error: "Missing token" });

  try {
    const payload = jwt.verify(token, JWT_SECRET) as any;
    if (!payload || typeof payload.userId !== "number" && typeof payload.userId !== "undefined" && typeof payload.user_id !== "undefined") {
      // try common key names
      req.userId = payload.userId ?? payload.user_id ?? payload.user?.id;
    } else {
      req.userId = payload.userId ?? payload.user_id ?? payload.user?.id;
    }
    // if still falsy, set from `sub` if used
    if (!req.userId && payload.sub) {
      const maybe = Number(payload.sub);
      if (!Number.isNaN(maybe)) req.userId = maybe;
    }
    if (!req.userId) {
      // still no id
      return res.status(401).json({ error: "Invalid token payload" });
    }
    next();
  } catch (err) {
    console.error("Token verify error:", err);
    return res.status(401).json({ error: "Invalid token" });
  }
}

// GET /api/auth/me
export async function getCurrentUser(req: AuthenticatedRequest, res: Response) {
  try {
    const uid = req.userId;
    if (!uid) return res.status(401).json({ error: "Not authenticated" });

    const user = await prisma.user.findUnique({
      where: { id: uid },
      select: { id: true, email: true, name: true, createdAt: true } // select only public fields
    });

    if (!user) return res.status(404).json({ error: "User not found" });
    return res.json({ user });
  } catch (err: any) {
    console.error("getCurrentUser error:", err);
    return res.status(500).json({ error: "Failed to fetch user", detail: err?.message });
  }
}

export async function signup(req: Request, res: Response) {
  try {
    const { email, password, name } = req.body as { email?: string; password?: string; name?: string };
    if (!email || !password) return res.status(400).json({ error: "Missing fields" });

    const existing = await prisma.user.findUnique({ where: { email } });
    if (existing) return res.status(409).json({ error: "Email already registered" });

    const hash = await bcrypt.hash(password, 10);
    const user = await prisma.user.create({
      data: { email, password: hash, name },
      select: { id: true, email: true, name: true },
    });

    const token = jwt.sign({ userId: user.id }, JWT_SECRET, { expiresIn: "7d" });

    res.json({ token, user: user as SafeUser });
  } catch (err) {
    console.error("signup error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
}

export async function login(req: Request, res: Response) {
  try {
    const { email, password } = req.body as { email?: string; password?: string };
    if (!email || !password) return res.status(400).json({ error: "Missing fields" });

    const user = await prisma.user.findUnique({ where: { email } });
    if (!user) return res.status(401).json({ error: "Invalid credentials" });

    const match = await bcrypt.compare(password, user.password);
    if (!match) return res.status(401).json({ error: "Invalid credentials" });

    const token = jwt.sign({ userId: user.id }, JWT_SECRET, { expiresIn: "7d" });
    res.json({ token, user: { id: user.id, email: user.email, name: user.name } as SafeUser });
  } catch (err) {
    console.error("login error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
}
