import { Request, Response } from "express";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
// import { PrismaClient } from "@prisma/client"; // keep commented if prisma client not ready

// const prisma = new PrismaClient();
const JWT_SECRET = process.env.JWT_SECRET || "change_me";

export async function signup(req: Request, res: Response) {
  const { email, password, name } = req.body;
  if (!email || !password) return res.status(400).json({ error: "Missing fields" });

  // if prisma ready, check existing user here
  // const existing = await prisma.user.findUnique({ where: { email } });

  // For now, just hash and return a fake user (dev stub)
  const hash = await bcrypt.hash(password, 10);
  const user = { id: Math.floor(Math.random() * 1000000), email, name };

  const token = jwt.sign({ userId: user.id }, JWT_SECRET, { expiresIn: "7d" });
  res.json({ token, user: { id: user.id, email: user.email, name: user.name } });
}

export async function login(req: Request, res: Response) {
  const { email, password } = req.body;
  if (!email || !password) return res.status(400).json({ error: "Missing fields" });

  // If Prisma is available, validate user & password against DB.
  // This stub accepts any credentials for dev flow:
  const fakeUser = { id: 1, email, name: "Dev User" };
  const token = jwt.sign({ userId: fakeUser.id }, JWT_SECRET, { expiresIn: "7d" });

  res.json({ token, user: fakeUser });
}
