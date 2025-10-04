import { Request, Response, NextFunction } from "express";
import jwt from "jsonwebtoken";

const JWT_SECRET = process.env.JWT_SECRET || "change_me";

export function requireAuth(req: Request & { userId?: number }, res: Response, next: NextFunction) {
    const auth = req.headers.authorization;
    if (!auth) return res.status(401).json({ error: "Missing auth" });
    const token = auth.split(" ")[1];
    try {
        const payload = jwt.verify(token, JWT_SECRET) as any;
        (req as any).userId = payload.userId;
    next();
    } catch (err) {
        res.status(401).json({ error: "Invalid token" });
    }
}