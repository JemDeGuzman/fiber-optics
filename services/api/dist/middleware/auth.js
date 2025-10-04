"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.requireAuth = requireAuth;
const jsonwebtoken_1 = __importDefault(require("jsonwebtoken"));
const JWT_SECRET = process.env.JWT_SECRET || "change_me";
function requireAuth(req, res, next) {
    const auth = req.headers.authorization;
    if (!auth)
        return res.status(401).json({ error: "Missing auth" });
    const token = auth.split(" ")[1];
    try {
        const payload = jsonwebtoken_1.default.verify(token, JWT_SECRET);
        req.userId = payload.userId;
        next();
    }
    catch (err) {
        res.status(401).json({ error: "Invalid token" });
    }
}
