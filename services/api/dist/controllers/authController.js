"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.signup = signup;
exports.login = login;
const bcryptjs_1 = __importDefault(require("bcryptjs"));
const jsonwebtoken_1 = __importDefault(require("jsonwebtoken"));
const client_1 = require("@prisma/client");
const prisma = new client_1.PrismaClient();
const JWT_SECRET = process.env.JWT_SECRET || "change_me";
async function signup(req, res) {
    const { email, password, name } = req.body;
    if (!email || !password)
        return res.status(400).json({ error: "Missing fields" });
    const existing = await prisma.user.findUnique({ where: { email } });
    if (existing)
        return res.status(409).json({ error: "Email in use" });
    const hash = await bcryptjs_1.default.hash(password, 10);
    const user = await prisma.user.create({ data: { email, password: hash, name } });
    const token = jsonwebtoken_1.default.sign({ userId: user.id }, JWT_SECRET, { expiresIn: "7d" });
    res.json({ token, user: { id: user.id, email: user.email, name: user.name } });
}
async function login(req, res) {
    const { email, password } = req.body;
    if (!email || !password)
        return res.status(400).json({ error: "Missing fields" });
    const user = await prisma.user.findUnique({ where: { email } });
    if (!user)
        return res.status(401).json({ error: "Invalid credentials" });
    const match = await bcryptjs_1.default.compare(password, user.password);
    if (!match)
        return res.status(401).json({ error: "Invalid credentials" });
    const token = jsonwebtoken_1.default.sign({ userId: user.id }, JWT_SECRET, { expiresIn: "7d" });
    res.json({ token, user: { id: user.id, email: user.email, name: user.name } });
}
