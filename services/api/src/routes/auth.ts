import { Router } from "express";
import { signup, login , getCurrentUser, authenticateToken } from "../controllers/authController";
import { validateBody } from "../middleware/validate";
import { signupSchema, loginSchema } from "../schemas/auth";

const router = Router();
router.post("/signup", validateBody(signupSchema), signup);
router.post("/login", validateBody(loginSchema), login);
router.get("/me", authenticateToken, getCurrentUser);

export default router;
