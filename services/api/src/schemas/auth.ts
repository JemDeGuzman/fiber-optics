import { z } from "zod";

export const signupSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8, "Password must be at least 8 characters"),
  passwordConfirm: z.string(),
  name: z.string().optional().nullable(),
}).refine((data) => data.password === data.passwordConfirm, {
  message: "Passwords do not match",
  path: ["passwordConfirm"],
});

export const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(1),
});
