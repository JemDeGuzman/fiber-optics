// services/api/src/middleware/validate.ts
import { ZodObject, ZodError } from "zod";
import { Request, Response, NextFunction } from "express";

export const validateBody = (schema: ZodObject<any>) => (req: Request, res: Response, next: NextFunction) => {
  try {
    req.body = schema.parse(req.body);
    next();
  } catch (err) {
    const zErr = err as ZodError;
    
    // FORCE CORS headers here so the browser lets us see the 400 error
    res.header("Access-Control-Allow-Origin", req.headers.origin || "*");
    res.header("Access-Control-Allow-Credentials", "true");

    return res.status(400).json({ 
      error: "Validation failed", 
      details: zErr.issues // Simplified for debugging
    });
  }
};