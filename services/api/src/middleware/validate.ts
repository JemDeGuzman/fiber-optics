import { ZodObject, ZodError, z } from "zod";
import { Request, Response, NextFunction } from "express";

export const validateBody = (schema: ZodObject<any>) => (req: Request, res: Response, next: NextFunction) => {
  try {
    req.body = schema.parse(req.body);
    next();
  } catch (err) {
    const zErr = err as ZodError;
    return res.status(400).json({ error: "Validation failed", details: z.treeifyError(zErr) });
  }
};
