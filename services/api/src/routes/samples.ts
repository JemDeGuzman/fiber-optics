// services/api/src/routes/samples.ts
import { Router, Request, Response } from "express";
import multer from "multer";
import path from "path";
import fs from "fs";
import { PrismaClient } from "@prisma/client";
import { updateSample, deleteSample, deleteManySamples } from "../controllers/batchController";

const router = Router();
const prisma = new PrismaClient();

type MulterRequest<TFile = Express.Multer.File> = Omit<Request, "file"> & {
  file?: TFile;
};
/* ------------------------ Upload config ------------------------ */
const UPLOAD_DIR = path.resolve(__dirname, "../../../uploads");
//console.log("MULTER UPLOAD_DIR ->", UPLOAD_DIR);
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname) || ".jpg";
    const basename = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    cb(null, basename + ext);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB
  fileFilter: (_req, file, cb) => {
    if (/^image\/(jpeg|jpg|png)$/.test(file.mimetype)) cb(null, true);
    else cb(new Error("Only JPEG/PNG images are allowed"));
  },
});

/**
 * POST /api/samples/upload
 * multipart/form-data:
 *  - file (required)
 *  - batchId (required)
 *  - classification, luster_value, roughness, tensile_strength (optional)
 *  - capturedAt (optional)
 */
router.post("/upload", upload.single("file"), async (req: Request, res: Response) => {
  const r = req as MulterRequest;

  try {
    // safe access to file
    const uploadedFile = r.file;
    if (!uploadedFile) {
      return res.status(400).json({ error: "No file uploaded (field name must be 'file')" });
    }

    const {
      batchId,
      classification,
      luster_value,
      roughness,
      tensile_strength,
      capturedAt,
    } = req.body;

    if (!batchId) {
      // cleanup if file exists
      try {
        if (uploadedFile && uploadedFile.path && fs.existsSync(uploadedFile.path)) {
          fs.unlinkSync(uploadedFile.path);
        }
      } catch (e) {}
      return res.status(400).json({ error: "batchId is required" });
    }

    // verify batch exists:
    const batch = await prisma.dataBatch.findUnique({ where: { id: Number(batchId) } });
    if (!batch) {
      try {
        if (uploadedFile && uploadedFile.path && fs.existsSync(uploadedFile.path)) {
          fs.unlinkSync(uploadedFile.path);
        }
      } catch (e) {}
      return res.status(400).json({ error: "batchId not found" });
    }

    // Build a URL for image_capture that clients can fetch
    const protocol = req.protocol;
    const host = req.get("host"); // e.g. localhost:4000
    const imageUrl = `${protocol}://${host}/uploads/${uploadedFile.filename}`;

    // Create DataSample record
    const sample = await prisma.dataSample.create({
      data: {
        batchId: Number(batchId),
        image_capture: imageUrl,
        classification: classification ?? "unknown",
        luster_value: luster_value ? parseFloat(luster_value) : undefined,
        roughness: roughness ? parseFloat(roughness) : undefined,
        tensile_strength: tensile_strength ? parseFloat(tensile_strength) : undefined,
        createdAt: capturedAt ? new Date(capturedAt) : undefined,
      },
    });

    return res.json({ ok: true, sample });
  } catch (err: any) {
    console.error("samples.upload error:", err);

    // attempt to cleanup uploaded file on error
    try {
      const uploadedFile = (req as MulterRequest).file;
      if (uploadedFile && uploadedFile.path && fs.existsSync(uploadedFile.path)) {
        fs.unlinkSync(uploadedFile.path);
      }
    } catch (cleanupErr) {
      console.warn("cleanup error:", cleanupErr);
    }

    return res.status(500).json({ error: "Upload failed", details: String(err.message ?? err) });
  }
});

/* ---------------------- existing sample routes ---------------------- */
// PATCH /api/samples/:id
router.patch("/:id", updateSample);

// DELETE /api/samples/:id
router.delete("/:id", deleteSample);

// DELETE Many /api/samples/deleteMany
router.post("/deleteMany", deleteManySamples);

export default router;
