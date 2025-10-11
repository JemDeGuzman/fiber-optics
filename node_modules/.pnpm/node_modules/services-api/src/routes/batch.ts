import { Router } from "express";
import { createBatch, getBatchStats, listSamples, exportCsv } from "../controllers/batchController";

const router = Router();

router.post("/", createBatch); // create new batch
router.get("/:id/stats", getBatchStats); // sample count + ratio
router.get("/:id/samples", listSamples); // paginated samples
router.get("/:id/export", exportCsv); // download csv

export default router;