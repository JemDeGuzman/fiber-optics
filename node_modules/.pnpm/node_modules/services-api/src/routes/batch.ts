import { Router } from "express";
import {
  createBatch,
  getBatchStats,
  listSamples,
  exportCsv,
  createSample, // <-- new
} from "../controllers/batchController";

const router = Router();

router.post("/", createBatch); // create new batch
router.get("/:id/stats", getBatchStats); // sample count + ratio
router.get("/:id/samples", listSamples); // paginated samples
router.post("/:id/samples", createSample); // <-- new: create sample in batch
router.get("/:id/export", exportCsv); // download csv

export default router;
