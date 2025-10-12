import { Router } from "express";
import {
  createBatch,
  getBatchStats,
  listSamples,
  exportCsv,
  createSample,
  listBatches,
  updateBatch,
  deleteBatch // <-- new
} from "../controllers/batchController";

const router = Router();

router.get("/", listBatches); // show all batches
router.post("/", createBatch); // create new batch
router.patch("/:id", updateBatch); // update batch name
router.delete("/:id", deleteBatch); // delete batch
router.get("/:id/stats", getBatchStats); // sample count + ratio
router.get("/:id/samples", listSamples); // paginated samples
router.post("/:id/samples", createSample); // create sample in batch
router.get("/:id/export", exportCsv);    // export all or page via query params
router.post("/:id/export", exportCsv);   // export selected ids via POST { ids: [..] }

export default router;
