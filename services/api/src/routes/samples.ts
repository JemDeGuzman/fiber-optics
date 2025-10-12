import { Router } from "express";
import { updateSample, deleteSample, deleteManySamples } from "../controllers/batchController"; // or samplesController

const router = Router();

// PATCH /api/samples/:id
router.patch("/:id", updateSample);

// DELETE /api/samples/:id
router.delete("/:id", deleteSample);

// DELETE Many /api/samples/deleteMany
router.post("/deleteMany", deleteManySamples);


export default router;