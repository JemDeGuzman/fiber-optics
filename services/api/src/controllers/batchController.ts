import { Request, Response } from "express";
import { PrismaClient } from "@prisma/client";
import { Parser } from "json2csv";

const prisma = new PrismaClient();

export async function listBatches(req: Request, res: Response) {
  try {
    const batches = await prisma.dataBatch.findMany({ orderBy: { createdAt: "desc" }});
    return res.json({ batches });
  } catch (err: any) {
    console.error("listBatches error:", err);
    return res.status(500).json({ error: "Failed to list batches", detail: err?.message ?? String(err) });
  }
}

export async function createBatch(req: Request, res: Response) {
  const { name } = req.body;
  if (!name) return res.status(400).json({ error: "Missing name" });
  const batch = await prisma.dataBatch.create({ data: { name }});
  res.json({ batch });
}

export async function getBatchStats(req: Request, res: Response) {
  const id = Number(req.params.id);
  if (!id) return res.status(400).json({ error: "Invalid id" });

  const total = await prisma.dataSample.count({ where: { batchId: id }});
  // group by classification
  const byClass = await prisma.$queryRaw<
    { classification: string; count: number }[]
  >`SELECT classification, COUNT(*) as count FROM "DataSample" WHERE "batchId" = ${id} GROUP BY classification`;

  // convert to object { classification: count }
  const ratio: Record<string, number> = {};
  byClass.forEach(r => { ratio[r.classification] = Number(r.count); });

  res.json({ batchId: id, total, ratio });
}

export async function listSamples(req: Request, res: Response) {
  try {
    const batchId = Number(req.params.id);
    if (!batchId || Number.isNaN(batchId)) {
      return res.status(400).json({ error: "Invalid batch id" });
    }

    const page = Math.max(1, Number(req.query.page as string ? req.query.page as any : 1) || 1);
    const limit = Math.min(500, Number(req.query.limit as string ? req.query.limit as any : 50) || 50);
    const skip = (page - 1) * limit;

    const [total, samples] = await Promise.all([
      prisma.dataSample.count({ where: { batchId } }),
      prisma.dataSample.findMany({
        where: { batchId },
        take: limit,
        skip,
        orderBy: { createdAt: "desc" },
        select: { id:true, image_capture:true, classification:true, luster_value:true, roughness:true, tensile_strength:true, createdAt:true }
      })
    ]);

    return res.json({ total, page, limit, samples });
  } catch (err: any) {
    console.error("listSamples error:", err);
    return res.status(500).json({ error: "Failed to list samples", detail: err?.message ?? String(err) });
  }
}

/* --- exportCsv: supports GET (all / page) and POST (selected ids) --- */
export async function exportCsv(req: Request, res: Response) {
  try {
    const batchId = Number(req.params.id);
    if (!batchId || Number.isNaN(batchId)) {
      return res.status(400).json({ error: "Invalid batch id" });
    }

    let samples;
    if (req.method === "POST") {
      // export specific ids: POST /api/batches/:id/export { ids: [1,2,3] }
      const ids: number[] = (req.body && Array.isArray(req.body.ids)) ? req.body.ids.map(Number) : [];
      if (ids.length === 0) {
        return res.status(400).json({ error: "Missing ids to export" });
      }
      samples = await prisma.dataSample.findMany({
        where: { batchId, id: { in: ids } },
        select: { id:true, image_capture:true, classification:true, luster_value:true, roughness:true, tensile_strength:true, createdAt:true }
      });
    } else {
      // GET: either all or page limited export
      const maybePage = req.query.page ? Number(req.query.page) : undefined;
      const maybeLimit = req.query.limit ? Number(req.query.limit) : undefined;

      if (maybePage && maybeLimit) {
        const page = Math.max(1, maybePage);
        const limit = Math.min(500, maybeLimit);
        const skip = (page - 1) * limit;
        samples = await prisma.dataSample.findMany({
          where: { batchId },
          take: limit,
          skip,
          orderBy: { createdAt: "desc" },
          select: { id:true, image_capture:true, classification:true, luster_value:true, roughness:true, tensile_strength:true, createdAt:true }
        });
      } else {
        // export all
        samples = await prisma.dataSample.findMany({
          where: { batchId },
          orderBy: { createdAt: "desc" },
          select: { id:true, image_capture:true, classification:true, luster_value:true, roughness:true, tensile_strength:true, createdAt:true }
        });
      }
    }

    const fields = ["id","image_capture","classification","luster_value","roughness","tensile_strength","createdAt"];
    const parser = new Parser({ fields });
    const csv = parser.parse(samples);

    res.setHeader("Content-Disposition", `attachment; filename=batch-${batchId}.csv`);
    res.setHeader("Content-Type", "text/csv");
    res.send(csv);

  } catch (err: any) {
    console.error("exportCsv error:", err);
    return res.status(500).json({ error: "Failed to export CSV", detail: err?.message ?? String(err) });
  }
}

export async function createSample(req: Request, res: Response) {
  try {
    const batchId = Number(req.params.id);
    if (!batchId || Number.isNaN(batchId)) {
      return res.status(400).json({ error: "Invalid batch id" });
    }

    const {
      classification,
      luster_value,
      roughness,
      tensile_strength,
      image_capture
    } = req.body ?? {};

    // Required field checks
    if (!classification || typeof classification !== "string") {
      return res.status(400).json({ error: "Missing or invalid field: classification" });
    }

    // Parse numeric fields if provided (allow strings that represent numbers)
    const parseNullableNumber = (v: any): number | null => {
      if (v === undefined || v === null || v === "") return null;
      const n = Number(v);
      return Number.isFinite(n) ? n : null;
    };
    const lusterVal = parseNullableNumber(luster_value);
    const roughnessVal = parseNullableNumber(roughness);
    const tensileVal = parseNullableNumber(tensile_strength);

    const sample = await prisma.dataSample.create({
      data: {
        batchId,
        classification,
        luster_value: lusterVal,
        roughness: roughnessVal,
        tensile_strength: tensileVal,
        image_capture: image_capture ?? null,
      }
    });

    return res.status(201).json({ sample });
  } catch (err: any) {
    console.error("createSample error:", err);
    // Prisma client errors sometimes contain `code` or `meta`, include detail for debugging
    return res.status(500).json({
      error: "Internal server error",
      detail: err?.message ?? String(err)
    });
  }
}

/** Update batch name */
export async function updateBatch(req: Request, res: Response) {
  try {
    const id = Number(req.params.id);
    if (!id || Number.isNaN(id)) return res.status(400).json({ error: "Invalid id" });

    const { name } = req.body;
    if (!name || typeof name !== "string") {
      return res.status(400).json({ error: "Missing or invalid name" });
    }

    const batch = await prisma.dataBatch.update({
      where: { id },
      data: { name },
    });
    return res.json({ batch });
  } catch (err: any) {
    console.error("updateBatch error:", err);
    return res.status(500).json({ error: "Failed to update batch", detail: err?.message });
  }
}

/** Delete a batch and its samples */
export async function deleteBatch(req: Request, res: Response) {
  try {
    const id = Number(req.params.id);
    if (!id || Number.isNaN(id)) return res.status(400).json({ error: "Invalid id" });

    // delete samples first (if FK constraints)
    await prisma.dataSample.deleteMany({ where: { batchId: id } });
    await prisma.dataBatch.delete({ where: { id } });

    return res.json({ ok: true });
  } catch (err: any) {
    console.error("deleteBatch error:", err);
    return res.status(500).json({ error: "Failed to delete batch", detail: err?.message });
  }
}

/** Update a sample by id */
export async function updateSample(req: Request, res: Response) {
  try {
    const id = Number(req.params.id);
    if (!id || Number.isNaN(id)) return res.status(400).json({ error: "Invalid sample id" });

    const {
      classification,
      luster_value,
      roughness,
      tensile_strength,
      image_capture
    } = req.body ?? {};

    // Build data object only with provided fields
    const data: any = {};
    if (classification !== undefined) data.classification = classification;
    if (luster_value !== undefined) data.luster_value = luster_value === "" ? null : Number(luster_value);
    if (roughness !== undefined) data.roughness = roughness === "" ? null : Number(roughness);
    if (tensile_strength !== undefined) data.tensile_strength = tensile_strength === "" ? null : Number(tensile_strength);
    if (image_capture !== undefined) data.image_capture = image_capture;

    const sample = await prisma.dataSample.update({
      where: { id },
      data,
    });

    return res.json({ sample });
  } catch (err: any) {
    console.error("updateSample error:", err);
    return res.status(500).json({ error: "Failed to update sample", detail: err?.message });
  }
}

/** Delete a sample by id */
export async function deleteSample(req: Request, res: Response) {
  try {
    const id = Number(req.params.id);
    if (!id || Number.isNaN(id)) return res.status(400).json({ error: "Invalid sample id" });

    await prisma.dataSample.delete({ where: { id } });
    return res.json({ ok: true });
  } catch (err: any) {
    console.error("deleteSample error:", err);
    return res.status(500).json({ error: "Failed to delete sample", detail: err?.message });
  }
}

// services/api/src/controllers/batchController.ts (or samples controller)
export async function deleteManySamples(req: Request, res: Response) {
  try {
    const ids = Array.isArray(req.body.ids) ? req.body.ids.map(Number) : [];
    if (ids.length === 0) return res.status(400).json({ error: "Missing ids" });

    // deleteMany supports where: { id: { in: ids } }
    await prisma.dataSample.deleteMany({ where: { id: { in: ids } } });

    return res.json({ ok: true, deleted: ids.length });
  } catch (err: any) {
    console.error("deleteManySamples error:", err);
    return res.status(500).json({ error: "Failed to delete samples", detail: err?.message });
  }
}
