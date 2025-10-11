import { Request, Response } from "express";
import { PrismaClient } from "@prisma/client";
import { Parser } from "json2csv";

const prisma = new PrismaClient();

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
  const id = Number(req.params.id);
  const page = Math.max(1, Number(req.query.page) || 1);
  const limit = Math.min(100, Number(req.query.limit) || 50);
  const skip = (page - 1) * limit;

  const [total, samples] = await Promise.all([
    prisma.dataSample.count({ where: { batchId: id } }),
    prisma.dataSample.findMany({
      where: { batchId: id },
      take: limit,
      skip,
      orderBy: { createdAt: "desc" },
      select: { id:true, image_capture:true, classification:true, luster_value:true, roughness:true, tensile_strength:true, createdAt:true }
    })
  ]);

  res.json({ total, page, limit, samples });
}

export async function exportCsv(req: Request, res: Response) {
  const id = Number(req.params.id);
  const samples = await prisma.dataSample.findMany({
    where: { batchId: id },
    select: { id:true, image_capture:true, classification:true, luster_value:true, roughness:true, tensile_strength:true, createdAt:true }
  });

  const fields = ["id","image_capture","classification","luster_value","roughness","tensile_strength","createdAt"];
  const parser = new Parser({ fields });
  const csv = parser.parse(samples);

  res.setHeader("Content-Disposition", `attachment; filename=batch-${id}.csv`);
  res.setHeader("Content-Type", "text/csv");
  res.send(csv);
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