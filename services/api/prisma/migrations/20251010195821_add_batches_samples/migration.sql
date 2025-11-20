-- CreateTable
CREATE TABLE "DataBatch" (
    "id" SERIAL NOT NULL,
    "name" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "DataBatch_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "DataSample" (
    "id" SERIAL NOT NULL,
    "batchId" INTEGER NOT NULL,
    "image_capture" TEXT NOT NULL,
    "classification" TEXT NOT NULL,
    "luster_value" DOUBLE PRECISION,
    "roughness" DOUBLE PRECISION,
    "tensile_strength" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "DataSample_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "DataSample_batchId_idx" ON "DataSample"("batchId");

-- AddForeignKey
ALTER TABLE "DataSample" ADD CONSTRAINT "DataSample_batchId_fkey" FOREIGN KEY ("batchId") REFERENCES "DataBatch"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
