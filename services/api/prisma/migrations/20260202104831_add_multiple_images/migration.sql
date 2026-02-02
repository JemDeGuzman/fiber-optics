-- CreateTable
CREATE TABLE "ImageCapture" (
    "id" SERIAL NOT NULL,
    "sampleId" INTEGER NOT NULL,
    "imageUrl" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ImageCapture_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "ImageCapture_sampleId_idx" ON "ImageCapture"("sampleId");

-- AddForeignKey
ALTER TABLE "ImageCapture" ADD CONSTRAINT "ImageCapture_sampleId_fkey" FOREIGN KEY ("sampleId") REFERENCES "DataSample"("id") ON DELETE CASCADE ON UPDATE CASCADE;
