const PrismaPkg = require("@prisma/client");
const PrismaClient = PrismaPkg.PrismaClient ?? PrismaPkg.default ?? PrismaPkg;
const prisma = new PrismaClient();
const classifications = ["Abaca", "Daratex", "Mixed"];
function randFloat(min = 0, max = 1) { return +(Math.random() * (max - min) + min).toFixed(3); }
function randChoice(arr) { return arr[Math.floor(Math.random() * arr.length)]; }
async function main() {
    // create a batch
    const batch = await prisma.dataBatch.create({ data: { name: `Batch ${Date.now()}` } });
    console.log("Created batch", batch.id);
    // create 200 random samples
    const samples = [];
    for (let i = 0; i < 200; i++) {
        samples.push({
            batchId: batch.id,
            image_capture: `img_${Date.now()}_${i}.jpg`,
            classification: randChoice(classifications),
            luster_value: randFloat(0, 100),
            roughness: randFloat(0, 10),
            tensile_strength: randFloat(0, 200)
        });
    }
    await prisma.dataSample.createMany({ data: samples });
    console.log("Inserted samples:", samples.length);
}
main()
    .catch(e => { console.error(e); process.exit(1); })
    .finally(() => prisma.$disconnect());
