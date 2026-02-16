// services/api/src/lib/prisma.ts
import { PrismaClient } from '../generated/client';

const prisma = new PrismaClient();

export default prisma;