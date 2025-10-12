# services/api/Dockerfile
FROM node:20-alpine
WORKDIR /app

# Install corepack & activate pnpm
RUN apk add --no-cache python3 make g++ && \
    corepack enable && corepack prepare pnpm@latest --activate

# Copy workspace lockfile and the api package.json (relative to repo root context)
COPY pnpm-lock.yaml ./
COPY services/api/package.json ./package.json

# Copy prisma schema early so postinstall/scripts can run if needed
COPY services/api/prisma ./prisma

# Install only api deps using workspace filtering (includes dev deps so prisma CLI is present)
RUN pnpm install --frozen-lockfile --filter ./services/api...

# Generate prisma client explicitly (safe & explicit)
RUN pnpm --filter ./services/api exec prisma generate --schema=./prisma/schema.prisma

# Copy the rest of the API source
COPY services/api ./

EXPOSE 4000

CMD ["pnpm", "dev", "--filter", "./services/api..."]
