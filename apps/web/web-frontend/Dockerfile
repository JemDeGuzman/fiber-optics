# apps/web/web-frontend/Dockerfile
FROM node:20-alpine
WORKDIR /app

# Install build tools + enable corepack/pnpm
RUN apk add --no-cache python3 make g++ && \
    corepack enable && corepack prepare pnpm@latest --activate

# Copy repo lockfile and the web package.json (relative to repo root context)
COPY pnpm-lock.yaml ./
COPY apps/web/web-frontend/package.json ./package.json

# Install only the web workspace deps using workspace filtering
RUN pnpm install --frozen-lockfile --filter ./apps/web/web-frontend...

# Copy the frontend sources into /app
COPY apps/web/web-frontend ./

# Expose Next dev port
EXPOSE 3000

# Use the workspace-filtered dev command
CMD ["pnpm", "dev", "--filter", "./apps/web/web-frontend..."]
