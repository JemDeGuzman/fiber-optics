# fiber-optics
Monorepo for Team 26's Design Project Application (Fiber Optics)

Leader: De Guzman, Jemuel Endrew C.

Members:

Catapang, Rob Andre (EmTech 3, SoftDes)

Castillo, Mark Laurence (SoftDes)

Pajarillo, Steven Dale 

Tayam, John Chester Irylle 

!!! IMPORTANT !!!

at root:
pnpm install

at packages/api-client:
pnpm install

at infra:
docker compose build
docker start edeed51f430ac1cea1dbb6837d6a673bbaaa171a6d68be5a0284bba981a20b34

at services/api:
pnpm install
// Crea te a .env file, copy .env.example, ask Jem for JWT_SECRET

pnpm add @prisma/client
npx prisma generate --schema=./prisma/schema.prisma

pnpm dev // launches backend, wait for API listening on http://localhost:4000

at apps/web/web-frontend:
pnpm add react react-dom next swr axios
pnpm add -D typescript @types/react @types/node
pnpm install styled-components
pnpm install -D @types/styled-components

pnpm dev // launches frontend, wait for Local: http://localhost:3000 Network: http://xx.xx.x.xx:3000
