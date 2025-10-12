import ClientOnly from "@/components/ClientOnly";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html>
      <head>
        <title>Fiber Optics Test App</title>
      </head>
      <body>
        <ClientOnly>{children}</ClientOnly>
      </body>
    </html>
  );
}
