import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "ai-investing cockpit",
  description: "Local AI Investing Platform — v3.1",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
