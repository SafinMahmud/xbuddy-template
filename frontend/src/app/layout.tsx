import type { Metadata, Viewport } from "next";

// Fonts are bundled from npm, so the build needs no request to a font service.
import "@fontsource-variable/bricolage-grotesque";
import "@fontsource-variable/public-sans";

import "./globals.css";

export const metadata: Metadata = {
  title: "JobBuddy: plan your job search",
  description:
    "A job search coach that walks you through five sections and writes an eight-week roadmap from your answers.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
