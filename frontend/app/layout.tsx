import "./globals.css";
export const metadata = { title: "Market Value Pulse", description: "Explainable football valuation intelligence" };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
