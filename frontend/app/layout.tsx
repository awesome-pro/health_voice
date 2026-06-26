import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "HealthVoice — Real-time Clinical Voice Scribe",
  description:
    "HealthVoice is a voice-AI clinical scribe prototype: live on-device transcription, speaker ID, medical NER, an AI SOAP note with safety checks, and clinician-reviewed FHIR filing. Synthetic data only.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}
