import { AudioLines, ShieldCheck } from "lucide-react";
import TranscriptionConsole from "./components/TranscriptionConsole";

export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col px-4 py-8 sm:px-6 lg:py-12">
      <header className="mb-8">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3.5">
            <div
              className="flex h-12 w-12 items-center justify-center rounded-2xl bg-clinical-accent text-white shadow-card"
              aria-hidden="true"
            >
              <AudioLines className="h-6 w-6" strokeWidth={2} />
            </div>
            <div>
              <h1 className="text-[26px] font-semibold leading-none tracking-tight text-slate-900">
                HealthVoice
              </h1>
              <p className="mt-1.5 text-sm text-slate-500">
                Voice-AI clinical scribe · transcription to reviewed FHIR note
              </p>
            </div>
          </div>

          <span className="inline-flex items-center gap-2 rounded-full border border-clinical-border bg-white px-3.5 py-1.5 text-xs font-medium text-slate-600 shadow-card">
            <ShieldCheck className="h-4 w-4 text-clinical-accent" />
            On-device · synthetic data
          </span>
        </div>
      </header>

      <TranscriptionConsole />

      <footer className="mt-12 border-t border-clinical-border pt-5 text-center text-xs text-slate-400">
        Prototype — synthetic data only. Speech, speaker ID, and entity
        extraction run locally on-device.
      </footer>
    </main>
  );
}
