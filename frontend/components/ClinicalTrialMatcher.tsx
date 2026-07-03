"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";

// ─────────────────────────────────────────────────────────────
// 1. Types & Interfaces
// ─────────────────────────────────────────────────────────────

interface PatientProfile {
  age: number | null;
  gender: string | null;
  cancer_type: string | null;
  cancer_stage: string | null;
  biomarkers: string[];
  previous_treatments: string[];
  country?: string | null;
  diagnosis?: string | null;
}

interface EligibilityResult {
  nct_id: string | null;
  title: string | null;
  hard_filter_pass: boolean;
  hard_filter_reasons: string[];
  score: number;
  score_reasons: string[];
  biomarker_check: string | null;
  treatment_check: string | null;
  assessment: string | null;
}

interface TrialMatchResult {
  final_recommendations: string;
  eligibility_results: EligibilityResult[];
  trials: unknown[];
  trial_count: number;
  cancer_type: string;
  success: boolean;
  error: string | null;
}

interface AnalysisResult {
  profile: PatientProfile;
  status: "PROFILE_READY" | "NEEDS_CLARIFICATION" | "MATCHING_FAILED";
  is_complete: boolean;
  missing_fields: string[];
  agent_suggestions: string[];
  trial_matches: TrialMatchResult | null;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

type InputMode = "upload" | "manual";

// ─────────────────────────────────────────────────────────────
// 2. Constants & Configuration
// ─────────────────────────────────────────────────────────────

const CONFIG = {
  API_BASE: process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000/api/v1",
  MAX_FILE_SIZE_BYTES: 10 * 1024 * 1024,
  ALLOWED_FILE_TYPES: ["application/pdf"],
};

const SUGGESTED_QUESTIONS = [
  "What does EGFR Exon 19 deletion mean for trial eligibility?",
  "Why didn't more trials match my patient?",
  "What does research say about HER2-low breast cancer trials?",
  "Are there targeted therapies available for Stage IV NSCLC?",
];

const TECH_STACK = [
  "Next.js 15", "PostgreSQL", "LangChain", "LangGraph",
  "Deep Agents", "Groq API", "LLaMA 3.3 70b",
  "Python 3.14", "FastAPI", "ClinicalTrials.gov",
];

const EMPTY_PROFILE: PatientProfile = {
  age: null,
  gender: null,
  cancer_type: null,
  cancer_stage: null,
  biomarkers: [],
  previous_treatments: [],
  country: null,
  diagnosis: null,
};

const generateId = (): string => Math.random().toString(36).substring(2, 9);

function parseCommaSeparated(value: string): string[] {
  return value.split(",").map((v) => v.trim()).filter(Boolean);
}

function stringifyList(value?: string[] | null): string {
  return value?.join(", ") ?? "";
}

// ─────────────────────────────────────────────────────────────
// 3. Tier Utilities
// ─────────────────────────────────────────────────────────────

type Tier = "TIER_1" | "TIER_2" | "TIER_3" | "EXCLUDED";

function assignTier(result: EligibilityResult): Tier {
  if (!result.hard_filter_pass) return "EXCLUDED";
  const score = result.score ?? 0;
  if (score >= 10) return "TIER_1";
  if (score >= 5)  return "TIER_2";
  if (score >= 1)  return "TIER_3";
  return "EXCLUDED";
}

const TIER_CONFIG: Record<
  Tier,
  { label: string; emoji: string; border: string; badge: string; hex: string }
> = {
  TIER_1: {
    label: "Strong Match",
    emoji: "🥇",
    border: "border-[#2dd4bf]/40 ring-1 ring-[#2dd4bf]/10 shadow-lg shadow-[#0d9488]/10",
    badge: "bg-[#0d9488]/20 text-[#2dd4bf] border border-[#2dd4bf]/30",
    hex: "#2dd4bf"
  },
  TIER_2: {
    label: "Moderate Match",
    emoji: "🥈",
    border: "border-white/10",
    badge: "bg-white/5 text-teal-400 border border-white/10",
    hex: "#0d9488"
  },
  TIER_3: {
    label: "Possible Match",
    emoji: "🥉",
    border: "border-white/5",
    badge: "bg-black/20 text-slate-300 border border-white/5",
    hex: "#94a3b8"
  },
  EXCLUDED: {
    label: "Not Matched",
    emoji: "❌",
    border: "border-rose-900/30",
    badge: "bg-rose-900/20 text-rose-400 border border-rose-800/30",
    hex: "#f43f5e"
  },
};

// ─────────────────────────────────────────────────────────────
// 4. Shared UI Components
// ─────────────────────────────────────────────────────────────

const StatusBadge = React.memo(({ status }: { status: AnalysisResult["status"] }) => {
  const styles: Record<AnalysisResult["status"], string> = {
    PROFILE_READY: "bg-[#0d9488]/20 text-[#2dd4bf] border-[#0d9488]/30",
    NEEDS_CLARIFICATION: "bg-amber-900/20 text-amber-400 border-amber-700/30",
    MATCHING_FAILED: "bg-rose-900/20 text-rose-400 border-rose-800/30",
  };
  const labels: Record<AnalysisResult["status"], string> = {
    PROFILE_READY: "Profile Ready",
    NEEDS_CLARIFICATION: "Needs Clarification",
    MATCHING_FAILED: "Matching Failed",
  };
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-bold ${styles[status]}`}>
      {labels[status]}
    </span>
  );
});
StatusBadge.displayName = "StatusBadge";

const OncoPilotLogo = ({ size = 32 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="16" cy="16" r="13" stroke="#0d9488" strokeWidth="1.5"></circle>
    <circle cx="16" cy="16" r="6" stroke="#2dd4bf" strokeWidth="1.5"></circle>
    <circle cx="16" cy="16" r="2.5" fill="#2dd4bf"></circle>
    <line x1="16" y1="2" x2="16" y2="9" stroke="#0d9488" strokeWidth="1.5"></line>
    <line x1="16" y1="23" x2="16" y2="30" stroke="#0d9488" strokeWidth="1.5"></line>
    <line x1="2" y1="16" x2="9" y2="16" stroke="#0d9488" strokeWidth="1.5"></line>
    <line x1="23" y1="16" x2="30" y2="16" stroke="#0d9488" strokeWidth="1.5"></line>
    <path d="M9 16 L12 16 L13.5 12 L15.5 20 L17 16 L23 16" stroke="#2dd4bf" strokeWidth="1.5" strokeLinejoin="round"></path>
  </svg>
);

const OncoPilotWordmark = () => (
  <div className="flex items-center gap-2">
    <OncoPilotLogo size={28} />
    <div className="text-2xl font-bold tracking-tight">
      <span className="text-white">Onco</span>
      <span className="text-[#2dd4bf]">Pilot</span>
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────
// 5. Manual Entry Form Component
// ─────────────────────────────────────────────────────────────

function ManualProfileForm({
  profile,
  onChange,
  onSubmit,
  isProcessing,
}: {
  profile: PatientProfile;
  onChange: (next: PatientProfile) => void;
  onSubmit: () => void;
  isProcessing: boolean;
}) {
  const updateField = <K extends keyof PatientProfile>(key: K, value: PatientProfile[K]) => {
    onChange({ ...profile, [key]: value });
  };

  const inputClass = "w-full rounded-xl border border-white/10 bg-black/40 py-3 px-4 text-sm text-white shadow-inner focus:border-[#2dd4bf] focus:ring-1 focus:ring-[#2dd4bf] outline-none placeholder:text-slate-500 backdrop-blur-md transition-colors";
  const labelClass = "block text-sm font-bold text-slate-300 mb-2";

  return (
    <div className="space-y-6 mt-4 text-left relative z-10">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div>
          <label className={labelClass}>Age</label>
          <input type="number" min={1} max={120} value={profile.age ?? ""} onChange={(e) => updateField("age", e.target.value ? Number(e.target.value) : null)} className={inputClass} placeholder="e.g. 58" />
        </div>
        <div>
          <label className={labelClass}>Gender</label>
          <select value={profile.gender ?? ""} onChange={(e) => updateField("gender", e.target.value || null)} className={inputClass}>
            <option value="" className="bg-slate-900">Select</option>
            <option value="male" className="bg-slate-900">Male</option>
            <option value="female" className="bg-slate-900">Female</option>
            <option value="other" className="bg-slate-900">Other / Unknown</option>
          </select>
        </div>
        <div>
          <label className={labelClass}>Cancer Type</label>
          <input type="text" value={profile.cancer_type ?? ""} onChange={(e) => updateField("cancer_type", e.target.value || null)} className={inputClass} placeholder="e.g. Non-small cell lung cancer" />
        </div>
        <div>
          <label className={labelClass}>Cancer Stage</label>
          <input type="text" value={profile.cancer_stage ?? ""} onChange={(e) => updateField("cancer_stage", e.target.value || null)} className={inputClass} placeholder="e.g. Stage IV metastatic" />
        </div>
        <div>
          <label className={labelClass}>Country</label>
          <input type="text" value={profile.country ?? ""} onChange={(e) => updateField("country", e.target.value || null)} className={inputClass} placeholder="e.g. India" />
        </div>
        <div>
          <label className={labelClass}>Biomarkers <span className="text-xs font-normal text-slate-500">(comma-separated)</span></label>
          <input type="text" value={stringifyList(profile.biomarkers)} onChange={(e) => updateField("biomarkers", parseCommaSeparated(e.target.value))} className={inputClass} placeholder="e.g. EGFR, ALK, PD-L1" />
        </div>
        <div className="md:col-span-2">
          <label className={labelClass}>Previous Treatments <span className="text-xs font-normal text-slate-500">(comma-separated)</span></label>
          <input type="text" value={stringifyList(profile.previous_treatments)} onChange={(e) => updateField("previous_treatments", parseCommaSeparated(e.target.value))} className={inputClass} placeholder="e.g. Carboplatin, Osimertinib" />
        </div>
        <div className="md:col-span-2">
          <label className={labelClass}>Diagnosis Notes</label>
          <textarea rows={3} value={profile.diagnosis ?? ""} onChange={(e) => updateField("diagnosis", e.target.value || null)} className={inputClass} placeholder="e.g. Metastatic HER2-positive breast cancer with liver involvement" />
        </div>
      </div>
      <button
        disabled={isProcessing}
        onClick={onSubmit}
        className="w-full mt-4 px-8 py-4 bg-[#2dd4bf] hover:bg-teal-300 disabled:bg-slate-700 disabled:text-slate-400 disabled:shadow-none text-[#020617] font-black text-lg rounded-xl transition-all shadow-[0_0_20px_rgba(45,212,191,0.3)] hover:shadow-[0_0_30px_rgba(45,212,191,0.5)] hover:-translate-y-0.5"
      >
        {isProcessing ? "Analyzing..." : "Analyze & Match Trials →"}
      </button>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// 6. Floating Chat Widget
// ─────────────────────────────────────────────────────────────

const FloatingChatWidget = React.memo(({ profile, trialMatches }: { profile: PatientProfile | null; trialMatches: TrialMatchResult | null }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [hasUnread, setHasUnread] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!profile || messages.length > 0) return;
    const type = trialMatches?.cancer_type ?? profile.cancer_type ?? "the oncology case";
    const count = trialMatches?.trial_count ?? 0;
    setMessages([{
      id: generateId(), 
      role: "assistant",
      content: `I've reviewed the clinical profile for ${type} and evaluated ${count} recruiting trials. What would you like to understand better?`,
      timestamp: new Date(),
    }]);
    if (!isOpen) setHasUnread(true);
  }, [profile, trialMatches, messages.length, isOpen]);

  useEffect(() => {
    if (isOpen) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading, isOpen]);

  useEffect(() => {
    if (isOpen) {
      setHasUnread(false);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
    const onEscape = (e: KeyboardEvent) => { if (e.key === "Escape") setIsOpen(false); };
    const onOutside = (e: MouseEvent) => {
      const target = e.target as Node | null;
      if (!target) return;
      const targetElement = target instanceof Element ? target : target.parentElement;
      if (isOpen && panelRef.current && !panelRef.current.contains(target) && !targetElement?.closest("#chat-fab")) {
        setIsOpen(false);
      }
    };
    window.addEventListener("keydown", onEscape);
    document.addEventListener("mousedown", onOutside);
    return () => {
      window.removeEventListener("keydown", onEscape);
      document.removeEventListener("mousedown", onOutside);
    };
  }, [isOpen]);

  const sendMessage = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isLoading) return;
    
    setMessages(prev => [...prev, {
      id: generateId(), role: "user",
      content: trimmed, timestamp: new Date(),
    }]);
    setInput("");
    setIsLoading(true);

    const compactTrialMatches = trialMatches ? {
      trial_count: trialMatches.trial_count,
      cancer_type: trialMatches.cancer_type,
      final_recommendations: trialMatches.final_recommendations?.slice(0, 1200) ?? "",
      eligibility_results: trialMatches.eligibility_results.slice(0, 5).map((trial) => ({
        nct_id: trial.nct_id, title: trial.title, score: trial.score, hard_filter_pass: trial.hard_filter_pass, score_reasons: trial.score_reasons?.slice(0, 4) ?? [],
      })),
    } : null;

    try {
      const res = await fetch(`${CONFIG.API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message: trimmed,
          patient_profile: profile,
          trial_matches: compactTrialMatches
        }),
      });
      if (!res.ok) throw new Error("Chat unavailable");
      const data = await res.json();
      if (!sessionId && data.session_id) setSessionId(data.session_id);
      setMessages(prev => [...prev, {
        id: generateId(), role: "assistant",
        content: data.response, timestamp: new Date(),
      }]);
    } catch {
      setMessages(prev => [...prev, {
        id: generateId(), role: "assistant",
        content: "I'm having trouble connecting right now. Please try again.",
        timestamp: new Date(),
      }]);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, profile, sessionId, trialMatches]);

  return (
    <>
      {isOpen && (
        <div ref={panelRef} className="fixed bottom-24 right-4 z-50 w-80 rounded-2xl bg-[#0f172a]/90 backdrop-blur-2xl shadow-2xl shadow-black/50 border border-white/10 sm:right-8 sm:w-96 flex flex-col h-[500px] overflow-hidden text-slate-200">
          <header className="flex items-center justify-between border-b border-white/10 bg-white/5 px-4 py-3 backdrop-blur-md">
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#0d9488]/20 text-[#2dd4bf] border border-[#2dd4bf]/20">
                <OncoPilotLogo size={18} />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-white">Oncology Agent</h3>
                <p className="text-xs text-slate-400">Clinical Intelligence · grounded</p>
              </div>
            </div>
            <button onClick={() => setIsOpen(false)} className="rounded-lg p-2 text-slate-400 hover:bg-white/10 hover:text-white transition-colors" aria-label="Close chat">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" className="w-5 h-5"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
            </button>
          </header>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length <= 1 && profile && (
              <div className="space-y-2">
                {SUGGESTED_QUESTIONS.map(q => (
                  <button key={q} onClick={() => sendMessage(q)} className="w-full rounded-xl border border-white/5 bg-white/5 p-3 text-left text-xs text-slate-300 hover:bg-white/10 hover:text-white transition-colors">
                    {q}
                  </button>
                ))}
              </div>
            )}
            <div className="space-y-4">
              {messages.map(msg => (
                <div key={msg.id} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div className={`max-w-[85%] rounded-2xl px-4 py-2.5 shadow-sm text-sm leading-relaxed ${msg.role === "user" ? "bg-[#0d9488] text-white" : "bg-white/5 text-slate-200 border border-white/5 backdrop-blur-sm"}`}>
                    <p>{msg.content}</p>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="rounded-2xl bg-white/5 border border-white/5 px-4 py-3 backdrop-blur-sm">
                    <div className="flex space-x-1.5">
                      <span className="h-2 w-2 animate-bounce rounded-full bg-[#2dd4bf]"></span>
                      <span className="h-2 w-2 animate-bounce rounded-full bg-[#2dd4bf] delay-75"></span>
                      <span className="h-2 w-2 animate-bounce rounded-full bg-[#2dd4bf] delay-150"></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          </div>

          <div className="border-t border-white/10 bg-white/5 p-4 backdrop-blur-md">
            <div className="relative">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && !e.nativeEvent.isComposing) sendMessage(input); }}
                placeholder={profile ? "Ask about a trial or biomarker..." : "Upload a profile to chat..."}
                disabled={!profile || isLoading}
                className="w-full rounded-xl border border-white/10 bg-black/20 py-3 pl-4 pr-12 text-sm text-white shadow-inner focus:border-[#2dd4bf] focus:ring-1 focus:ring-[#2dd4bf] outline-none placeholder:text-slate-500 backdrop-blur-md"
              />
              <button
                onClick={() => sendMessage(input)}
                disabled={!input.trim() || isLoading}
                aria-label="Send"
                className="absolute right-2 top-1.5 p-1.5 rounded-lg bg-[#0d9488] text-white disabled:opacity-50 hover:bg-teal-500 transition-colors"
              >
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" className="w-5 h-5"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
              </button>
            </div>
          </div>
        </div>
      )}

      <button
        id="chat-fab"
        onClick={() => setIsOpen(v => !v)}
        aria-label="Toggle Oncology Agent"
        className="fixed bottom-4 md:bottom-8 right-4 md:right-8 z-50 w-14 h-14 bg-[#0d9488] hover:bg-teal-500 text-white rounded-full shadow-lg shadow-[#0d9488]/40 flex items-center justify-center transition-all hover:-translate-y-1 border border-[#2dd4bf]/30 backdrop-blur-md"
      >
        {!isOpen ? (
          <OncoPilotLogo size={24} />
        ) : (
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" className="w-6 h-6"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12"></path></svg>
        )}
        {hasUnread && !isOpen && <span className="absolute -right-1 -top-1 flex h-4 w-4 rounded-full bg-[#2dd4bf] ring-2 ring-[#020617]"></span>}
      </button>
    </>
  );
});
FloatingChatWidget.displayName = "FloatingChatWidget";

// ─────────────────────────────────────────────────────────────
// 7. Trial Card & Patient Profile
// ─────────────────────────────────────────────────────────────

const TrialCard = React.memo(({ result, index }: { result: EligibilityResult; index: number }) => {
  const [expanded, setExpanded] = useState(false);
  const tier = assignTier(result);
  const config = TIER_CONFIG[tier];
  if (tier === "EXCLUDED") return null;

  const cautionKeywords = ["not", "mismatch", "excluded", "exceed", "unclear", "caution", "unable", "missing", "negative", "conflict"];
  const positiveReasons = result.score_reasons.filter(r => !cautionKeywords.some(k => r.toLowerCase().includes(k)));
  const cautionReasons = result.score_reasons.filter(r => cautionKeywords.some(k => r.toLowerCase().includes(k)));

  return (
    <div className={`rounded-xl bg-[#0f172a]/50 backdrop-blur-xl p-5 transition-all shadow-lg hover:bg-[#0f172a]/70 ${config.border}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-bold rounded-md ${config.badge}`}>
            {config.emoji} {config.label}
          </span>
          <span className="text-sm font-medium text-slate-400">Score: {result.score}</span>
        </div>
        <span className="text-sm font-mono text-slate-500 bg-white/5 px-2 py-0.5 rounded">#{index + 1}</span>
      </div>
      <h3 className="text-lg font-bold text-white mb-1.5 leading-snug">{result.title ?? "Untitled Trial"}</h3>
      {result.nct_id && <p className="text-sm text-[#2dd4bf] font-mono mb-4">{result.nct_id}</p>}

      {positiveReasons.length > 0 && (
        <ul className="space-y-1.5 mb-3">
          {positiveReasons.slice(0, 3).map((r, i) => (
            <li key={i} className="flex gap-2.5 text-sm text-slate-300"><span className="text-[#2dd4bf] font-bold">✓</span><span>{r}</span></li>
          ))}
        </ul>
      )}
      {cautionReasons.length > 0 && (
        <ul className="space-y-1.5 mb-4">
          {cautionReasons.slice(0, 2).map((r, i) => (
            <li key={i} className="flex gap-2.5 text-sm text-slate-300"><span className="text-amber-500 font-bold">⚠</span><span>{r}</span></li>
          ))}
        </ul>
      )}

      <button onClick={() => setExpanded(v => !v)} className="mt-2 text-xs font-bold tracking-wide text-[#2dd4bf] hover:text-teal-300 transition-colors uppercase">
        {expanded ? "▲ Show less" : "▼ Show full assessment"}
      </button>

      {expanded && result.assessment && (
        <div className="mt-5 pt-5 border-t border-white/10 space-y-4">
          <div>
            <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">LLM Assessment</p>
            <p className="text-sm text-slate-300 leading-relaxed bg-black/20 p-4 rounded-xl border border-white/5">{result.assessment}</p>
          </div>
          {result.nct_id && (
            <a href={`https://clinicaltrials.gov/study/${result.nct_id}`} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1.5 text-sm font-bold text-[#2dd4bf] hover:text-teal-300 transition-colors">
              🔗 View on ClinicalTrials.gov
            </a>
          )}
        </div>
      )}
    </div>
  );
});
TrialCard.displayName = "TrialCard";

const PatientProfilePanel = React.memo(({ analysis }: { analysis: AnalysisResult }) => {
  const { profile, status, missing_fields, trial_matches } = analysis;
  const tier1 = trial_matches?.eligibility_results.filter(r => assignTier(r) === "TIER_1").length ?? 0;
  const tier2 = trial_matches?.eligibility_results.filter(r => assignTier(r) === "TIER_2").length ?? 0;
  const tier3 = trial_matches?.eligibility_results.filter(r => assignTier(r) === "TIER_3").length ?? 0;

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-white/10 bg-[#0f172a]/50 backdrop-blur-xl p-6 shadow-xl">
        <div className="mb-6 flex items-center justify-between border-b border-white/10 pb-4">
          <h2 className="text-lg font-bold text-white">Patient Profile</h2>
          <StatusBadge status={status} />
        </div>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-6 mb-6">
          {[
            { label: "Age",         value: profile.age ? `${profile.age} years` : null },
            { label: "Gender",      value: profile.gender },
            { label: "Cancer Type", value: profile.cancer_type ?? profile.diagnosis },
            { label: "Stage",       value: profile.cancer_stage },
            { label: "Country",     value: profile.country },
          ].map(({ label, value }) => (
            <div key={label}>
              <dt className="text-xs font-bold text-slate-500 uppercase tracking-wide mb-1">{label}</dt>
              <dd className="text-sm font-medium text-slate-200">{value ?? <span className="text-slate-600 italic">Not available</span>}</dd>
            </div>
          ))}
        </dl>

        {profile.biomarkers.length > 0 && (
          <div className="mb-6">
            <p className="text-xs font-bold text-slate-500 uppercase tracking-wide mb-3">
              Biomarkers <span className="text-slate-600 font-normal normal-case">— flags that guide decisions</span>
            </p>
            <div className="flex flex-wrap gap-2">
              {profile.biomarkers.map(b => (
                <span key={b} className="px-2.5 py-1 text-xs font-mono bg-[#0d9488]/20 text-[#2dd4bf] border border-[#0d9488]/30 rounded-md">{b}</span>
              ))}
            </div>
          </div>
        )}

        {profile.previous_treatments.length > 0 && (
          <div className="mb-6">
            <p className="text-xs font-bold text-slate-500 uppercase tracking-wide mb-3">
              Prior Treatments <span className="text-slate-600 font-normal normal-case">— therapies received</span>
            </p>
            <div className="flex flex-wrap gap-2">
              {profile.previous_treatments.map(t => (
                <span key={t} className="px-2.5 py-1 text-xs bg-slate-800 text-slate-300 border border-slate-700 rounded-md">{t}</span>
              ))}
            </div>
          </div>
        )}

        {missing_fields.length > 0 && (
          <div className="p-4 bg-amber-900/10 border border-amber-700/20 rounded-xl mt-6">
            <p className="text-xs font-bold text-amber-500 uppercase tracking-wider mb-1">⚠ Missing Fields</p>
            <p className="text-sm text-slate-300 mb-1">{missing_fields.join(", ")}</p>
            <p className="text-xs text-slate-500">Adding these may surface more matches.</p>
          </div>
        )}
      </div>

      {trial_matches && (
        <div className="rounded-2xl border border-white/10 bg-[#0f172a]/50 backdrop-blur-xl p-6 shadow-xl">
          <h2 className="text-lg font-bold text-white mb-1">Match Summary</h2>
          <p className="text-xs text-slate-400 mb-5">Trials ranked by clinical alignment strength</p>
          <div className="grid grid-cols-3 gap-3 mb-4">
            <div className="text-center p-3 rounded-xl bg-[#0d9488]/20 border border-[#0d9488]/30 shadow-inner">
              <p className="text-2xl font-bold text-[#2dd4bf]">{tier1}</p>
              <p className="text-[10px] font-bold text-[#2dd4bf] mt-1 uppercase tracking-wider">🥇 Strong</p>
            </div>
            <div className="text-center p-3 rounded-xl bg-white/5 border border-white/5 shadow-inner">
              <p className="text-2xl font-bold text-teal-400">{tier2}</p>
              <p className="text-[10px] font-bold text-slate-400 mt-1 uppercase tracking-wider">🥈 Moderate</p>
            </div>
            <div className="text-center p-3 rounded-xl bg-black/20 border border-white/5 shadow-inner">
              <p className="text-2xl font-bold text-slate-400">{tier3}</p>
              <p className="text-[10px] font-bold text-slate-500 mt-1 uppercase tracking-wider">🥉 Possible</p>
            </div>
          </div>
          <p className="text-xs text-center font-medium text-slate-500">
            {trial_matches.trial_count} total trials evaluated
          </p>
          
          <div className="mt-6 pt-5 border-t border-white/10">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4">What the tiers mean</h3>
            <div className="space-y-4">
              {[
                { tier: "🥇 Strong",   score: "Score ≥ 10", desc: "High alignment across cancer type, stage, biomarkers, and location." },
                { tier: "🥈 Moderate", score: "Score 5–9",  desc: "Good alignment on most dimensions. Worth reviewing with a clinician." },
                { tier: "🥉 Possible", score: "Score 1–4",  desc: "Partial alignment. May still be relevant — check the full assessment." },
              ].map(item => (
                <div key={item.tier} className="flex gap-3">
                  <span className="text-sm font-bold text-slate-300 w-24 shrink-0">{item.tier}</span>
                  <div>
                    <p className="text-xs font-mono text-[#2dd4bf] mb-0.5">{item.score}</p>
                    <p className="text-xs text-slate-400">{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
});
PatientProfilePanel.displayName = "PatientProfilePanel";

// ─────────────────────────────────────────────────────────────
// 8. Results View
// ─────────────────────────────────────────────────────────────

const ResultsView = React.memo(({ analysis }: { analysis: AnalysisResult }) => {
  const results = analysis.trial_matches?.eligibility_results ?? [];
  const ranked  = results
    .filter(r => assignTier(r) !== "EXCLUDED")
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

  return (
    <div className="flex flex-col lg:flex-row gap-8 max-w-7xl mx-auto px-8 pt-8">
      <aside className="w-full lg:w-1/3">
        <PatientProfilePanel analysis={analysis} />
      </aside>
      <section className="w-full lg:w-2/3">
        <div className="flex items-end justify-between mb-6 border-b border-white/10 pb-4">
          <div>
            <h2 className="text-2xl font-bold text-white mb-1">Matched Trials</h2>
            <p className="text-xs text-slate-400 font-medium">Ranked by clinical alignment · live from ClinicalTrials.gov</p>
          </div>
          <span className="text-slate-400 font-medium text-sm bg-white/5 px-3 py-1 rounded-full border border-white/10">
            {ranked.length} result{ranked.length !== 1 ? "s" : ""}
          </span>
        </div>

        {ranked.length === 0 ? (
          <div className="rounded-2xl p-8 bg-slate-900/40 border border-white/10 text-center backdrop-blur-md">
            <p className="text-lg font-bold text-white mb-2">No matching trials found for this profile.</p>
            <p className="text-sm text-slate-400 max-w-md mx-auto">
              This can happen if the patient's stage or biomarkers don't align with currently recruiting trials. Try updating any missing fields or broadening the search.
            </p>
          </div>
        ) : (
          <div className="space-y-5">
            {ranked.map((result, i) => (
              <TrialCard key={result.nct_id || i} result={result} index={i} />
            ))}
          </div>
        )}

        {analysis.trial_matches?.final_recommendations && (
          <div className="mt-8 rounded-2xl p-6 bg-[#0f172a]/50 backdrop-blur-xl border border-[#0d9488]/30 shadow-lg relative overflow-hidden">
            <div className="absolute top-0 right-0 p-8 opacity-10 pointer-events-none">
              <OncoPilotLogo size={120} />
            </div>
            <div className="relative z-10">
              <div className="flex items-center gap-3 mb-2">
                <OncoPilotLogo size={20} />
                <h3 className="text-lg font-bold text-white">AI Recommendation Summary</h3>
              </div>
              <p className="text-xs text-[#2dd4bf] uppercase tracking-wider font-bold mb-4">
                Generated by OncoPilot · grounded in trial data · not a substitute for clinical judgment
              </p>
              <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
                {analysis.trial_matches.final_recommendations}
              </p>
            </div>
          </div>
        )}

        <div className="mt-8 p-4 rounded-xl bg-black/40 border border-white/5 backdrop-blur-md">
          <p className="text-xs text-slate-500 leading-relaxed">
            <span className="font-bold text-slate-400">Important: </span>
            OncoPilot is a clinical decision-support tool. All trial matches are screening support only — not a final eligibility determination. Final suitability depends on full protocol review, investigator assessment, and site-specific screening. Always involve a qualified oncologist before acting on these results.
          </p>
        </div>
      </section>
    </div>
  );
});
ResultsView.displayName = "ResultsView";

// ─────────────────────────────────────────────────────────────
// 9. Main Application
// ─────────────────────────────────────────────────────────────

export default function OncoPilotApp() {
  const [file, setFile] = useState<File | null>(null);
  const [inputMode, setInputMode] = useState<InputMode>("upload");
  const [manualProfile, setManualProfile] = useState<PatientProfile>(EMPTY_PROFILE);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (selectedFile: File | undefined) => {
    setError(null);
    if (!selectedFile) return;
    if (!CONFIG.ALLOWED_FILE_TYPES.includes(selectedFile.type as any)) {
      setError("Please select a valid PDF file.");
      return;
    }
    if (selectedFile.size > CONFIG.MAX_FILE_SIZE_BYTES) {
      setError("File exceeds 10MB limit.");
      return;
    }
    setFile(selectedFile);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  }, []);

  const processUpload = async () => {
    if (!file) return;
    setIsProcessing(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${CONFIG.API_BASE}/profile/analyze`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`Analysis failed (${res.status})`);
      const data: AnalysisResult = await res.json();
      setAnalysis(data);
      setShowUpload(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to process medical record.");
    } finally {
      setIsProcessing(false);
    }
  };

  const processManual = async () => {
    setIsProcessing(true);
    setError(null);
    try {
      const res = await fetch(`${CONFIG.API_BASE}/profile/manual`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile: manualProfile }),
      });
      if (!res.ok) throw new Error(`Manual analysis failed (${res.status})`);
      const data: AnalysisResult = await res.json();
      setAnalysis(data);
      setShowUpload(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to process manual entry.");
    } finally {
      setIsProcessing(false);
    }
  };

  const reset = () => {
    setFile(null);
    setManualProfile(EMPTY_PROFILE);
    setInputMode("upload");
    setAnalysis(null);
    setError(null);
    setIsProcessing(false);
    setShowUpload(false);
  };

  const isLanding = !analysis && !isProcessing && !showUpload;

  return (
    <main className="relative min-h-screen font-sans text-slate-200 flex flex-col overflow-x-hidden bg-[#020617]">
      <style dangerouslySetInnerHTML={{__html: `
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        .font-sans { font-family: 'Plus Jakarta Sans', sans-serif; }
        @keyframes marquee {
          0%   { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
        .animate-marquee { display: flex; width: max-content; animation: marquee 30s linear infinite; }
        .animate-marquee:hover { animation-play-state: paused; }
      `}} />

      {/* ── Fixed Background Layer ── */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none bg-[#020617]">
        {/* 1. Base Image - opacity boosted, removed mix-blend to ensure visibility */}
        <img src="/BG.jpg" alt="Background" className="absolute inset-0 w-full h-full object-cover opacity-70" />
        {/* 2. Frosted Glass - strong blur with a lighter dark tint */}
        <div className="absolute inset-0 bg-[#020617]/40 backdrop-blur-lg" />
        {/* 3. Aesthetic Gradient - subtle teal glow in the corners */}
        <div className="absolute inset-0 bg-gradient-to-br from-[#0d9488]/30 via-transparent to-[#020617]/90" />
      </div>

      {/* ── Nav ── */}
      <nav className="sticky top-0 z-50 w-full bg-[#020617]/40 backdrop-blur-xl border-b border-white/10 shadow-lg px-8 py-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <button onClick={reset} className="focus:outline-none flex items-center">
            <OncoPilotWordmark />
          </button>

          <div className="flex items-center gap-4">
            {analysis && (
              <button onClick={reset} className="text-sm font-bold text-slate-300 hover:text-white transition-colors bg-white/10 px-4 py-2 rounded-lg border border-white/10 hover:border-white/20 shadow-sm backdrop-blur-sm">
                ← New Analysis
              </button>
            )}
            {isLanding && (
              <button onClick={() => setShowUpload(true)} className="text-sm bg-[#0d9488] hover:bg-teal-500 text-white px-5 py-2.5 rounded-xl transition-colors font-bold shadow-lg shadow-[#0d9488]/20 hidden sm:block">
                Analyze a patient →
              </button>
            )}
            <span className="text-[10px] font-bold text-slate-400 tracking-widest uppercase hidden md:block">
              Powered by Groq + LLaMA 3.3
            </span>
          </div>
        </div>
      </nav>

      <div className="relative z-10 flex-1 flex flex-col">

        {/* ── Landing Page ── */}
        {isLanding && (
          <div className="flex-1 flex flex-col items-center justify-center pt-20 pb-24">
            
            {/* Hero content */}
            <div className="w-full max-w-5xl mx-auto px-8 text-center flex flex-col items-center gap-8">
              <div className="inline-flex items-center gap-3 px-4 py-1.5 rounded-full bg-white/5 border border-white/10 shadow-lg backdrop-blur-md">
                <OncoPilotLogo size={14} />
                <span className="text-[11px] font-bold text-slate-300">Agentic clinical trial intelligence</span>
              </div>

              <h1 className="text-5xl md:text-7xl font-extrabold leading-[1.1] text-white tracking-tight drop-shadow-lg">
                The right trial for <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#2dd4bf] to-[#0d9488]">every patient.</span>
              </h1>

              <p className="text-xl text-[#2dd4bf] font-semibold tracking-wide">
                Because the right trial changes everything.
              </p>

              <p className="text-lg text-slate-200 leading-relaxed max-w-3xl font-medium drop-shadow-md">
                OncoPilot reads a patient's medical report — cancer type, stage, biomarkers, and treatment history — searches every recruiting trial on ClinicalTrials.gov, and explains every match in plain language. A real AI agent, not a keyword filter.
              </p>

              <div className="flex items-center gap-4 pt-4">
                <button onClick={() => setShowUpload(true)} className="px-8 py-4 bg-[#0d9488] hover:bg-teal-500 text-white font-bold text-lg rounded-xl transition-all shadow-[0_0_20px_rgba(13,148,136,0.3)] hover:shadow-[0_0_30px_rgba(45,212,191,0.4)] hover:-translate-y-0.5">
                  Analyze a patient →
                </button>
                <a href="#how-it-works" className="px-8 py-4 bg-white/5 hover:bg-white/10 text-white font-bold text-lg rounded-xl border border-white/10 transition-all backdrop-blur-sm">
                  See how it works
                </a>
              </div>
            </div>

            {/* ── Trust Strip ── */}
            <div className="w-full max-w-6xl mx-auto px-8 mt-24">
              <div className="grid md:grid-cols-3 gap-6 rounded-2xl bg-[#0f172a]/40 border border-white/10 backdrop-blur-xl p-8 shadow-2xl">
                {[
                  { stat: "400,000+", label: "Trials on ClinicalTrials.gov", sub: "searched live on every query" },
                  { stat: "7", label: "Scoring dimensions", sub: "cancer type, stage, biomarkers, treatment, location, phase, study type" },
                  { stat: "3 tiers", label: "Strong · Moderate · Possible", sub: "every match ranked by clinical strength" },
                ].map((item, i) => (
                  <div key={i} className="text-center md:border-r last:border-0 border-white/10 px-4">
                    <p className="text-4xl font-extrabold text-[#2dd4bf] mb-2 drop-shadow-sm">{item.stat}</p>
                    <p className="text-lg font-bold text-white mb-1">{item.label}</p>
                    <p className="text-sm text-slate-300">{item.sub}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* ── What OncoPilot reads ── */}
            <div className="w-full max-w-6xl mx-auto px-8 mt-24">
              <h2 className="text-2xl font-bold text-center text-white mb-10 drop-shadow-sm">What OncoPilot reads from every patient report</h2>
              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                {[
                  { icon: "◷", label: "Cancer Type & Stage", sub: "NSCLC, Stage IV, metastatic setting" },
                  { icon: "❏", label: "Biomarkers", sub: "EGFR, HER2, BRCA — the molecular flags that guide treatment" },
                  { icon: "◆", label: "Treatment History", sub: "prior lines, what worked, what didn't" },
                  { icon: "◈", label: "Demographics", sub: "age, gender, country — for location-aware matching" },
                ].map((item, i) => (
                  <div key={i} className="rounded-2xl p-6 bg-[#0f172a]/50 border border-white/10 backdrop-blur-xl hover:bg-[#0f172a]/70 transition-colors shadow-lg">
                    <p className="text-3xl mb-4 text-[#2dd4bf]">{item.icon}</p>
                    <p className="text-lg font-bold text-white mb-2">{item.label}</p>
                    <p className="text-sm text-slate-300 leading-relaxed">{item.sub}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Agent graph indicator ── */}
            <div id="how-it-works" className="w-full max-w-6xl mx-auto px-8 mt-32">
              <div className="flex flex-col items-center mb-12">
                <div className="inline-flex items-center gap-3 px-4 py-1.5 rounded-full bg-[#0d9488]/20 border border-[#0d9488]/30 backdrop-blur-md">
                  <p className="text-xs font-bold text-[#2dd4bf] uppercase tracking-widest">Agent pipeline</p>
                  <span className="flex items-center gap-2 text-xs font-bold text-white">
                    <span className="relative flex h-2 w-2"><span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal-400 opacity-75"></span><span className="relative inline-flex rounded-full h-2 w-2 bg-teal-500"></span></span>
                    Live processing
                  </span>
                </div>
              </div>
              
              <div className="grid md:grid-cols-4 gap-6 relative">
                <div className="hidden md:block absolute top-6 left-[10%] right-[10%] h-0.5 bg-gradient-to-r from-[#0d9488]/0 via-[#0d9488]/50 to-[#0d9488]/0"></div>
                {[
                  { step: "01", label: "Extract", desc: "AI reads the PDF and structures the full clinical profile" },
                  { step: "02", label: "Search", desc: "Queries ClinicalTrials.gov live for recruiting trials" },
                  { step: "03", label: "Score", desc: "Scores every trial across 7 clinical dimensions" },
                  { step: "04", label: "Explain", desc: "Surfaces the best matches with plain language reasoning" },
                ].map((node, i) => (
                  <div key={i} className="relative z-10 flex flex-col items-center text-center">
                    <div className="w-12 h-12 rounded-full bg-[#020617] border-2 border-[#2dd4bf] flex items-center justify-center mb-6 shadow-[0_0_20px_rgba(45,212,191,0.3)]">
                      <span className="text-sm font-bold text-white">{node.step}</span>
                    </div>
                    <p className="text-xl font-bold text-white mb-2">{node.label}</p>
                    <p className="text-sm text-slate-300 leading-relaxed max-w-[200px]">{node.desc}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Final CTA ── */}
            <div className="w-full max-w-3xl mx-auto px-8 mt-32 text-center pb-12">
              <h2 className="text-4xl font-extrabold text-white mb-6 drop-shadow-md">Ready to find the right trial?</h2>
              <p className="text-lg text-slate-200 mb-10 drop-shadow-md">Upload a patient PDF and get structured trial matches in under a minute. No setup. No account needed for the demo.</p>
              <button onClick={() => setShowUpload(true)} className="px-10 py-5 bg-[#2dd4bf] hover:bg-teal-300 text-[#020617] font-black text-xl rounded-2xl transition-all shadow-[0_0_30px_rgba(45,212,191,0.4)] hover:-translate-y-1 mb-6">
                Analyze a patient →
              </button>
              <p className="text-xs font-bold text-slate-400">
                🔒 Processed securely · not stored beyond this session · data never leaves your browser unencrypted
              </p>
            </div>
          </div>
        )}

        {/* ── Processing ── */}
        {isProcessing && (
          <div className="flex flex-col items-center justify-center flex-1 min-h-[80vh]">
            <div className="relative w-28 h-28 mb-8">
              <div className="absolute inset-0 animate-spin rounded-full shadow-[0_0_15px_rgba(45,212,191,0.5)]" style={{ border: '3px solid rgba(13, 148, 136, 0.2)', borderTopColor: '#2dd4bf' }} />
              <div className="absolute inset-0 flex items-center justify-center bg-black/40 rounded-full backdrop-blur-md border border-white/10">
                <OncoPilotLogo size={36} />
              </div>
            </div>
            <h2 className="text-3xl font-bold text-white mb-4 drop-shadow-md">Analyzing Patient Record</h2>
            <p className="text-slate-200 text-lg mb-8 max-w-md text-center drop-shadow-md">OncoPilot is running the full 4-node pipeline — this takes about 15–30 seconds.</p>
            
            <div className="w-full max-w-md bg-[#0f172a]/70 backdrop-blur-xl rounded-2xl border border-white/10 p-6 space-y-4 shadow-2xl">
              {[
                { step: "01", text: "Extracting clinical profile from PDF..." },
                { step: "02", text: "Querying ClinicalTrials.gov live..." },
                { step: "03", text: "Scoring biomarker and stage alignment..." },
                { step: "04", text: "Generating plain language recommendations..." },
              ].map((s, i) => (
                <div key={i} className="flex items-center gap-4 text-sm font-medium text-slate-200">
                  <span className="text-xs font-bold text-[#2dd4bf] bg-[#0d9488]/20 px-2 py-1 rounded border border-[#0d9488]/30">{s.step}</span>
                  {s.text}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Upload / Manual Card ── */}
        {showUpload && !isProcessing && !analysis && (
          <div className="flex-1 flex flex-col items-center justify-center pt-10 pb-24 px-4">
            <div className="w-full max-w-2xl bg-[#0f172a]/60 backdrop-blur-2xl rounded-3xl border border-white/10 shadow-2xl p-10 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-[#0d9488] to-[#2dd4bf]"></div>
              
              {/* Toggle Input Mode */}
              <div className="flex justify-center mb-8 relative z-10">
                <div className="inline-flex bg-black/40 p-1 rounded-xl border border-white/10 backdrop-blur-md">
                  <button
                    onClick={() => setInputMode("upload")}
                    className={`px-6 py-2.5 rounded-lg text-sm font-bold transition-colors ${inputMode === "upload" ? "bg-[#0d9488] text-white shadow-md" : "text-slate-400 hover:text-white"}`}
                  >
                    Upload PDF
                  </button>
                  <button
                    onClick={() => setInputMode("manual")}
                    className={`px-6 py-2.5 rounded-lg text-sm font-bold transition-colors ${inputMode === "manual" ? "bg-[#0d9488] text-white shadow-md" : "text-slate-400 hover:text-white"}`}
                  >
                    Enter Manually
                  </button>
                </div>
              </div>

              {inputMode === "upload" ? (
                <>
                  <div className="text-center mb-8">
                    <h2 className="text-3xl font-extrabold text-white mb-3">Upload a patient report</h2>
                    <p className="text-slate-300">Any oncology PDF — discharge summary, pathology note, clinical letter. No template needed.</p>
                  </div>

                  <div
                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    className={`relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 ${
                      isDragging ? "border-[#2dd4bf] bg-[#2dd4bf]/10" : file ? "border-[#0d9488] bg-[#0d9488]/20" : "border-slate-500 hover:border-slate-400 bg-black/40 backdrop-blur-sm"
                    }`}
                  >
                    <input type="file" accept=".pdf" className="hidden" ref={fileInputRef} onChange={(e) => handleFileChange(e.target.files?.[0])} />

                    <div className="flex justify-center mb-4">
                      {file ? (
                        <div className="w-16 h-16 rounded-full bg-[#0d9488]/30 flex items-center justify-center text-[#2dd4bf] shadow-inner">
                          <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        </div>
                      ) : (
                        <div className="w-16 h-16 rounded-full bg-white/10 flex items-center justify-center text-slate-300 shadow-inner">
                          <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                        </div>
                      )}
                    </div>

                    {file ? (
                      <>
                        <p className="text-lg font-bold text-white mb-1 drop-shadow-sm">{file.name}</p>
                        <p className="text-sm text-[#2dd4bf] font-medium">{(file.size / 1024).toFixed(0)} KB · click to change</p>
                      </>
                    ) : (
                      <>
                        <p className="text-lg font-bold text-white mb-1 drop-shadow-sm">Drop patient PDF here</p>
                        <p className="text-sm text-slate-300">or click to browse · max 10MB</p>
                      </>
                    )}
                  </div>

                  {error && (
                    <div className="mt-6 p-4 rounded-xl bg-rose-900/30 border border-rose-800/50 flex items-center gap-3 text-rose-300 text-sm font-bold backdrop-blur-md">
                      <span>⚠</span> {error}
                    </div>
                  )}

                  <button
                    disabled={!file || isProcessing}
                    onClick={processUpload}
                    className="w-full mt-8 px-8 py-4 bg-[#2dd4bf] hover:bg-teal-300 disabled:bg-slate-700 disabled:text-slate-400 disabled:shadow-none text-[#020617] font-black text-lg rounded-xl transition-all shadow-[0_0_20px_rgba(45,212,191,0.3)] hover:shadow-[0_0_30px_rgba(45,212,191,0.5)] hover:-translate-y-0.5"
                  >
                    {isProcessing ? "Analyzing..." : "Analyze & Match Trials →"}
                  </button>
                </>
              ) : (
                <>
                  <div className="text-center mb-4">
                    <h2 className="text-3xl font-extrabold text-white mb-3">Enter details manually</h2>
                    <p className="text-slate-300">Provide structured oncology data directly for trial matching.</p>
                  </div>
                  
                  {error && (
                    <div className="mb-4 p-4 rounded-xl bg-rose-900/30 border border-rose-800/50 flex items-center gap-3 text-rose-300 text-sm font-bold backdrop-blur-md relative z-10">
                      <span>⚠</span> {error}
                    </div>
                  )}

                  <ManualProfileForm 
                    profile={manualProfile} 
                    onChange={setManualProfile} 
                    onSubmit={processManual} 
                    isProcessing={isProcessing} 
                  />
                </>
              )}

              <p className="text-xs text-center font-bold text-slate-400 mt-6 relative z-10">
                🔒 Processed securely · not stored beyond this session · data never leaves your browser unencrypted
              </p>
            </div>
            
            <button onClick={() => setShowUpload(false)} className="mt-8 text-sm font-bold text-slate-300 hover:text-white transition-colors bg-black/40 px-6 py-2 rounded-full border border-white/10 backdrop-blur-md">
              Cancel
            </button>
          </div>
        )}

        {/* ── Results ── */}
        {analysis && !isProcessing && (
          <div className="pb-24">
            <ResultsView analysis={analysis} />
            <FloatingChatWidget profile={analysis.profile} trialMatches={analysis.trial_matches} />
          </div>
        )}
      </div>

      {/* ── Tech Stack Marquee ── */}
      <div className="border-t border-white/10 bg-[#0f172a]/20 backdrop-blur-xl relative py-6 overflow-hidden z-20 mt-auto">
        <div className="relative">
          <div className="animate-marquee gap-16 opacity-50 grayscale hover:grayscale-0 transition-all duration-500 px-8">
            {[...TECH_STACK, ...TECH_STACK].map((tech, i) => (
              <span key={i} className="flex items-center gap-3 text-slate-300 font-bold whitespace-nowrap text-lg">
                <span className="text-[#2dd4bf]">◈</span> {tech}
              </span>
            ))}
          </div>
          <div className="pointer-events-none absolute inset-y-0 left-0 w-32 bg-gradient-to-r from-[#020617] to-transparent" />
          <div className="pointer-events-none absolute inset-y-0 right-0 w-32 bg-gradient-to-l from-[#020617] to-transparent" />
        </div>
      </div>

      {/* ── Footer ── */}
      <footer className="bg-[#020617]/80 backdrop-blur-xl border-t border-white/5 py-8 px-8 relative z-20">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3 opacity-80">
            <OncoPilotLogo size={16} />
            <span className="text-sm font-bold text-slate-300">OncoPilot — clinical trial intelligence</span>
          </div>
          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">
            Signal-driven · plain language · agentic by design · grounded in ClinicalTrials.gov
          </p>
        </div>
      </footer>
    </main>
  );
}