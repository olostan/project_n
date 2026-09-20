/**
 * Project N: Navigation Sidebar Component.
 */

import React from "react";
import {
  Activity,
  BookOpen,
  Calendar,
  Compass,
  FileCheck,
  Film,
  HeartPulse,
} from "lucide-react";

export type ActiveTab =
  | "telemetry"
  | "diary"
  | "inspector"
  | "triage"
  | "lexicon"
  | "facts"
  | "promotion";

interface SidebarNavProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  unconfirmedFactsCount?: number;
  activeTasksCount?: number;
}

interface NavItem {
  id: ActiveTab;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: number;
}

export const SidebarNav: React.FC<SidebarNavProps> = ({
  activeTab,
  onSelectTab,
  unconfirmedFactsCount = 0,
  activeTasksCount = 0,
}) => {
  const items: NavItem[] = [
    { id: "telemetry", label: "Live Telemetry", icon: Activity, badge: activeTasksCount },
    { id: "diary", label: "Episode Diary", icon: Calendar },
    { id: "inspector", label: "Video Inspector", icon: Film },
    { id: "triage", label: "Pain & NCCPC Triage", icon: HeartPulse },
    { id: "lexicon", label: "2D Lexicon Map", icon: Compass },
    { id: "facts", label: "Clinical RAG & Facts", icon: BookOpen, badge: unconfirmedFactsCount },
    { id: "promotion", label: "Model Promotion", icon: FileCheck },
  ];

  return (
    <aside className="w-56 bg-slate-900/90 border-r border-slate-800 p-3 flex flex-col justify-between shrink-0">
      <nav className="flex flex-col gap-1">
        <div className="px-3 py-2 text-[10px] uppercase font-semibold text-slate-500 tracking-wider">
          Navigation
        </div>
        {items.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                isActive
                  ? "bg-sky-600/20 text-sky-400 border border-sky-500/30"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <div className="flex items-center gap-2.5">
                <Icon className={`w-4 h-4 ${isActive ? "text-sky-400" : "text-slate-400"}`} />
                <span>{item.label}</span>
              </div>
              {item.badge && item.badge > 0 ? (
                <span className="px-1.5 py-0.5 rounded-full bg-sky-950 text-sky-400 border border-sky-800 text-[10px] font-mono">
                  {item.badge}
                </span>
              ) : null}
            </button>
          );
        })}
      </nav>

      {/* Safety Invariant badge footer */}
      <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80 text-[11px] text-slate-500 flex flex-col gap-1">
        <div className="flex items-center justify-between text-slate-400 font-medium">
          <span>Local Enclave</span>
          <span className="text-[10px] text-emerald-400">100% Offline</span>
        </div>
        <p className="text-[10px] text-slate-500 leading-tight">
          Zero cloud exfiltration. AES-256 encrypted vault.
        </p>
      </div>
    </aside>
  );
};
