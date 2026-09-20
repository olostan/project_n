/**
 * Project N: Root Application Shell Component.
 */

import React, { useState } from "react";
import { useSSEStream } from "./hooks/useSSEStream";
import { useSystemTelemetry } from "./hooks/useSystemTelemetry";
import { useEpisodes } from "./hooks/useEpisodes";
import { TopHeaderBar } from "./components/layout/TopHeaderBar";
import { ActiveTab, SidebarNav } from "./components/layout/SidebarNav";
import { LiveTelemetryPage } from "./pages/LiveTelemetryPage";
import { DiaryPage } from "./pages/DiaryPage";
import { InspectorPage } from "./pages/InspectorPage";
import { PainTriagePage } from "./pages/PainTriagePage";
import { LexiconPage } from "./pages/LexiconPage";
import { FactsLibraryPage } from "./pages/FactsLibraryPage";
import { PromotionGatePage } from "./pages/PromotionGatePage";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>("telemetry");
  const [triageEpisodeId, setTriageEpisodeId] = useState<string | null>(null);

  // Global subscriptions and queries
  const sseState = useSSEStream();
  const { telemetry } = useSystemTelemetry(sseState.latestTelemetry);
  const {
    episodes,
    total,
    loading,
    antecedentFilter,
    setAntecedentFilter,
    selectedEpisodeId,
    setSelectedEpisodeId,
    selectedEpisode,
    refetch: refetchEpisodes,
  } = useEpisodes();

  const handleOpenTriage = (episodeId: string) => {
    setTriageEpisodeId(episodeId);
    setActiveTab("triage");
  };

  const handleSelectFromLexicon = (episodeId: string) => {
    setSelectedEpisodeId(episodeId);
    setActiveTab("inspector");
  };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#0b0f19] text-slate-100 font-sans">
      {/* Top Header with live VRAM and status */}
      <TopHeaderBar telemetry={telemetry} sseConnected={sseState.connected} />

      {/* Main Viewport */}
      <div className="flex flex-1 overflow-hidden">
        {/* Navigation Sidebar */}
        <SidebarNav
          activeTab={activeTab}
          onSelectTab={(tab) => {
            setActiveTab(tab);
          }}
          activeTasksCount={Object.keys(sseState.activeTasks).length}
        />

        {/* Content Viewport */}
        <main className="flex-1 overflow-y-auto bg-slate-950/40">
          {activeTab === "telemetry" && (
            <LiveTelemetryPage telemetry={telemetry} sseState={sseState} />
          )}

          {activeTab === "diary" && (
            <DiaryPage
              episodes={episodes}
              total={total}
              loading={loading}
              antecedentFilter={antecedentFilter}
              onSelectAntecedent={setAntecedentFilter}
              selectedEpisodeId={selectedEpisodeId}
              onSelectEpisodeId={setSelectedEpisodeId}
              onRefresh={refetchEpisodes}
              onNavigateToInspector={() => setActiveTab("inspector")}
              onNavigateToTriage={handleOpenTriage}
            />
          )}

          {activeTab === "inspector" && (
            <InspectorPage
              episode={selectedEpisode}
              cardData={sseState.latestFourLayerCard}
              onBackToDiary={() => setActiveTab("diary")}
              onNavigateToTriage={handleOpenTriage}
              onEpisodeUpdated={refetchEpisodes}
            />
          )}

          {activeTab === "triage" && (
            <PainTriagePage
              episodeId={triageEpisodeId || selectedEpisodeId || "ep_current"}
              onBack={() => setActiveTab("diary")}
              onSubmitted={refetchEpisodes}
            />
          )}

          {activeTab === "lexicon" && (
            <LexiconPage
              episodes={episodes}
              onSelectEpisodeForInspector={handleSelectFromLexicon}
            />
          )}

          {activeTab === "facts" && <FactsLibraryPage />}

          {activeTab === "promotion" && <PromotionGatePage />}
        </main>
      </div>
    </div>
  );
};
