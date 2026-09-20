/**
 * Project N: Clinical RAG & Personal Facts Library Page.
 */

import React, { useCallback, useEffect, useState } from "react";
import { BookOpen, ShieldCheck } from "lucide-react";
import { confirmFact, fetchFacts } from "../api/factsApi";
import { ChildProfileFact, FactCategory } from "../types/facts";
import { FactItemCard } from "../components/facts/FactItemCard";
import { FactFilters } from "../components/facts/FactFilters";
import { CreateFactModal } from "../components/facts/CreateFactModal";
import { NonDiagnosticBanner } from "../components/common/NonDiagnosticBanner";

export const FactsLibraryPage: React.FC = () => {
  const [facts, setFacts] = useState<ChildProfileFact[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedCategory, setSelectedCategory] = useState<FactCategory | "all">("all");
  const [confirmedOnly, setConfirmedOnly] = useState<boolean>(false);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState<boolean>(false);
  const [confirmingId, setConfirmingId] = useState<string | null>(null);

  const loadFacts = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetchFacts({
        category: selectedCategory === "all" ? undefined : selectedCategory,
        confirmed_only: confirmedOnly,
      });
      setFacts(res.facts);
    } catch (err) {
      console.error("Failed to load facts", err);
    } finally {
      setLoading(false);
    }
  }, [selectedCategory, confirmedOnly]);

  useEffect(() => {
    loadFacts();
  }, [loadFacts]);

  const handleConfirm = async (factId: string, confirmed: boolean) => {
    setConfirmingId(factId);
    try {
      await confirmFact(factId, { confirmed });
      await loadFacts();
    } catch (err) {
      console.error("Failed to confirm fact", err);
    } finally {
      setConfirmingId(null);
    }
  };

  const unconfirmedCount = facts.filter((f) => f.confirmed_by_caregiver === 0).length;

  return (
    <div className="flex flex-col gap-5 max-w-5xl mx-auto p-4 sm:p-6 text-slate-100">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-sky-400" />
            <h2 className="text-base font-semibold text-slate-100">
              Clinical RAG & Personal Facts Library
            </h2>
          </div>
          <p className="text-xs text-slate-400">
            Caregiver-curated preferences, sensory triggers, and clinician-suggested OT/SLP techniques.
          </p>
        </div>

        {unconfirmedCount > 0 && (
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-950/70 border border-amber-800 text-amber-300 text-xs font-medium">
            <ShieldCheck className="w-4 h-4 text-amber-400" />
            <span>{unconfirmedCount} Technique{unconfirmedCount === 1 ? "" : "s"} Awaiting Caregiver Confirmation</span>
          </div>
        )}
      </div>

      <FactFilters
        selectedCategory={selectedCategory}
        onSelectCategory={setSelectedCategory}
        confirmedOnly={confirmedOnly}
        onToggleConfirmedOnly={setConfirmedOnly}
        onOpenCreateModal={() => setIsCreateModalOpen(true)}
      />

      {/* Facts List */}
      <div className="flex flex-col gap-3">
        {facts.length === 0 && !loading && (
          <div className="p-8 rounded-xl bg-slate-900/60 border border-slate-800 text-center text-slate-500 text-xs">
            No profile facts found for the selected category.
          </div>
        )}

        {facts.map((fact) => (
          <FactItemCard
            key={fact.id}
            fact={fact}
            onConfirm={handleConfirm}
            confirming={confirmingId === fact.id}
          />
        ))}
      </div>

      {isCreateModalOpen && (
        <CreateFactModal
          isOpen={isCreateModalOpen}
          onClose={() => setIsCreateModalOpen(false)}
          onCreated={loadFacts}
        />
      )}

      <NonDiagnosticBanner />
    </div>
  );
};
