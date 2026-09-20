/**
 * Project N: Outcome State Badge Component.
 */

import React from "react";
import { OutcomeState } from "../../types/episodes";
import { Badge, BadgeVariant } from "../common/Badge";

interface OutcomeBadgeProps {
  outcome: OutcomeState | null;
  className?: string;
}

const outcomeConfig: Record<OutcomeState, { label: string; variant: BadgeVariant }> = {
  settled_immediately: { label: "Settled Immediately", variant: "success" },
  settled_delayed: { label: "Settled (Delayed)", variant: "info" },
  no_change: { label: "No Change", variant: "default" },
  escalated: { label: "Escalated", variant: "danger" },
};

export const OutcomeBadge: React.FC<OutcomeBadgeProps> = ({ outcome, className = "" }) => {
  if (!outcome) {
    return (
      <Badge variant="default" size="sm" className={className}>
        Unconfirmed
      </Badge>
    );
  }

  const config = outcomeConfig[outcome] || { label: outcome, variant: "default" };
  return (
    <Badge variant={config.variant} size="sm" className={className}>
      {config.label}
    </Badge>
  );
};
