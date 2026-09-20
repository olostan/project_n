/**
 * Project N: Reusable Status & Category Badge Component.
 */

import React from "react";

export type BadgeVariant =
  | "default"
  | "success"
  | "warning"
  | "danger"
  | "info"
  | "purple";

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  size?: "sm" | "md";
  className?: string;
}

const variantStyles: Record<BadgeVariant, string> = {
  default: "bg-slate-800 text-slate-300 border-slate-700",
  success: "bg-emerald-950/70 text-emerald-300 border-emerald-800/80",
  warning: "bg-amber-950/70 text-amber-300 border-amber-800/80",
  danger: "bg-rose-950/70 text-rose-300 border-rose-800/80",
  info: "bg-sky-950/70 text-sky-300 border-sky-800/80",
  purple: "bg-purple-950/70 text-purple-300 border-purple-800/80",
};

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = "default",
  size = "md",
  className = "",
}) => {
  const sizeClass = size === "sm" ? "px-1.5 py-0.5 text-xs" : "px-2.5 py-1 text-xs";

  return (
    <span
      className={`inline-flex items-center font-medium rounded-full border ${sizeClass} ${variantStyles[variant]} ${className}`}
    >
      {children}
    </span>
  );
};
