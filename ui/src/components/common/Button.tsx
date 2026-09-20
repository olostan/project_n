/**
 * Project N: Accessible Button Component.
 */

import React from "react";

export type ButtonVariant = "primary" | "secondary" | "danger" | "ghost" | "outline";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: "sm" | "md" | "lg";
  loading?: boolean;
}

const variantStyles: Record<ButtonVariant, string> = {
  primary: "bg-sky-600 hover:bg-sky-500 text-white font-medium shadow-sm focus:ring-sky-500",
  secondary: "bg-slate-700 hover:bg-slate-600 text-slate-200 border border-slate-600 focus:ring-slate-400",
  danger: "bg-rose-600 hover:bg-rose-500 text-white font-medium focus:ring-rose-500",
  ghost: "bg-transparent hover:bg-slate-800 text-slate-300 focus:ring-slate-400",
  outline: "bg-transparent hover:bg-slate-800 text-sky-400 border border-sky-600/60 focus:ring-sky-500",
};

const sizeStyles = {
  sm: "px-2.5 py-1.5 text-xs rounded-md",
  md: "px-3.5 py-2 text-sm rounded-lg",
  lg: "px-4 py-2.5 text-base rounded-lg",
};

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = "primary",
  size = "md",
  loading = false,
  disabled = false,
  className = "",
  ...props
}) => {
  return (
    <button
      disabled={disabled || loading}
      className={`inline-flex items-center justify-center transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 disabled:opacity-50 disabled:cursor-not-allowed ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
      {...props}
    >
      {loading ? (
        <>
          <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-current" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
          Loading...
        </>
      ) : (
        children
      )}
    </button>
  );
};
