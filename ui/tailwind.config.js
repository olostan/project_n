/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      colors: {
        background: "#0b0f19",
        surface: {
          50: "#1e293b",
          100: "#172033",
          200: "#0f172a",
        },
      },
    },
  },
  plugins: [],
};
