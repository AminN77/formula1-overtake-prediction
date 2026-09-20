import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        f1: {
          bg: "#15151e",
          surface: "#1e1e2e",
          card: "#2d2d3d",
          red: "#e10600",
          muted: "#8b949e",
        },
      },
      fontFamily: {
        display: ["Titillium Web", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
} satisfies Config;
