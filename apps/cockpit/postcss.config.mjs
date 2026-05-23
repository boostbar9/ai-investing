// Tailwind v4 PostCSS plugin. Required for `@import "tailwindcss"` in
// globals.css to actually compile utility classes during `next build`.
export default {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};
