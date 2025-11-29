/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#2196F3',
        success: '#4CAF50',
        warning: '#FFC107',
        danger: '#F44336',
      },
    },
  },
  plugins: [],
}