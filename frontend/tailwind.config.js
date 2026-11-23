/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#071120',
          900: '#0B192C',
          800: '#102542',
          700: '#1B3A63',
          600: '#274F82',
        },
        clinical: {
          bg: '#F7FAFD',
          surface: '#FFFFFF',
          border: '#E2ECF5',
        },
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'Inter', 'system-ui', '-apple-system', 'sans-serif'],
        display: ['"Plus Jakarta Sans"', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        glass: '0 8px 32px rgba(11, 25, 44, 0.08)',
        'glass-lg': '0 20px 50px rgba(11, 25, 44, 0.12)',
        glow: '0 0 36px rgba(6, 182, 212, 0.22)',
      },
      backgroundImage: {
        'hero-mesh':
          'radial-gradient(at 15% 18%, rgba(37, 99, 235, 0.16) 0px, transparent 48%), radial-gradient(at 88% 12%, rgba(6, 182, 212, 0.18) 0px, transparent 46%), radial-gradient(at 50% 95%, rgba(14, 165, 233, 0.10) 0px, transparent 50%)',
      },
    },
  },
  plugins: [],
};
