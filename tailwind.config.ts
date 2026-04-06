import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#05080A',
        surface: '#0D1117',
        card: '#161B22',
        cardAlt: '#1C2333',
        accent: '#00E676',
        accentDark: '#00A152',
        text: '#E6EDF3',
        sec: '#8B949E',
        dim: '#484F58',
        border: '#21262D',
        error: '#F85149',
        info: '#58A6FF',
        star: '#E3B341',
      },
    },
  },
  plugins: [],
};

export default config;
