import React, { useEffect, useState } from 'react';
import Home from './pages/Home';
import { Leaf, Sun, Moon } from 'lucide-react';

export default function App() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem('theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const shouldDark = stored ? stored === 'dark' : prefersDark;
    setDark(shouldDark);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (dark) {
      root.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      root.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [dark]);

  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-30 border-b border-black/5 bg-white/70 dark:bg-slate-900/60 backdrop-blur supports-[backdrop-filter]:bg-white/60 dark:supports-[backdrop-filter]:bg-slate-900/50">
        <div className="mx-auto max-w-5xl px-4 py-3 sm:px-6 lg:px-8 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center justify-center h-8 w-8 rounded-xl bg-gradient-to-br from-nature-green to-nature-brown text-white shadow-soft">
              <Leaf className="h-4 w-4" />
            </span>
            <h1 className="text-base sm:text-lg font-semibold tracking-tight text-slate-900 dark:text-slate-100">Soil-Veg AI Analyzer</h1>
          </div>
          <button
            onClick={() => setDark((d) => !d)}
            aria-label="Toggle dark mode"
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-1.5 text-slate-700 dark:text-slate-200 shadow-sm hover:shadow transition"
          >
            {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            <span className="hidden sm:inline text-sm">{dark ? 'Light' : 'Dark'}</span>
          </button>
        </div>
      </header>
      <main className="flex-1">
        <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6 lg:px-8">
          <Home />
        </div>
      </main>
    </div>
  );
}

