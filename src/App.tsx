import React, { useState } from 'react';
import { allSports, demoRefs, earningsData, months, Referee } from './data';
import { Av, Bdg, Btn, Stars, StatCard } from './components';

const MiniChart = ({ data, h = 60 }: { data: number[]; h?: number }) => {
  const max = Math.max(...data);
  return (
    <div className="flex items-end gap-1" style={{ height: h }}>
      {data.map((v, i) => (
        <div
          key={i}
          className={`flex-1 rounded-sm transition-all duration-300 ${i === data.length - 1 ? 'bg-accent' : 'bg-accent/40'}`}
          style={{ height: `${(v / max) * 100}%`, minHeight: 2 }}
        />
      ))}
    </div>
  );
};

export default function App() {
  const [auth, setAuth] = useState(false);
  const [view, setView] = useState('home');
  const [selRef, setSelRef] = useState<Referee | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [refs, setRefs] = useState(demoRefs);
  const [sosActive, setSos] = useState(false);

  const showToast = (m: string) => {
    setToast(m);
    setTimeout(() => setToast(null), 2500);
  };

  const toggleFav = (id: number) => {
    setRefs((current) => current.map((r) => (r.id === id ? { ...r, fav: !r.fav } : r)));
    showToast('Favorieten geüpdatet');
  };

  if (!auth) {
    return (
      <div className="mx-auto flex min-h-[100dvh] max-w-[430px] flex-col items-center justify-center bg-bg p-7 text-center font-sans">
        <div className="mb-4 flex h-20 w-20 items-center justify-center rounded-3xl border-2 border-accent/30 bg-accent/10 text-4xl shadow-[0_0_15px_rgba(0,230,118,0.2)]">🏆</div>
        <h1 className="m-0 mb-1 text-[26px] font-extrabold tracking-tight text-text">
          Ref<span className="text-accent">Connect</span>
        </h1>
        <p className="m-0 mb-6 text-xs text-sec">The world's first multi-sport referee marketplace</p>
        <div className="mb-8 flex gap-1.5 text-lg opacity-70">
          <span>⚽</span>
          <span>🏑</span>
          <span>🤾</span>
          <span>🏀</span>
          <span>🏐</span>
          <span>🎾</span>
        </div>
        <div className="flex w-full max-w-[280px] flex-col gap-2.5">
          <Btn full onClick={() => setAuth(true)}>
            🏁 I'm a Referee
          </Btn>
          <Btn full v="secondary" onClick={() => setAuth(true)}>
            🏟️ I'm a Club / Team
          </Btn>
        </div>
      </div>
    );
  }

  const SOS = sosActive && (
    <div className="fixed inset-0 z-[3000] flex flex-col items-center justify-center bg-error/95 p-7 backdrop-blur-sm">
      <div className="mb-4 text-[64px] drop-shadow-xl">🆘</div>
      <h2 className="m-0 mb-2 text-[22px] font-extrabold text-white">Emergency Alert Sent</h2>
      <p className="mb-6 text-center text-[13px] leading-relaxed text-white/80">
        Your location has been shared with RefConnect support.
        <br />
        Local authorities can be contacted if needed.
      </p>
      <div className="mt-4 flex w-full max-w-[280px] gap-2.5">
        <button
          onClick={() => setSos(false)}
          className="flex-1 rounded-xl border-2 border-white/40 bg-transparent py-3.5 text-[13px] font-bold text-white"
        >
          I'm Safe
        </button>
        <button className="flex-1 rounded-xl border-none bg-white py-3.5 text-[13px] font-bold text-error">Call 112</button>
      </div>
    </div>
  );

  return (
    <div className="relative mx-auto min-h-[100dvh] max-w-[430px] overflow-x-hidden bg-bg pb-20 font-sans">
      {SOS}
      {toast && (
        <div className="fixed left-1/2 top-11 z-[2000] -translate-x-1/2 rounded-full bg-accent px-5 py-2 text-xs font-bold text-black shadow-lg">
          {toast}
        </div>
      )}

      <div className="sticky top-0 z-50 flex items-center justify-between bg-bg/80 px-4.5 pb-2 pt-3 backdrop-blur-md">
        <div>
          <h1 className="m-0 text-xl font-extrabold tracking-tight text-text">
            Ref<span className="text-accent">Connect</span>
          </h1>
          <p className="m-0 text-[10px] text-sec">Multi-sport marketplace</p>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setSos(true)}
            className="flex h-[34px] w-[34px] items-center justify-center rounded-lg border-2 border-error bg-error/10 text-[11px] font-extrabold text-error"
          >
            SOS
          </button>
          <Av t="OT" s={34} />
        </div>
      </div>

      <div className="px-4.5 pt-2">
        {view === 'home' && !selRef && (
          <>
            <div className="mb-3.5 grid grid-cols-4 gap-1.5">
              <StatCard v="47" l="Bookings" />
              <StatCard v="€235" l="Revenue" c="text-star" />
              <StatCard v="23" l="Referees" c="text-info" />
              <StatCard v="87%" l="Fill Rate" c="text-teal-500" />
            </div>

            <div className="mb-3.5 rounded-xl border border-border bg-card p-3 shadow-sm">
              <div className="mb-1.5 flex items-center justify-between">
                <span className="text-[11px] font-bold text-text">Revenue (12 months)</span>
                <span className="text-[11px] font-semibold text-accent">
                  €{earningsData.reduce((a, b) => a + b, 0).toLocaleString()}
                </span>
              </div>
              <MiniChart data={earningsData} />
              <div className="mt-1 flex justify-between">
                {months.map((m) => (
                  <span key={m} className="text-[7px] text-dim">
                    {m}
                  </span>
                ))}
              </div>
            </div>

            <div className="mb-4">
              <div className="mb-2 flex items-center justify-between">
                <h3 className="m-0 text-sm font-bold text-text">Top Rated Referees</h3>
              </div>
              <div className="hide-scrollbar flex gap-2 overflow-x-auto pb-1">
                {refs.map((r) => (
                  <div
                    key={r.id}
                    onClick={() => setSelRef(r)}
                    className="relative min-w-[120px] cursor-pointer rounded-xl border border-border bg-card p-3 text-center transition-colors hover:bg-cardAlt"
                  >
                    {r.fav && <span className="absolute right-1.5 top-1.5 text-xs">❤️</span>}
                    <div className="flex justify-center">
                      <Av t={r.av} s={40} />
                    </div>
                    <div className="mt-1.5 text-xs font-bold text-text">{r.name.split(' ')[0]}</div>
                    <div className="mt-0.5 flex items-center justify-center gap-0.5">
                      <Stars r={r.rating} s={10} />
                    </div>
                    <div className="mt-0.5 text-[10px] font-semibold text-accent">€{r.tarief}</div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {view === 'home' && selRef && (
          <div className="animate-in slide-in-from-bottom-4 fade-in duration-300">
            <div className="mb-3.5 flex items-center justify-between">
              <button
                onClick={() => setSelRef(null)}
                className="flex cursor-pointer items-center gap-1 border-none bg-transparent text-base text-sec"
              >
                ← Back
              </button>
              <button
                onClick={() => toggleFav(selRef.id)}
                className="cursor-pointer border-none bg-transparent text-[22px]"
              >
                {selRef.fav ? '❤️' : '🤍'}
              </button>
            </div>

            <div className="mb-4 text-center">
              <div className="flex justify-center">
                <Av t={selRef.av} s={68} />
              </div>
              <h2 className="my-2 text-[19px] font-bold text-text">{selRef.name}</h2>
              <div className="flex items-center justify-center gap-1">
                <Stars r={selRef.rating} />
                <span className="text-xs text-sec">
                  {selRef.rating} ({selRef.reviews})
                </span>
              </div>
              <div className="mt-1.5 flex flex-wrap justify-center gap-1">
                {selRef.idVerified && (
                  <Bdg c="text-info" border="border-info/20" bg="bg-info/10">
                    ✓ ID Verified
                  </Bdg>
                )}
                {selRef.badges.map((b) => (
                  <Bdg
                    key={b}
                    c={b === 'Top Rated' ? 'text-star' : 'text-accent'}
                    bg={b === 'Top Rated' ? 'bg-star/10' : 'bg-accent/10'}
                  >
                    {b}
                  </Bdg>
                ))}
              </div>
            </div>

            <div className="mb-2.5 rounded-xl border border-border bg-card p-3">
              <p className="m-0 text-xs italic leading-relaxed text-sec">"{selRef.bio}"</p>
            </div>

            <div className="mb-2.5 grid grid-cols-4 gap-1.5">
              <StatCard v={selRef.matches} l="Matches" c="text-text" />
              <StatCard v={`${selRef.resp}%`} l="Response" c="text-text" />
              <StatCard v={selRef.respTime} l="Reply" c="text-text" />
              <StatCard v={`€${selRef.tarief}`} l="Fee" c="text-accent" />
            </div>

            <div className="mb-3.5 rounded-xl border border-border bg-card p-3">
              <div className="mb-2 text-[11px] font-bold text-text">💰 Cost Breakdown</div>
              <div className="flex justify-between py-1 text-[11px] text-sec">
                <span>Match fee</span>
                <span>€{selRef.tarief}</span>
              </div>
              <div className="flex justify-between py-1 text-[11px] text-sec">
                <span>Travel ({selRef.km}km)</span>
                <span>€{(selRef.km * 0.21).toFixed(2)}</span>
              </div>
              <div className="flex justify-between py-1 text-[11px] text-sec">
                <span>Platform fee</span>
                <span>€5.00</span>
              </div>
              <div className="mt-1.5 flex items-center justify-between border-t border-border pt-1.5">
                <span className="text-xs font-bold text-text">Total Escrow</span>
                <span className="text-[15px] font-extrabold text-accent">
                  €{(selRef.tarief + selRef.km * 0.21 + 5).toFixed(2)}
                </span>
              </div>
            </div>

            <div className="flex gap-2">
              <Btn full v="secondary">
                💬 Chat
              </Btn>
              <Btn
                full
                onClick={() => {
                  setSelRef(null);
                  showToast('✓ Booking confirmed! Escrow locked.');
                }}
              >
                Book Now →
              </Btn>
            </div>
          </div>
        )}
      </div>

      <div className="fixed bottom-0 left-1/2 flex w-full max-w-[430px] -translate-x-1/2 border-t border-border bg-surface/95 pb-5 pt-1.5 backdrop-blur-md z-[100]">
        {[
          { id: 'home', l: 'Home', i: '🏠' },
          { id: 'search', l: 'Search', i: '🔍' },
          { id: 'post', l: 'Post', i: '➕', special: true },
          { id: 'inbox', l: 'Inbox', i: '💬' },
          { id: 'profile', l: 'Profile', i: '👤' },
        ].map((n) => (
          <button
            key={n.id}
            onClick={() => {
              setView(n.id);
              setSelRef(null);
            }}
            className="relative flex flex-1 cursor-pointer flex-col items-center gap-[2px] border-none bg-transparent py-1"
          >
            <span className={n.special ? 'rounded-lg bg-accent px-3 py-0.5 text-xl text-black' : 'text-base'}>{n.i}</span>
            <span className={`text-[9px] font-semibold ${view === n.id ? 'text-accent' : 'text-sec'}`}>{n.l}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
