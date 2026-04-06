import React, { useState } from 'react';
import { allSports, demoDisputes, demoMsgs, demoRefs, earningsData, months, Referee } from './data';
import { ChatItem, ChatView } from './screens/Chat';
  const [chat, setChat] = useState<ChatItem | null>(null);
  const [subView, setSubView] = useState<string | null>(null);
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
      {chat && <ChatView chat={chat} onBack={() => setChat(null)} />}
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
        {view === 'home' && !selRef && !subView && (
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

            <div className="mb-3.5 grid grid-cols-3 gap-1.5">
              {[
                ['📋', 'Post Match', () => {
                  setView('post');
                  setSubView(null);
                }],
                ['⚖️', 'Disputes', () => setSubView('disputes')],
                ['📄', 'Invoices', () => setSubView('invoices')],
              ].map(([icon, label, action], idx) => (
                <button
                  key={idx}
                  onClick={action as () => void}
                  className="cursor-pointer rounded-xl border border-border bg-card p-3 text-center transition-colors hover:bg-cardAlt"
                >
                  <div className="mb-0.5 text-lg">{icon as string}</div>
                  <div className="text-[10px] font-semibold text-text">{label as string}</div>
                </button>
              ))}
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
              <Btn
                full
                v="secondary"
                onClick={() => setChat({ from: selRef.name, av: selRef.av, last: "Hi, I'd like to discuss the match.", time: 'now' })}
              >
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
                onClick={() => setChat(m)}
        {view === 'profile' && !subView && (
                ['📊', 'Analytics', 'Bekijk je statistieken'],
                  onClick={() => {
                    if (title === 'Analytics') setSubView('analytics');
                    if (title === 'Invoices & Tax') setSubView('invoices');
                    if (title === 'Disputes') setSubView('disputes');
                  }}

        {subView === 'analytics' && (
          <div className="animate-in slide-in-from-right-8 absolute inset-0 z-50 bg-bg p-4 duration-300">
            <div className="mt-2 mb-4 flex items-center gap-2.5">
              <button onClick={() => setSubView(null)} className="cursor-pointer border-none bg-transparent text-base text-sec">
                ←
              </button>
              <h2 className="m-0 text-[17px] font-bold text-text">Analytics</h2>
            </div>
            <div className="mb-3.5 rounded-xl border border-border bg-card p-3">
              <div className="mb-1.5 flex items-center justify-between">
                <span className="text-[11px] font-bold text-text">Revenue</span>
                <span className="text-[11px] font-semibold text-accent">€5,460</span>
              </div>
              <MiniChart data={earningsData} h={80} />
              <div className="mt-2 flex justify-between">
                {months.map((m) => (
                  <span key={m} className="text-[7px] text-dim">
                    {m}
                  </span>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <StatCard v="156" l="Matches" />
              <StatCard v="4.8" l="Avg Rating" c="text-star" />
              <StatCard v="98%" l="Response Rate" c="text-info" />
              <StatCard v="2,340" l="Km Driven" />
            </div>
          </div>
        )}

        {subView === 'disputes' && (
          <div className="animate-in slide-in-from-right-8 absolute inset-0 z-50 bg-bg p-4 duration-300">
            <div className="mt-2 mb-4 flex items-center gap-2.5">
              <button onClick={() => setSubView(null)} className="cursor-pointer border-none bg-transparent text-base text-sec">
                ←
              </button>
              <h2 className="m-0 text-[17px] font-bold text-text">Disputes</h2>
            </div>
            {demoDisputes.map((d) => (
              <div key={d.id} className="mb-2 rounded-xl border border-border bg-card p-3.5">
                <div className="mb-1.5 flex justify-between">
                  <span className="text-xs font-bold text-text">{d.match}</span>
                  <Bdg
                    c={d.status === 'open' ? 'text-warning' : 'text-accent'}
                    bg={d.status === 'open' ? 'bg-warning/10' : 'bg-accent/10'}
                  >
                    {d.status}
                  </Bdg>
                </div>
                <div className="mb-2 text-[11px] text-sec">{d.desc}</div>
                <div className="mt-2 flex items-center justify-between border-t border-border pt-2">
                  <span className="text-[10px] text-dim">
                    📅 {d.date} • {d.type}
                  </span>
                  {d.status === 'open' && (
                    <Btn sm v="outline" onClick={() => showToast('Response sent')}>
                      Respond
                    </Btn>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {subView === 'invoices' && (
          <div className="animate-in slide-in-from-right-8 absolute inset-0 z-50 overflow-y-auto bg-bg p-4 pb-20 duration-300">
            <div className="mt-2 mb-4 flex items-center gap-2.5">
              <button onClick={() => setSubView(null)} className="cursor-pointer border-none bg-transparent text-base text-sec">
                ←
              </button>
              <h2 className="m-0 text-[17px] font-bold text-text">Invoices & Tax</h2>
            </div>
            {[
              { id: 'INV-2026-047', club: 'SC Eibergen', date: '12 Apr', amount: '€46.30', status: 'paid' },
              { id: 'INV-2026-046', club: 'HC Groenlo', date: '6 Apr', amount: '€52.62', status: 'paid' },
              { id: 'INV-2026-045', club: 'HV Achterhoek', date: '29 Mar', amount: '€38.05', status: 'pending' },
            ].map((inv) => (
              <div
                key={inv.id}
                className="mb-1.5 flex items-center justify-between rounded-xl border border-border bg-card p-3 transition-colors hover:bg-cardAlt"
              >
                <div>
                  <div className="text-xs font-bold text-text">{inv.club}</div>
                  <div className="text-[10px] text-sec">
                    {inv.id} • {inv.date}
                  </div>
                </div>
                <div className="text-right">
                  <div className="mb-0.5 text-[13px] font-bold text-accent">{inv.amount}</div>
                  <Bdg c={inv.status === 'paid' ? 'text-accent' : 'text-warning'} bg={inv.status === 'paid' ? 'bg-accent/10' : 'bg-warning/10'}>
                    {inv.status}
                  </Bdg>
                </div>
              </div>
            ))}

            <div className="mt-3.5 grid grid-cols-2 gap-2">
              <button
                onClick={() => showToast('PDF downloaded')}
                className="cursor-pointer rounded-xl border border-border bg-card p-3.5 text-center hover:bg-cardAlt"
              >
                <div className="text-lg">📄</div>
                <div className="mt-1 text-[10px] font-semibold text-text">Download Invoice PDF</div>
              </button>
              <button
                onClick={() => showToast('CSV exported')}
                className="cursor-pointer rounded-xl border border-border bg-card p-3.5 text-center hover:bg-cardAlt"
              >
                <div className="text-lg">📊</div>
                <div className="mt-1 text-[10px] font-semibold text-text">Export Year Report (CSV)</div>
              </button>
            </div>

            <div className="mt-3 rounded-xl border border-border bg-surface p-3.5">
              <h4 className="m-0 mb-1.5 text-xs font-bold text-text">2026 Tax Summary</h4>
              {[
                ['Total earned', '€5,460'],
                ['Total travel (km)', '2,340 km'],
                ['Travel deduction', '€491.40'],
                ['Matches officiated', '156'],
                ['Platform fees paid', '€780'],
              ].map(([label, value], i) => (
                <div key={i} className={`flex justify-between py-1 text-[11px] text-sec ${i < 4 ? 'border-b border-border' : ''}`}>
                  <span>{label}</span>
                  <span className="font-semibold text-text">{value}</span>
                </div>
              ))}
            </div>
          </div>
        )}
              setSubView(null);
