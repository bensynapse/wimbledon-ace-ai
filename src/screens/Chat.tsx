import React, { useState } from 'react';
import { Av } from '../components';

export interface ChatItem {
  from: string;
  av: string;
  last?: string;
  time?: string;
}

export function ChatView({ chat, onBack }: { chat: ChatItem; onBack: () => void }) {
  const [msg, setMsg] = useState('');
  const [msgs, setMsgs] = useState([
    { f: 'them', t: chat.last || 'Hi!', time: chat.time || 'now' },
    { f: 'me', t: 'Hi! Let me check my schedule.', time: '14:25' },
  ]);

  const send = () => {
    if (!msg.trim()) return;
    setMsgs((current) => [
      ...current,
      {
        f: 'me',
        t: msg,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
    setMsg('');
  };

  return (
    <div className="animate-in slide-in-from-right-8 absolute inset-0 z-[5000] flex h-[100dvh] w-full flex-col bg-bg duration-300">
      <div className="sticky top-0 flex items-center gap-2.5 border-b border-border bg-surface/90 px-4 py-3 backdrop-blur-md">
        <button onClick={onBack} className="cursor-pointer border-none bg-transparent px-2 text-lg text-sec">
          ←
        </button>
        <Av t={chat.av} s={32} />
        <div>
          <div className="text-[13px] font-bold text-text">{chat.from}</div>
          <div className="text-[10px] font-semibold text-accent">Online</div>
        </div>
      </div>

      <div className="hide-scrollbar flex flex-1 flex-col gap-1.5 overflow-y-auto p-3.5 pb-20">
        {msgs.map((m, i) => (
          <div key={i} className={`flex max-w-[78%] flex-col ${m.f === 'me' ? 'self-end' : 'self-start'}`}>
            <div
              className={`px-3.5 py-2.5 text-[12.5px] leading-relaxed shadow-sm ${
                m.f === 'me' ? 'rounded-2xl rounded-br-sm bg-accent text-black' : 'rounded-2xl rounded-bl-sm border border-border bg-card text-text'
              }`}
            >
              {m.t}
            </div>
            <div className={`mt-1 text-[9px] text-dim ${m.f === 'me' ? 'text-right' : 'text-left'}`}>{m.time}</div>
          </div>
        ))}
      </div>

      <div className="absolute bottom-0 flex w-full gap-1.5 border-t border-border bg-surface p-3">
        <button className="flex h-10 w-10 items-center justify-center rounded-full border border-border bg-bg text-sm hover:bg-card">📎</button>
        <input
          value={msg}
          onChange={(e) => setMsg(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
          placeholder="Type a message..."
          className="flex-1 rounded-full border border-border bg-bg px-3.5 text-xs text-text outline-none transition-colors focus:border-accent/50"
        />
        <button
          onClick={send}
          disabled={!msg.trim()}
          className="flex h-10 w-10 items-center justify-center rounded-full border-none bg-accent text-sm font-bold text-black disabled:bg-border disabled:opacity-50"
        >
          ↑
        </button>
      </div>
    </div>
  );
}
