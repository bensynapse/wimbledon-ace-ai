import React from 'react';

export const Av = ({ t, s = 44, g = true }: { t: string; s?: number; g?: boolean }) => (
  <div
    className={`flex shrink-0 items-center justify-center rounded-[28%] font-bold text-white ${g ? 'bg-gradient-to-br from-accentDark to-accent' : 'bg-card'}`}
    style={{ width: s, height: s, fontSize: s * 0.3 }}
  >
    {t}
  </div>
);

export const Stars = ({ r, s = 13 }: { r: number; s?: number }) => (
  <span className="inline-flex gap-[1px]" style={{ fontSize: s }}>
    {[1, 2, 3, 4, 5].map((i) => (
      <span key={i} className={i <= Math.round(r) ? 'text-star' : 'text-border'}>
        ★
      </span>
    ))}
  </span>
);

export const Bdg = ({
  children,
  c = 'text-accent',
  bg = 'bg-accent/10',
  border = 'border-accent/20',
}: {
  children: React.ReactNode;
  c?: string;
  bg?: string;
  border?: string;
}) => (
  <span className={`inline-block rounded-2xl border px-2 py-0.5 text-[9px] font-bold tracking-wide ${c} ${bg} ${border}`}>
    {children}
  </span>
);

export const Inp = ({
  label,
  value,
  onChange,
  placeholder,
  type = 'text',
  icon,
  disabled,
}: {
  label?: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  type?: string;
  icon?: React.ReactNode;
  disabled?: boolean;
}) => (
  <div className="mb-3">
    {label && <label className="mb-1 block text-[11px] font-semibold text-sec">{label}</label>}
    <div className={`flex items-center gap-2 rounded-xl border border-border bg-surface px-3 ${disabled ? 'opacity-50' : 'opacity-100'}`}>
      {icon && <span className="text-sm">{icon}</span>}
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        className="w-full flex-1 border-none bg-transparent py-2.5 text-[13px] text-text outline-none"
      />
    </div>
  </div>
);

export const Btn = ({
  children,
  onClick,
  full,
  v = 'primary',
  sm,
  dis,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  full?: boolean;
  v?: 'primary' | 'secondary' | 'danger' | 'outline';
  sm?: boolean;
  dis?: boolean;
}) => {
  const styles = {
    primary: 'border-none bg-gradient-to-br from-accentDark to-accent text-white',
    secondary: 'border border-border bg-transparent text-text',
    danger: 'border border-error/30 bg-error/10 text-error',
    outline: 'border border-accent/40 bg-transparent text-accent',
  };

  return (
    <button
      disabled={dis}
      onClick={onClick}
      className={`${styles[v]} ${sm ? 'px-3.5 py-[7px] text-[11px]' : 'px-5 py-3 text-[13px]'} rounded-xl font-bold tracking-wide transition-opacity ${full ? 'w-full' : 'w-auto'} ${dis ? 'cursor-not-allowed opacity-50' : 'cursor-pointer hover:opacity-90'}`}
    >
      {children}
    </button>
  );
};

export const Toggle = ({
  on,
  onToggle,
  label,
  sub,
}: {
  on: boolean;
  onToggle: () => void;
  label: string;
  sub?: string;
}) => (
  <div className="mb-2.5 flex items-center justify-between rounded-xl border border-border bg-surface p-3">
    <div>
      <div className="text-xs font-semibold text-text">{label}</div>
      {sub && <div className="text-[10px] text-sec">{sub}</div>}
    </div>
    <button onClick={onToggle} className={`relative h-6 w-11 rounded-full transition-colors ${on ? 'bg-accent' : 'bg-border'}`}>
      <div className={`absolute top-1 h-4 w-4 rounded-full bg-white transition-all ${on ? 'left-[23px]' : 'left-1'}`} />
    </button>
  </div>
);

export const StatCard = ({
  v,
  l,
  c = 'text-accent',
  i,
}: {
  v: string | number;
  l: string;
  c?: string;
  i?: React.ReactNode;
}) => (
  <div className="flex flex-col items-center justify-center rounded-xl border border-border bg-card p-2.5 text-center">
    {i && <div className="mb-0.5 text-sm">{i}</div>}
    <div className={`text-base font-extrabold ${c}`}>{v}</div>
    <div className="mt-1 text-[8px] font-semibold uppercase tracking-widest text-sec">{l}</div>
  </div>
);
