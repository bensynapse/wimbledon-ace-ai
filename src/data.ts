export type SportType =
  | 'voetbal'
  | 'hockey'
  | 'handbal'
  | 'basketbal'
  | 'volleybal'
  | 'korfbal'
  | 'tennis'
  | 'rugby'
  | 'cricket'
  | 'futsal'
  | 'waterpolo'
  | 'ijshockey';

export interface Referee {
  id: number;
  name: string;
  sports: SportType[];
  rating: number;
  reviews: number;
  tarief: number;
  level: string;
  km: number;
  city: string;
  av: string;
  verified: boolean;
  idVerified: boolean;
  badges: string[];
  bio: string;
  matches: number;
  resp: number;
  respTime: string;
  langs: string[];
  avail: boolean[];
  earned: number;
  fav: boolean;
  insuranceOpt: boolean;
  videoIntro: boolean;
}

export const allSports: { id: SportType; n: string; i: string; refs: number }[] = [
  { id: 'voetbal', n: 'Voetbal', i: '⚽', refs: 3 },
  { id: 'hockey', n: 'Hockey', i: '🏑', refs: 2 },
  { id: 'handbal', n: 'Handbal', i: '🤾', refs: 2 },
  { id: 'basketbal', n: 'Basketbal', i: '🏀', refs: 2 },
  { id: 'volleybal', n: 'Volleybal', i: '🏐', refs: 2 },
  { id: 'korfbal', n: 'Korfbal', i: '🥅', refs: 1 },
  { id: 'tennis', n: 'Tennis', i: '🎾', refs: 1 },
  { id: 'rugby', n: 'Rugby', i: '🏉', refs: 3 },
  { id: 'cricket', n: 'Cricket', i: '🏏', refs: 2 },
  { id: 'futsal', n: 'Futsal', i: '⚽', refs: 2 },
  { id: 'waterpolo', n: 'Waterpolo', i: '🤽', refs: 2 },
  { id: 'ijshockey', n: 'IJshockey', i: '🏒', refs: 3 },
];

export const days = ['Ma', 'Di', 'Wo', 'Do', 'Vr', 'Za', 'Zo'];
export const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
export const earningsData = [1250, 980, 1540, 1320, 1680, 2100, 1890, 2340, 1560, 1780, 2450, 2800];

export const demoRefs: Referee[] = [
  {
    id: 1,
    name: 'Arda Tastan',
    sports: ['voetbal', 'futsal'],
    rating: 4.8,
    reviews: 47,
    tarief: 35,
    level: 'District',
    km: 12,
    city: 'Eibergen',
    av: 'AT',
    verified: true,
    idVerified: true,
    badges: ['Top Rated', 'KNVB'],
    bio: '5+ years amateur football experience. Fair, communicative, always on time.',
    matches: 156,
    resp: 98,
    respTime: '< 1h',
    langs: ['NL', 'TR', 'EN'],
    avail: [true, true, false, true, true, true, false],
    earned: 5460,
    fav: false,
    insuranceOpt: true,
    videoIntro: true,
  },
  {
    id: 2,
    name: 'Marieke de Vries',
    sports: ['voetbal'],
    rating: 4.6,
    reviews: 31,
    tarief: 30,
    level: 'Regio',
    km: 8,
    city: 'Enschede',
    av: 'MV',
    verified: true,
    idVerified: true,
    badges: ['Quick Reply'],
    bio: "Dedicated to fair play. Specializing in youth and women's football.",
    matches: 89,
    resp: 95,
    respTime: '< 2h',
    langs: ['NL', 'EN'],
    avail: [true, false, true, true, false, true, true],
    earned: 2670,
    fav: true,
    insuranceOpt: false,
    videoIntro: false,
  },
];

export const demoDisputes = [
  {
    id: 1,
    match: 'SC Eibergen vs VV Rekken',
    status: 'open',
    type: 'No-show',
    date: '5 Apr',
    desc: 'Referee did not show up. Club requests refund.',
  },
  {
    id: 2,
    match: 'HC Groenlo vs MHC Twente',
    status: 'resolved',
    type: 'Payment',
    date: '28 Mar',
    desc: 'Payment delayed. Resolved — paid within 72h.',
  },
];
