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

export type Badge =
  | 'Top Rated'
  | 'KNVB'
  | 'Quick Reply'
  | 'KNHB'
  | 'Superhost'
  | 'New'
  | 'International'
  | 'FA'
  | 'Rising Star';

export type WeekAvailability = [boolean, boolean, boolean, boolean, boolean, boolean, boolean];

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
  badges: Badge[];
  bio: string;
  matches: number;
  resp: number;
  respTime: string;
  langs: string[];
  avail: WeekAvailability;
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
  {
    id: 3,
    name: 'Bas Hendriks',
    sports: ['hockey'],
    rating: 4.9,
    reviews: 62,
    tarief: 40,
    level: 'Bonds',
    km: 22,
    city: 'Deventer',
    av: 'BH',
    verified: true,
    idVerified: true,
    badges: ['Top Rated', 'KNHB', 'Superhost'],
    bio: 'KNHB certified umpire for all levels. Reliable and professional.',
    matches: 234,
    resp: 100,
    respTime: '< 30m',
    langs: ['NL', 'EN', 'DE'],
    avail: [true, true, true, true, true, false, false],
    earned: 9360,
    fav: false,
    insuranceOpt: true,
    videoIntro: true,
  },
  {
    id: 4,
    name: 'Sophie van Dijk',
    sports: ['handbal', 'volleybal'],
    rating: 4.4,
    reviews: 18,
    tarief: 28,
    level: 'Club',
    km: 5,
    city: 'Borculo',
    av: 'SD',
    verified: false,
    idVerified: false,
    badges: ['New'],
    bio: 'Starting referee eager to gain experience across multiple sports.',
    matches: 12,
    resp: 87,
    respTime: '< 3h',
    langs: ['NL'],
    avail: [false, false, true, true, false, true, true],
    earned: 336,
    fav: false,
    insuranceOpt: false,
    videoIntro: false,
  },
  {
    id: 5,
    name: 'James Wilson',
    sports: ['voetbal', 'rugby'],
    rating: 4.7,
    reviews: 55,
    tarief: 45,
    level: 'National',
    km: 35,
    city: 'London',
    av: 'JW',
    verified: true,
    idVerified: true,
    badges: ['International', 'FA'],
    bio: 'FA Level 5. Grassroots to semi-pro. Available weekends.',
    matches: 312,
    resp: 96,
    respTime: '< 1h',
    langs: ['EN', 'FR'],
    avail: [true, true, true, false, false, true, true],
    earned: 14040,
    fav: false,
    insuranceOpt: true,
    videoIntro: true,
  },
  {
    id: 6,
    name: 'Fatima El-Amin',
    sports: ['basketbal'],
    rating: 4.5,
    reviews: 29,
    tarief: 32,
    level: 'District',
    km: 15,
    city: 'Utrecht',
    av: 'FA',
    verified: true,
    idVerified: true,
    badges: ['Rising Star'],
    bio: 'NBB referee. Strong communication. Available midweek and weekends.',
    matches: 67,
    resp: 92,
    respTime: '< 2h',
    langs: ['NL', 'AR', 'EN'],
    avail: [true, false, true, true, true, false, true],
    earned: 2144,
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
