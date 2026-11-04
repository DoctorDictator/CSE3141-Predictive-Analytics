// Metres, seconds, front-bumper longitudinal coordinates. Geometry is independent of ML.
export const LANE_WIDTH = 3.7;
export const STEP = 1 / 120;
export const SCENARIOS = [
  "Normal Highway Traffic",
  "Sudden Braking",
  "Unsafe Lane Change",
  "Heavy Traffic",
  "Rear-End Collision Demo",
  "Multiple-Vehicle Incident",
];
const clone = (x) => structuredClone(x);
export function random(seed) {
  let n = Number(seed) >>> 0;
  return () => {
    n += 0x6d2b79f5;
    let t = n;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function vehicle(id, lane, x, speed = 15, extra = {}) {
  return {
    id,
    lane,
    y: (lane + 0.5) * LANE_WIDTH,
    x,
    speed,
    accel: 0,
    desired: speed,
    length: 4.6,
    width: 1.8,
    type: "Car",
    color: ["#4ed8bb", "#8daeff", "#f2c46f", "#cb9cef", "#f38c88"][id % 5],
    crashed: false,
    changing: false,
    ...extra,
  };
}
export function generate(lanes = 3, count = 6, seed = 42) {
  const rng = random(seed),
    cars = [];
  for (let i = 0; i < count; i++) {
    const lane = i % lanes,
      row = Math.floor(i / lanes),
      truck = rng() < 0.15;
    cars.push(
      vehicle(i + 1, lane, 30 + row * 35 + rng() * 8, 12 + rng() * 9, {
        length: truck ? 9 : 4.6,
        width: truck ? 2.4 : 1.8,
        type: truck ? "Truck" : "Car",
        accel: Number(((rng() - 0.5) * 0.3).toFixed(2)),
      }),
    );
  }
  return cars;
}
export function createWorld(lanes = 3, count = 6, seed = 42) {
  const w = {
    lanes,
    cars: generate(lanes, count, seed),
    time: 0,
    events: [],
    collisions: [],
    laneChanges: 0,
    nextId: count + 1,
    mode: "Manual Traffic",
    avoidance: true,
    autoLane: false,
    history: [],
    historyClock: 0,
    initial: null,
    epoch: 0,
  };
  saveInitial(w);
  return w;
}
export function saveInitial(w) {
  w.initial = clone({
    cars: w.cars,
    lanes: w.lanes,
    nextId: w.nextId,
    mode: w.mode,
    avoidance: w.avoidance,
    autoLane: w.autoLane,
  });
}
export function reset(w) {
  Object.assign(w, clone(w.initial), {
    time: 0,
    events: [],
    collisions: [],
    laneChanges: 0,
    history: [],
    historyClock: 0,
    epoch: w.epoch + 1,
  });
}
export function event(w, text, type = "control") {
  w.events.unshift({ time: w.time, text, type });
  w.events = w.events.slice(0, 100);
}
export function leader(w, car) {
  if (car.changing || car.crashed) return null;
  return (
    w.cars
      .filter(
        (c) =>
          c.id !== car.id && !c.changing && c.lane === car.lane && c.x > car.x,
      )
      .sort((a, b) => a.x - b.x)[0] || null
  );
}
export function overlap(a, b) {
  return (
    a.x > b.x - b.length &&
    b.x > a.x - a.length &&
    Math.abs(a.y - b.y) < (a.width + b.width) / 2
  );
}
export function changeLane(w, car, lane) {
  if (
    car.crashed ||
    car.changing ||
    lane < 0 ||
    lane >= w.lanes ||
    lane === car.lane
  )
    return false;
  car.changing = true;
  car.fromY = car.y;
  car.toY = (lane + 0.5) * LANE_WIDTH;
  car.changeTime = 0;
  car.targetLane = lane;
  w.laneChanges++;
  event(
    w,
    `Vehicle ${car.id} changing from Lane ${car.lane + 1} to Lane ${lane + 1}`,
    "lane",
  );
  return true;
}
export function setLanes(w, lanes) {
  w.lanes = lanes;
  // Old snapshots refer to different road geometry and must not be replayed.
  w.history = [];
  w.historyClock = 0;
  // Lane compression redistributes vehicles safely rather than creating accidental contacts.
  const occupied = Array.from({ length: lanes }, () => []);
  for (const c of w.cars) {
    c.lane = Math.min(c.lane, lanes - 1);
    c.y = (c.lane + 0.5) * LANE_WIDTH;
    c.changing = false;
    while (
      occupied[c.lane].some(
        (o) => Math.abs(o.x - c.x) < Math.max(o.length, c.length) + 5,
      )
    )
      c.x += 15;
    occupied[c.lane].push(c);
  }
  w.epoch++;
  event(w, `Road rebuilt with ${lanes} lanes`);
}
export function brake(w, c) {
  if (c.crashed) return;
  c.targetActive = false;
  c.accel = -5;
  c.brakeUntil = w.time + 3;
  event(w, `Vehicle ${c.id} started braking`, "brake");
}
export function addCar(w) {
  if (w.cars.length >= 30) return null;
  for (let lane = 0; lane < w.lanes; lane++) {
    const x = Math.max(15, Math.min(...w.cars.map((c) => c.x), 45) - 20);
    if (w.cars.every((c) => c.lane !== lane || Math.abs(c.x - x) > 18)) {
      const c = vehicle(w.nextId++, lane, x, 15);
      w.cars.push(c);
      event(w, `Vehicle ${c.id} added`);
      return c;
    }
  }
  event(w, "Entrance occupied; no vehicle added");
  return null;
}
export function loadScenario(w, name) {
  Object.assign(
    w,
    createWorld(
      name === "Heavy Traffic" ? 4 : 3,
      name === "Heavy Traffic" ? 24 : 6,
      42,
    ),
  );
  if (name === "Normal Highway Traffic")
    w.cars.forEach((c) => {
      c.speed = 14 + c.lane * 2;
      c.desired = c.speed;
      c.accel = 0;
    });
  if (name === "Heavy Traffic") {
    w.cars.forEach((c, i) => {
      c.x = 25 + Math.floor(i / 4) * 18 + c.lane * 2;
      c.speed = 6 + (i % 5);
      c.desired = c.speed;
      c.accel = 0;
    });
    w.mode = "Automatic Traffic";
  }
  if (name === "Sudden Braking") {
    w.cars = [
      vehicle(1, 1, 90, 16, { brakeAt: 1 }),
      vehicle(2, 1, 45, 22),
      vehicle(3, 0, 65, 17),
    ];
  }
  if (name === "Rear-End Collision Demo")
    w.cars = [vehicle(1, 1, 80, 5), vehicle(2, 1, 40, 22)];
  if (name === "Unsafe Lane Change")
    w.cars = [
      vehicle(1, 0, 60, 15, { laneAt: 0.5, laneTo: 1 }),
      vehicle(2, 1, 60, 15),
      vehicle(3, 2, 110, 18),
    ];
  if (name === "Multiple-Vehicle Incident")
    w.cars = [
      vehicle(1, 1, 95, 0),
      vehicle(2, 1, 66, 18),
      vehicle(3, 1, 35, 23),
      vehicle(4, 0, 65, 15),
    ];
  w.avoidance = ![
    "Rear-End Collision Demo",
    "Unsafe Lane Change",
    "Multiple-Vehicle Incident",
  ].includes(name);
  w.nextId = Math.max(...w.cars.map((c) => c.id)) + 1;
  saveInitial(w);
}
export function tick(w, dt = STEP) {
  dt = Math.min(dt, 0.02);
  w.time += dt;
  for (const c of w.cars) {
    if (c.crashed) continue;
    if (c.brakeAt != null && w.time >= c.brakeAt) {
      brake(w, c);
      delete c.brakeAt;
    }
    if (c.laneAt != null && w.time >= c.laneAt) {
      changeLane(w, c, c.laneTo);
      delete c.laneAt;
    }
    if (c.brakeUntil && w.time >= c.brakeUntil) {
      c.accel = 0;
      delete c.brakeUntil;
    }
    let a = c.accel;
    if (w.mode === "Manual Traffic" && c.targetActive && !c.brakeUntil) {
      a = Math.max(-4, Math.min(2.5, (c.desired - c.speed) * 1.5));
      if (Math.abs(c.desired - c.speed) < 0.05) {
        c.targetActive = false;
        c.accel = 0;
        a = 0;
      }
    }
    if (w.mode === "Automatic Traffic" && !c.brakeUntil) {
      const ahead = leader(w, c);
      a = Math.max(-2, Math.min(1.4, (c.desired - c.speed) * 0.5));
      if (w.avoidance && ahead) {
        const gap = ahead.x - ahead.length - c.x;
        if (gap < Math.max(5, c.speed * 1.7))
          a = -Math.min(8, 2 + Math.max(0, c.speed - ahead.speed));
        if (w.autoLane && gap < c.speed * 1.4 && !c.changing) {
          const target = [c.lane - 1, c.lane + 1].find(
            (l) =>
              l >= 0 &&
              l < w.lanes &&
              w.cars.every(
                (o) =>
                  o.id === c.id ||
                  Math.abs(o.y - (l + 0.5) * LANE_WIDTH) > LANE_WIDTH * 0.8 ||
                  Math.abs(o.x - c.x) > Math.max(20, c.speed * 1.5),
              ),
          );
          if (target != null) changeLane(w, c, target);
        }
      }
    }
    c.actualAccel = a;
    const next = Math.max(0, Math.min(45, c.speed + a * dt));
    c.x += (c.speed + next) * 0.5 * dt;
    c.speed = next;
    if (c.changing) {
      c.changeTime += dt;
      const t = Math.min(1, c.changeTime / 1.6),
        ease = t * t * (3 - 2 * t);
      c.y = c.fromY + (c.toY - c.fromY) * ease;
      if (t === 1) {
        c.lane = c.targetLane;
        c.changing = false;
      }
    }
  }
  for (let i = 0; i < w.cars.length; i++)
    for (let j = i + 1; j < w.cars.length; j++) {
      const a = w.cars[i],
        b = w.cars[j];
      if (
        !overlap(a, b) ||
        w.collisions.some((e) => e.ids.includes(a.id) && e.ids.includes(b.id))
      )
        continue;
      const impact = {
        ids: [a.id, b.id],
        time: w.time,
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2,
      };
      w.collisions.push(impact);
      for (const c of [a, b]) {
        c.crashed = true;
        c.speed = 0;
        c.accel = 0;
        c.actualAccel = 0;
        c.changing = false;
      }
      event(
        w,
        `Hypothetical Simulated Collision: Vehicles ${a.id} and ${b.id}`,
        "collision",
      );
    }
  w.historyClock += dt;
  if (w.historyClock >= 0.1) {
    w.historyClock = 0;
    w.history.push(
      clone({
        cars: w.cars,
        time: w.time,
        events: w.events,
        collisions: w.collisions,
        laneChanges: w.laneChanges,
      }),
    );
    if (w.history.length > 150) w.history.shift();
  }
}
export function rewind(w) {
  const impact = w.collisions.at(-1)?.time ?? w.time;
  const state = w.history.find(
    (s) => s.time >= Math.max(0, impact - 2) && s.time < impact - 0.1,
  );
  if (!state) {
    reset(w);
    return;
  }
  Object.assign(w, clone(state));
  w.history = [];
  w.epoch++;
}
export function observations(w) {
  return w.cars.flatMap((c) => {
    const l = leader(w, c);
    if (!l || l.crashed || c.speed < 0.5) return [];
    const spacing = l.x - c.x;
    if (spacing <= l.length) return [];
    return [
      {
        id: String(c.id),
        leaderId: l.id,
        inputs: {
          follower_speed: c.speed,
          leader_speed: l.speed,
          follower_accel: c.actualAccel ?? c.accel,
          leader_accel: l.actualAccel ?? l.accel,
          spacing_m: spacing,
          leader_length: l.length,
        },
      },
    ];
  });
}
