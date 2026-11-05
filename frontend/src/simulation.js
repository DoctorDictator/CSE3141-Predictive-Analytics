// Metres, seconds, front-bumper longitudinal coordinates. Geometry is independent of ML.
export const LANE_WIDTH = 3.7;
export const STEP = 1 / 120;
export const MAX_SPEED = 80;
export const SCENARIOS = [
  "Normal Highway Traffic",
  "Sudden Braking",
  "Unsafe Lane Change",
  "Heavy Traffic",
  "Rear-End Collision Demo",
  "Multiple-Vehicle Incident",
  "Six-Lane Blockage - Safe Bypass",
  "Blocked Lane - No Safe Gap",
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
    lateralSpeed: 0,
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
    completedLaneChanges: 0,
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
    initializationSource: w.initializationSource,
  });
}
export function reset(w) {
  Object.assign(w, clone(w.initial), {
    time: 0,
    events: [],
    collisions: [],
    laneChanges: 0,
    completedLaneChanges: 0,
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
  if (car.crashed) return null;
  return (
    w.cars
      .filter(
        (c) =>
          c.id !== car.id &&
          Math.abs(c.y - car.y) < (c.width + car.width) / 2 + 0.2 &&
          c.x - c.length / 2 > car.x - car.length / 2,
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
    Math.abs(lane - car.lane) !== 1
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

export function laneAssessment(w, car, lane) {
  const result = {
    lane,
    clear: false,
    reason: "",
    front: null,
    rear: null,
  };
  if (lane < 0 || lane >= w.lanes || Math.abs(lane - car.lane) !== 1)
    return { ...result, reason: "No adjacent lane" };
  if (car.crashed)
    return {
      ...result,
      reason: "Vehicle already collided; rewind before impact",
    };
  if (car.changing) return { ...result, reason: "Lane change in progress" };
  const y = (lane + 0.5) * LANE_WIDTH;
  const occupants = w.cars.filter(
    (o) =>
      o.id !== car.id &&
      (Math.abs(o.y - y) < LANE_WIDTH * 0.8 ||
        (o.changing && o.targetLane === lane)),
  );
  result.empty = occupants.length === 0;
  result.front =
    occupants.filter((o) => o.x > car.x).sort((a, b) => a.x - b.x)[0] || null;
  result.rear =
    occupants.filter((o) => o.x <= car.x).sort((a, b) => b.x - a.x)[0] || null;
  const frontGap = result.front
    ? result.front.x - result.front.length - car.x
    : null;
  const rearGap = result.rear ? car.x - car.length - result.rear.x : null;
  result.frontGap = frontGap;
  result.rearGap = rearGap;
  if (
    result.front &&
    frontGap <
      Math.max(3, car.speed * 1.2, (car.speed - result.front.speed) * 2)
  )
    return { ...result, reason: "Front gap is too small or closing" };
  if (
    result.rear &&
    rearGap <
      Math.max(3, result.rear.speed * 1.2, (result.rear.speed - car.speed) * 2)
  )
    return {
      ...result,
      reason: "Rear vehicle is too close or approaching quickly",
    };
  // Check the full 1.6 s path, including the present-lane obstacle and other lane changers.
  for (let t = 0; t <= 1.6001; t += 0.1) {
    const fraction = t / 1.6,
      smooth = fraction * fraction * (3 - 2 * fraction);
    const projected = {
      ...car,
      x: car.x + car.speed * t,
      y: car.y + (y - car.y) * smooth,
      length: car.length + 1,
      width: car.width + 0.2,
    };
    for (const other of w.cars) {
      if (other.id === car.id) continue;
      let otherY = other.y;
      if (other.changing) {
        const f = Math.min(1, (other.changeTime + t) / 1.6);
        otherY = other.fromY + (other.toY - other.fromY) * f * f * (3 - 2 * f);
      }
      if (
        overlap(projected, {
          ...other,
          x: other.x + other.speed * t,
          y: otherY,
          length: other.length + 1,
          width: other.width + 0.2,
        })
      )
        return {
          ...result,
          reason: "Vehicle body intersects the predicted lane-change path",
        };
    }
  }
  return {
    ...result,
    clear: true,
    reason: result.empty
      ? "Empty adjacent lane; projected path clear"
      : "Front/rear clearance and projected path clear",
  };
}

export function requestLaneChange(w, car, lane) {
  const assessment = laneAssessment(w, car, lane);
  car.laneDecision = {
    ...assessment,
    front: assessment.front?.id,
    rear: assessment.rear?.id,
    time: w.time,
  };
  if (w.avoidance && !assessment.clear) {
    event(w, `Vehicle ${car.id} waits: ${assessment.reason}`, "decision");
    return false;
  }
  return changeLane(w, car, lane);
}

export function setLanes(w, lanes) {
  w.lanes = lanes;
  // Clear motion history when road geometry changes.
  w.history = [];
  w.historyClock = 0;
  // Lane compression redistributes vehicles safely rather than creating accidental contacts.
  const occupied = Array.from({ length: lanes }, () => []);
  for (const c of w.cars) {
    c.lane = Math.min(c.lane, lanes - 1);
    c.y = (c.lane + 0.5) * LANE_WIDTH;
    c.changing = false;
    c.lateralSpeed = 0;
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
  const trafficProfile = w.trafficProfile;
  const behaviorModel = w.behaviorModel;
  Object.assign(
    w,
    createWorld(
      name === "Heavy Traffic" ? 4 : 3,
      name === "Heavy Traffic" ? 24 : 6,
      42,
    ),
  );
  w.trafficProfile = trafficProfile;
  w.behaviorModel = behaviorModel;
  w.initializationSource = `Programmed preset: ${name}`;
  if (name === "Normal Highway Traffic")
    w.cars.forEach((c) => {
      c.speed = 14 + c.lane * 2;
      c.desired = c.speed;
      c.accel = 0;
    });
  if (name === "Normal Highway Traffic") {
    w.mode = "Automatic Traffic";
    w.autoLane = true;
  }
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
  if (name === "Six-Lane Blockage - Safe Bypass") {
    w.lanes = 6;
    w.mode = "Automatic Traffic";
    w.autoLane = true;
    w.cars = [
      vehicle(1, 2, 130, 0, { desired: 0, type: "Stopped obstacle" }),
      vehicle(2, 2, 65, 14, { desired: 18 }),
      vehicle(3, 0, 80, 16),
      vehicle(4, 1, 110, 17),
      vehicle(5, 4, 75, 18),
      vehicle(6, 5, 95, 16),
    ];
  }
  if (name === "Blocked Lane - No Safe Gap") {
    w.lanes = 3;
    w.mode = "Automatic Traffic";
    w.autoLane = true;
    w.cars = [
      vehicle(1, 1, 100, 0, { desired: 0, type: "Stopped obstacle" }),
      vehicle(2, 1, 45, 12, { desired: 16 }),
      vehicle(3, 0, 100, 0, { desired: 0, type: "Stopped obstacle" }),
      vehicle(4, 2, 100, 0, { desired: 0, type: "Stopped obstacle" }),
    ];
  }
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
    c.lateralSpeed = 0;
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
    if (c.manualAccelUntil && w.time >= c.manualAccelUntil) {
      delete c.manualAccelUntil;
      c.accel = 0;
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
    if (
      w.mode === "Automatic Traffic" &&
      !c.brakeUntil &&
      !c.manualAccelUntil
    ) {
      const ahead = leader(w, c);
      a = Math.max(-2, Math.min(1.4, (c.desired - c.speed) * 0.5));
      if (w.avoidance && ahead) {
        const gap = ahead.x - ahead.length - c.x;
        if (gap < Math.max(5, c.speed * 1.7))
          a = -Math.min(8, 2 + Math.max(0, c.speed - ahead.speed));
        const closing = Math.max(0, c.speed - ahead.speed);
        const obstructed =
          gap < Math.max(12, c.speed * 3, closing * 4 + 8) &&
          (ahead.speed < c.desired - 1 || gap < Math.max(5, c.speed * 1.5));
        if (
          w.autoLane &&
          obstructed &&
          !c.changing &&
          (c.nextLaneCheck ?? 0) <= w.time
        ) {
          c.nextLaneCheck = w.time + 0.4;
          const options = [c.lane - 1, c.lane + 1]
            .filter((l) => l >= 0 && l < w.lanes)
            .map((l) => laneAssessment(w, c, l));
          const available = options
            .filter(
              (o) =>
                o.clear &&
                (!o.front ||
                  o.front.speed > ahead.speed + 1 ||
                  o.frontGap > gap + 10),
            )
            .sort((x, y) => Number(y.empty) - Number(x.empty));
          const chosen = available[0];
          const reason =
            chosen?.reason ||
            options.find((o) => !o.clear)?.reason ||
            "Adjacent lanes offer no progress advantage";
          c.laneDecision = {
            lane: chosen?.lane ?? null,
            clear: !!chosen,
            reason,
            time: w.time,
          };
          if (chosen) {
            event(
              w,
              `Vehicle ${c.id} assessed Lane ${chosen.lane + 1}: ${reason}`,
              "decision",
            );
            changeLane(w, c, chosen.lane);
          } else if (c.lastWaitReason !== reason) {
            event(w, `Vehicle ${c.id} waits/brakes: ${reason}`, "decision");
            c.lastWaitReason = reason;
          }
        }
      }
    }
    c.actualAccel = a;
    const next = Math.max(0, Math.min(MAX_SPEED, c.speed + a * dt));
    c.x += (c.speed + next) * 0.5 * dt;
    c.speed = next;
    if (c.changing) {
      c.changeTime += dt;
      const t = Math.min(1, c.changeTime / 1.6),
        ease = t * t * (3 - 2 * t);
      c.y = c.fromY + (c.toY - c.fromY) * ease;
      c.lateralSpeed = ((c.toY - c.fromY) / 1.6) * 6 * t * (1 - t);
      if (t === 1) {
        c.lane = c.targetLane;
        c.changing = false;
        c.completedLane = c.lane;
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
        c.lateralSpeed = 0;
        c.changing = false;
      }
      event(
        w,
        `Hypothetical Simulated Collision: Vehicles ${a.id} and ${b.id}`,
        "collision",
      );
    }
  for (const c of w.cars)
    if (c.completedLane != null) {
      if (!c.crashed) {
        w.completedLaneChanges = (w.completedLaneChanges ?? 0) + 1;
        event(
          w,
          `Vehicle ${c.id} completed change to Lane ${c.completedLane + 1} with no simulated contact`,
          "lane",
        );
      }
      delete c.completedLane;
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
        completedLaneChanges: w.completedLaneChanges,
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
export function behaviorObservation(w, c) {
  if (c.crashed) return null;
  const currentLane = Math.max(0, Math.min(w.lanes - 1, Math.floor(c.y / LANE_WIDTH)));
  const l = leader(w, c);
  const neighbor = (lane, position) => {
    if (lane < 0 || lane >= w.lanes) return null;
    const candidates = w.cars.filter(
      (v) =>
        v.id !== c.id &&
        Math.floor(v.y / LANE_WIDTH) === lane &&
        (position === "front"
          ? v.x - v.length / 2 > c.x - c.length / 2
          : v.x - v.length / 2 <= c.x - c.length / 2),
    );
    return position === "front"
      ? candidates.sort((a, b) => a.x - b.x)[0] || null
      : candidates.sort((a, b) => b.x - a.x)[0] || null;
  };
  const adjacent = (side, position) => {
    const other = neighbor(currentLane + (side === "left" ? -1 : 1), position);
    const isFront = position === "front";
    const gap = other
      ? Math.max(
          0,
          Math.min(
            150,
            isFront
              ? other.x - other.length - c.x
              : c.x - c.length - other.x,
          ),
        )
      : 150;
    return {
      [`${side}_${position}_gap`]: gap,
      [`${side}_${position}_closing_speed`]: other
        ? isFront
          ? c.speed - other.speed
          : other.speed - c.speed
        : 0,
      [`${side}_${position}_present`]: other ? 1 : 0,
    };
  };
  const inputs = {
    speed: c.speed,
    acceleration: c.actualAccel ?? c.accel,
    lateral_speed: Math.abs(c.lateralSpeed ?? 0),
    front_gap: l ? Math.max(0, Math.min(150, l.x - l.length - c.x)) : 150,
    closing_speed: l ? c.speed - l.speed : 0,
    has_leader: l ? 1 : 0,
    ...adjacent("left", "front"),
    ...adjacent("left", "rear"),
    ...adjacent("right", "front"),
    ...adjacent("right", "rear"),
    is_truck: String(c.type).toLowerCase().includes("truck") ? 1 : 0,
  };
  return {
    id: String(c.id),
    signature: JSON.stringify([c.id, currentLane, inputs]),
    inputs,
  };
}
export function observations(w) {
  return w.cars.map((c) => behaviorObservation(w, c)).filter(Boolean);
}

export function generateFromProfile(lanes, count, seed, profile) {
  if (!profile?.speed_bins?.length || !profile?.acceleration_bins?.length)
    throw new Error("Dataset distribution profile is not loaded.");
  const rng = random(seed);
  const draw = (bins, fallback) => {
    const available = bins.filter((bin) => bin.count > 0);
    const total = available.reduce((sum, bin) => sum + bin.count, 0);
    if (!total) return fallback;
    let threshold = rng() * total;
    for (const bin of available) {
      threshold -= bin.count;
      if (threshold <= 0) return bin.value;
    }
    return available.at(-1).value;
  };
  return Array.from({ length: count }, (_, i) => {
    const truck = rng() < profile.truck_probability;
    const lane = i % lanes;
    const row = Math.floor(i / lanes);
    const length = truck ? profile.truck_length : profile.car_length;
    const width = truck ? profile.truck_width : profile.car_width;
    return vehicle(i + 1, lane, 18 + row * 30 + rng() * 6, draw(profile.speed_bins, 20), {
      accel: draw(profile.acceleration_bins, 0),
      length,
      width,
      type: truck ? "Truck" : "Car",
    });
  });
}
