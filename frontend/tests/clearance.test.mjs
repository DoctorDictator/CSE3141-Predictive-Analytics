import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createWorld,vehicle,laneAssessment,loadScenario,tick,STEP,requestLaneChange,generateFromProfile,observations} from '../src/simulation.js';
const advance=(w,s)=>{for(let i=0;i<s/STEP;i++)tick(w);};
test('six-lane bypass remains independent of any behavior prediction',()=>{
  const a=createWorld(),b=createWorld();loadScenario(a,'Six-Lane Blockage - Safe Bypass');loadScenario(b,'Six-Lane Blockage - Safe Bypass');
  a.behaviorPrediction=0;b.behaviorPrediction=1;
  advance(a,9);advance(b,9);assert.deepEqual(a.cars,b.cars);assert.equal(a.collisions.length,0);assert.equal(a.cars.find(c=>c.id===2).lane,3);assert.ok(a.completedLaneChanges>0);
});
test('stopped uncollided vehicle enters an empty lane',()=>{
  const w=createWorld();w.mode='Automatic Traffic';w.autoLane=true;w.cars=[vehicle(1,1,50,0,{desired:15}),vehicle(2,1,60,0,{desired:0})];advance(w,3);assert.notEqual(w.cars[0].lane,1);assert.equal(w.collisions.length,0);
});
test('blocked alternatives cause waiting, not forced merging',()=>{const w=createWorld();loadScenario(w,'Blocked Lane - No Safe Gap');advance(w,10);assert.equal(w.collisions.length,0);assert.equal(w.laneChanges,0);});
test('rear approach and full lateral path are checked geometrically',()=>{
  const w=createWorld();w.cars=[vehicle(1,0,100,10),vehicle(2,1,70,30)];assert.equal(laneAssessment(w,w.cars[0],1).clear,false);
  w.cars=[vehicle(1,0,20,20),vehicle(2,0,30,0)];assert.equal(laneAssessment(w,w.cars[0],1).clear,false);
});
test('simultaneous merges reserve the path',()=>{const w=createWorld();w.cars=[vehicle(1,0,50,12),vehicle(2,2,50,12)];assert.equal(requestLaneChange(w,w.cars[0],1),true);assert.equal(requestLaneChange(w,w.cars[1],1),false);});
test('aggregate-profile traffic is seeded and remains within the selected road',()=>{
  const a=JSON.parse(fs.readFileSync(new URL('../public/data/analytics.json',import.meta.url)));
  const cars=generateFromProfile(6,30,42,a.simulator_profile);
  assert.deepEqual(cars,generateFromProfile(6,30,42,a.simulator_profile));
  assert.equal(cars.length,30);
  assert.ok(cars.every(c=>c.lane>=0&&c.lane<6&&c.speed>=0&&c.sourceId===undefined));
});
