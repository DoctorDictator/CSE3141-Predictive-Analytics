import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {inferBehavior,TASK} from '../src/services/behavior-inference.js';
import {validateResearch,validatePredictions} from '../src/services/research-schema.js';
const load=name=>JSON.parse(fs.readFileSync(new URL(`../public/data/${name}.json`,import.meta.url)));
const info=load('info');
const sample=()=>Object.fromEntries(info.features.map(key=>{
  const [low,high]=info.ranges[key];
  if(key==='has_leader'||key==='is_truck'||key.endsWith('_present'))return [key,0];
  if(key.endsWith('_gap'))return [key,low<=150&&high>=150?150:low];
  if(key==='closing_speed'||key.endsWith('_closing_speed'))return [key,0];
  return [key,(low+high)/2];
}));
test('portable behavior model accepts the exported adjacent-lane feature schema',()=>{
  const result=inferBehavior(info,sample());
  assert.equal(info.version,'behavior-3');
  assert.equal(info.features.length,19);
  assert.equal(result.eligible,true);
  assert.ok(result.probability>=0&&result.probability<=1);
  assert.equal(result.prediction,Number(result.probability>=info.decision_threshold));
});
test('QDA portable Gaussian transform works independently of selected algorithm',()=>{
  const qda={task:TASK,name:'QDA',version:'behavior-3',features:['x'],ranges:{x:[-10,10]},decision_threshold:.5,browser_model:{kind:'scaled-qda',mean:[0],scale:[1],means:[[0],[2]],rotations:[[[1]],[[1]]],variances:[[1],[1]],priors:[.5,.5]}};
  assert.ok(Math.abs(inferBehavior(qda,{x:1}).probability-.5)<1e-12);
  assert.ok(Math.abs(inferBehavior(qda,{x:2}).probability-1/(1+Math.exp(-2)))<1e-12);
});
test('unsupported data has no probability',()=>{
  const result=inferBehavior(info,{...sample(),speed:100});
  assert.equal(result.eligible,false);
  assert.equal(result.probability,undefined);
});
for(const name of ['info','metrics','analytics']){
  test(`${name}: Automatum behavior export accepted and old task rejected`,()=>{
    assert.equal(validateResearch(name,load(name)).task,TASK);
    for(const old of [{},null,{task:'close-following',version:'nearmiss-1'},{task:TASK,version:'behavior-1'}])assert.throws(()=>validateResearch(name,old));
  });
}
test('old conflict predictions cannot be mistaken for behavior forecasts',()=>{
  assert.throws(()=>validatePredictions({predictions:[{id:'1',eligible:true,probability:.8}]}));
  const envelope={task:TASK,version:info.version,model_id:info.model_id};
  const data={...envelope,predictions:[{...envelope,id:'1',eligible:true,prediction:1,probability:.8}]};
  assert.doesNotThrow(()=>validatePredictions(data,info,['1']));
  assert.throws(()=>validatePredictions({...data,version:'behavior-1'},info,['1']));
  assert.throws(()=>validatePredictions({...data,model_id:'0'.repeat(64)},info,['1']));
  assert.throws(()=>validatePredictions({...data,predictions:[data.predictions[0],data.predictions[0]]},info,['1']));
  assert.throws(()=>validatePredictions(data,info,['2']));
});
