import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {inferBehavior,TASK} from '../src/services/behavior-inference.js';
import {validateResearch,validatePredictions} from '../src/services/research-schema.js';
const load=name=>JSON.parse(fs.readFileSync(new URL(`../public/data/${name}.json`,import.meta.url)));
const info=load('info');
test('portable selected model matches sklearn held-out scores and validation cutoff',()=>{
  for(const row of info.checks){const p=inferBehavior(info,row.inputs);assert.equal(p.eligible,true);assert.ok(Math.abs(p.probability-row.probability)<1e-12);assert.equal(p.prediction,Number(row.probability>=info.decision_threshold));}
});
test('QDA portable Gaussian transform works independently of selected algorithm',()=>{
  const qda={task:TASK,name:'QDA',version:'behavior-1',features:['x'],ranges:{x:[-10,10]},decision_threshold:.5,browser_model:{kind:'scaled-qda',mean:[0],scale:[1],means:[[0],[2]],rotations:[[[1]],[[1]]],variances:[[1],[1]],priors:[.5,.5]}};
  assert.ok(Math.abs(inferBehavior(qda,{x:1}).probability-.5)<1e-12);
  assert.ok(Math.abs(inferBehavior(qda,{x:2}).probability-1/(1+Math.exp(-2)))<1e-12);
});
test('unsupported data has no probability',()=>{
  const r=inferBehavior(info,{...info.checks[0].inputs,speed:100});assert.equal(r.eligible,false);assert.equal(r.probability,undefined);
});
for(const name of ['info','metrics','analytics','replay']){
  test(`${name}: real behavior export accepted and old task rejected`,()=>{
    assert.equal(validateResearch(name,load(name)).task,TASK);
    for(const old of [{},null,{task:'close-following',version:'nearmiss-1'},{task:TASK,version:'behavior-1'}])assert.throws(()=>validateResearch(name,old));
  });
}
test('old conflict predictions cannot be mistaken for behavior forecasts',()=>{
  assert.throws(()=>validatePredictions({predictions:[{id:'1',eligible:true,probability:.8}]}));
  assert.doesNotThrow(()=>validatePredictions({task:TASK,predictions:[{id:'1',eligible:true,task:TASK,prediction:1,probability:.8}]}));
});
