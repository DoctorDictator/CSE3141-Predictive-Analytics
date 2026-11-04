import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validateResearch,validatePredictions} from '../src/services/research-schema.js';

for (const name of ['analytics','metrics','info']) {
  const data=JSON.parse(fs.readFileSync(new URL(`../public/data/${name}.json`,import.meta.url)));
  test(`${name}: actual NearMiss export satisfies rendering contract`,()=>assert.equal(validateResearch(name,data),data));
  test(`${name}: legacy or incomplete API response is rejected before rendering`,()=>{
    for(const legacy of [{},null,{model_name:'time_weather',hourly:[],volume_threshold:5000}])assert.throws(()=>validateResearch(name,legacy));
  });
}
test('undefined or nonfinite rendered metrics are rejected',()=>{
  const data=JSON.parse(fs.readFileSync(new URL('../public/data/metrics.json',import.meta.url)));
  delete data.test.f1;
  assert.throws(()=>validateResearch('metrics',data));
});
test('legacy prediction responses cannot produce fabricated or NaN percentages',()=>{
  for(const value of [{high_probability:.9},{predictions:[{id:'1',eligible:true}]},{predictions:[{id:'1',eligible:true,probability:NaN}]}])assert.throws(()=>validatePredictions(value));
  assert.doesNotThrow(()=>validatePredictions({predictions:[{id:'1',eligible:true,probability:.5},{id:'2',eligible:false,reason:'Outside training support'}]}));
});
