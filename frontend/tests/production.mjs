// Run against the built files, including the real GitHub Pages subpath.
import {chromium} from 'playwright';
import {spawn} from 'node:child_process';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const base='/CSE3141-Predictive-Analytics/';
const server=spawn(process.execPath,['node_modules/vite/bin/vite.js','preview','--host','127.0.0.1','--port','4173','--strictPort'],{env:{...process.env,VITE_BASE_PATH:base},stdio:'pipe'});
let output='';server.stdout.on('data',chunk=>output+=chunk);server.stderr.on('data',chunk=>output+=chunk);
let browser;
try {
  let ready=false;
  for(let i=0;i<100;i++){
    try{const r=await fetch(`http://127.0.0.1:4173${base}`);if(r.ok){ready=true;break;}}catch{}
    if(server.exitCode!==null)throw new Error(output);
    await new Promise(r=>setTimeout(r,100));
  }
  assert.ok(ready,output);
  browser=await chromium.launch(process.platform==='win32'?{executablePath:'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',headless:true}:{headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[],missing=[],staticRequests=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('response',r=>{if(new URL(r.url()).origin==='http://127.0.0.1:4173'&&r.status()>=400)missing.push(r.url());});
  page.on('request',r=>{if(r.url().includes('/data/'))staticRequests.push(r.url());});
  // Reproduce the deployed legacy backend's HTTP-200 / incompatible schema.
  await page.route('**/api/**',route=>route.fulfill({json:{model_name:'time_weather',hourly:[],volume_threshold:5000}}));
  const url=`http://127.0.0.1:4173${base}`;
  for(const [route,title] of [['','Traffic, understood through motion.'],['analysis','Inside the trajectories'],['performance','Let the validation decide.'],['about','A clear question. An honest experiment.']]){
    await page.goto(`${url}#/${route}`);
    for(let attempt=0;attempt<3;attempt++){
      if(attempt)await page.reload();
      await page.getByRole('heading',{name:title,exact:true}).waitFor();
      if(!route)await page.getByText('1,180,598',{exact:true}).waitFor();
      else if(route==='performance')await page.getByText('87.1%',{exact:true}).first().waitFor();
      else if(route==='analysis')await page.getByRole('heading',{name:'Feature correlations',exact:true}).waitFor();
      else await page.getByRole('heading',{name:'Exported model schema',exact:true}).waitFor();
      assert.ok((await page.locator('main').innerText()).length>100);
    }
  }
  await page.goto(`${url}#/simulator`);
  await page.getByRole('button',{name:'Start Simulation',exact:true}).click();
  await page.getByText('Predictions unavailable · retrying',{exact:true}).waitFor();
  const time=await page.getByTestId('sim-time').textContent();
  await page.waitForTimeout(500);
  assert.notEqual(await page.getByTestId('sim-time').textContent(),time);
  assert.equal(await page.locator('.risk-box strong').textContent(),'—');
  await page.getByRole('button',{name:'Pause Simulation',exact:true}).click();
  await page.goto(url);
  await page.getByText('1,180,598',{exact:true}).waitFor();
  fs.mkdirSync('test-results',{recursive:true});
  await page.screenshot({path:'test-results/production-dashboard.png',fullPage:true});
  assert.ok(staticRequests.length>0);
  assert.ok(staticRequests.every(url=>new URL(url).pathname.startsWith(`${base}data/`)));
  assert.deepEqual(missing,[]);assert.deepEqual(errors,[]);
  console.log('PASS: production bundle under GitHub Pages subpath; 12 repeated research-page loads with legacy HTTP-200 data, saved NearMiss fallback, malformed predictions, continuous simulation, zero missing assets or render errors.');
} finally {await browser?.close();server.kill();}
