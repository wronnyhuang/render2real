// headless Chrome check of the comparison page: screenshots at 390 px + desktop, sound toggle plays audio.
const { chromium } = require('node_modules/playwright-core');
const PAGE = process.env.PAGE_URL || 'file://$OUT_DIR/render2real-poc/index.html';
const SHOTS = process.env.SHOTS || '$WORK/audio/shots';
const TEST = ['fast_turbo_lora__M000_P0115', 'wan_lora_prompt_cv6__M023_P0370', 'fast_unet_l1lpips__muse_01'];
(async () => {
  const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true,
    args: ['--autoplay-policy=no-user-gesture-required', '--no-sandbox'] });
  const out = { mobile: {}, desktop: {} };
  // ---- desktop
  let ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  let page = await ctx.newPage(); await page.goto(PAGE); await page.waitForTimeout(800);
  await page.screenshot({ path: `${SHOTS}/desktop_top.png` });
  await page.screenshot({ path: `${SHOTS}/desktop_full.png`, fullPage: true });
  const fig = page.locator('figure.v').first(); await fig.scrollIntoViewIfNeeded();
  await page.locator('section.card').first().screenshot({ path: `${SHOTS}/desktop_card1.png` });
  await page.locator('section.card').nth(3).screenshot({ path: `${SHOTS}/desktop_card4_wan.png` });
  await ctx.close();
  // ---- mobile 390 px, touch
  ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  page = await ctx.newPage(); await page.goto(PAGE); await page.waitForTimeout(800);
  await page.screenshot({ path: `${SHOTS}/m390_top.png` });
  out.mobile.overflowX = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  out.mobile.videos = await page.evaluate(() => document.querySelectorAll('figure.v video').length);
  out.mobile.buttons = await page.evaluate(() => document.querySelectorAll('button.snd').length);
  out.mobile.noaudio_tags = await page.evaluate(() => document.querySelectorAll('.noaud').length);
  for (const name of TEST) {
    const f = page.locator(`figure.v:has(source[src="videos/${name}.mp4"])`);
    await f.scrollIntoViewIfNeeded();
    await f.screenshot({ path: `${SHOTS}/m390_row_${name}.png` });
    const btn = f.locator('button.snd');
    const box = await btn.boundingBox();
    const hit = await btn.evaluate(b => { const r = b.getBoundingClientRect(); const e = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2); return e === b || b.contains(e); });
    await btn.tap();
    await page.waitForTimeout(3000);
    const st = await f.locator('video').evaluate(v => ({ muted: v.muted, paused: v.paused, currentTime: +v.currentTime.toFixed(2), duration: v.duration,
      webkitAudioDecodedByteCount: v.webkitAudioDecodedByteCount, webkitVideoDecodedByteCount: v.webkitVideoDecodedByteCount,
      audioTracks: v.audioTracks ? v.audioTracks.length : 'n/a', mozHasAudio: v.mozHasAudio }));
    const t2 = await f.locator('video').evaluate(v => new Promise(r => { const a = v.currentTime; setTimeout(() => r(+(v.currentTime - a).toFixed(2)), 1000); }));
    const audible = await page.evaluate(() => [...document.querySelectorAll('figure.v video')].filter(v => !v.muted && !v.paused).map(v => v.querySelector('source').getAttribute('src')));
    const label = await btn.innerText();
    await f.screenshot({ path: `${SHOTS}/m390_row_${name}_playing.png` });
    out.mobile[name] = { button_box: box && { w: Math.round(box.width), h: Math.round(box.height) }, button_hit_test: hit, after_tap: st,
      currentTime_advance_1s: t2, audible_videos: audible, button_label: label };
  }
  // WebAudio RMS on the last tapped video (analyser on the element output)
  const last = TEST[TEST.length - 1];
  out.mobile.webaudio = await page.locator(`figure.v:has(source[src="videos/${last}.mp4"]) video`).evaluate(v => new Promise(res => {
    try {
      const ac = new AudioContext(); const src = ac.createMediaElementSource(v); const an = ac.createAnalyser(); an.fftSize = 2048;
      src.connect(an); an.connect(ac.destination); ac.resume(); v.currentTime = 0; v.muted = false; v.play();
      const buf = new Float32Array(an.fftSize); let peak = 0, sum = 0, n = 0;
      const iv = setInterval(() => { an.getFloatTimeDomainData(buf); let s = 0; for (const x of buf) s += x * x; const r = Math.sqrt(s / buf.length); peak = Math.max(peak, r); sum += r; n++; }, 50);
      setTimeout(() => { clearInterval(iv); res({ ctxState: ac.state, frames: n, meanRms: +(sum / n).toFixed(5), peakRms: +peak.toFixed(5), peakDb: +(20 * Math.log10(peak + 1e-12)).toFixed(1) }); }, 4000);
    } catch (e) { res({ error: String(e) }); }
  }));
  // tap the sound button again -> muted
  const lf = page.locator(`figure.v:has(source[src="videos/${TEST[0]}.mp4"])`);
  await lf.scrollIntoViewIfNeeded();
  await page.locator('section.card').first().screenshot({ path: `${SHOTS}/m390_card1.png` });
  await page.locator('section.card').nth(3).screenshot({ path: `${SHOTS}/m390_card4_wan.png` });
  await page.screenshot({ path: `${SHOTS}/m390_full.png`, fullPage: true });
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
