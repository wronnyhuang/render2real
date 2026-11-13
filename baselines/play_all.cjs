// tap "Tap for sound" on EVERY video of the page at phone width; confirm it plays with decoded audio and is the only audible one.
const { chromium } = require('node_modules/playwright-core');
const fs = require('fs');
(async () => {
  const b = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: ['--autoplay-policy=no-user-gesture-required'] });
  const ctx = await b.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
  const p = await ctx.newPage();
  await p.goto('file://$OUT_DIR/render2real-poc/index.html');
  const n = await p.evaluate(() => document.querySelectorAll('figure.v').length);
  const res = [];
  for (let i = 0; i < n; i++) {
    const fig = p.locator('figure.v').nth(i);
    await fig.scrollIntoViewIfNeeded();
    const btn = fig.locator('button.snd');
    if (await btn.count() === 0) { res.push({ i, src: await fig.locator('source').getAttribute('src'), ok: false, why: 'no sound button' }); continue; }
    await btn.tap();
    let r = null;
    for (let k = 0; k < 20; k++) {
      await p.waitForTimeout(500);
      r = await p.evaluate((i) => {
        const v = document.querySelectorAll('figure.v')[i].querySelector('video');
        const aud = [...document.querySelectorAll('figure.v video')].filter(x => !x.muted && !x.paused).length;
        return { src: v.querySelector('source').getAttribute('src'), muted: v.muted, paused: v.paused, t: v.currentTime, bytes: v.webkitAudioDecodedByteCount, audible: aud, err: v.error ? v.error.code : null };
      }, i);
      if (!r.muted && !r.paused && r.t > 0.8 && r.bytes > 0) break;
    }
    r.i = i; r.ok = !r.muted && !r.paused && r.t > 0.8 && r.bytes > 0 && r.audible === 1 && !r.err;
    res.push(r);
    await btn.tap(); // mute again
    await p.evaluate((i) => document.querySelectorAll('figure.v')[i].querySelector('video').pause(), i);
  }
  const bad = res.filter(r => !r.ok);
  fs.writeFileSync('$WORK/play_all.json', JSON.stringify(res, null, 1));
  console.log(`figures ${n}; played with decoded audio ${res.length - bad.length}; failed ${bad.length}`);
  console.log('overflow px', await p.evaluate(() => document.documentElement.scrollWidth - window.innerWidth));
  for (const r of bad) console.log('BAD', JSON.stringify(r));
  await b.close();
})();
