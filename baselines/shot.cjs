const { chromium } = require('node_modules/playwright-core');
(async () => {
  const b = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true });
  for (const [w, h, n] of [[390, 844, 'm'], [1200, 900, 'd']]) {
    const p = await b.newPage({ viewport: { width: w, height: h } });
    await p.goto('file://' + process.argv[2]);
    const el = await p.$('table.sb'); await el.scrollIntoViewIfNeeded(); await p.screenshot({ path: `${process.argv[3]}/${n}_sb.png` });
    const f = await p.$('figure.w4'); await f.scrollIntoViewIfNeeded(); await f.screenshot({ path: `${process.argv[3]}/${n}_h2h4.png` });
    const g = await p.$('.gridwrap ~ figcaption');
    const all = await p.$$('figure.w3'); for (const x of all) { if (await x.$('.cap.r4')) { await x.scrollIntoViewIfNeeded(); await x.screenshot({ path: `${process.argv[3]}/${n}_h2h3.png` }); break; } }
    console.log(n, 'overflow', await p.evaluate(() => document.documentElement.scrollWidth - window.innerWidth));
  }
  await b.close();
})();
