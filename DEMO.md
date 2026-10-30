# Demo script (about 5 minutes)

**One message:** 25 presses per key is enough to name over 90% of keys from sound, and the model knows when it is right.

**Set-up:** open https://eshwarpotturi.github.io/keystroke-replication/ in one tab, zoom ~150%, sound on.

1. **The experiment tab.** Read the headline. Point at the four tiles: 93% phone, 91% Zoom, 98% right when 80%+ sure, 97.5% top-three.
2. **One real example.** Play the "password1" presses, then show the model's reading beside them.
3. **Knows when it knows.** Click the 80% bar: 246 of 251 right.
4. **Listen to a word.** Type a word, press Listen, hear the presses and see each guess.
5. **What you can use it for.** Device audits, hardware comparison, confidence gates. Say these are possible uses.
6. **Where it fits.** Train on the device you want to measure; show the pair chart if asked.

| Question | Answer |
|---|---|
| Is this your method? | The method and data are from the published paper (arXiv 2308.01074); we re-implemented it and ran it ourselves. |
| How big is the test? | 90 unseen presses per run, several runs; see "Every run". |
| Other devices? | Tune it per device; cross-device runs are on the page. |

**Keep accurate:** one person, one laptop, one room; uses are proposals, no client has asked for it.

## Live version (the "Live" tab)

Anand asked to see it live: type ~50 words, then guess a password from the
sound alone. It runs for real — mic in, model in the browser, nothing sent
anywhere — but with one honest constraint carried over from the findings
above: **it only works on the device/mic it was trained on** (accuracy
collapses to ~4% on a different device). So "a stranger types on an
untrained laptop and it guesses their password" isn't what this does. What
it does: run it on the laptop and mic the model was trained on, type the
passage yourself (or hand it to someone at that same machine), and the
reveal is real — the model has genuinely never seen that password before.

1. Open the **Live (your mic)** tab, pick the matching device chip, press
   **Start microphone**, allow the mic.
2. Type a sentence or two naturally in the warm-up box — each letter gets a
   live guess with a short (~1s) lag while it waits for the key's own sound
   to finish recording.
3. Once the password field appears, type a short password (hidden, not
   echoed to the screen) and press **Reveal what the mic heard**.

If someone asks why you can't just hand them the keyboard cold: say so
plainly — same reason a different laptop scores ~4% in "Where it fits" on
this page. The honest live demo is "it reads me back, on my machine,
something it was never shown" — which is still the real finding.
