# Demo notes: Can a phone hear what you type?

## The one message
**It works in the lab. It falls to chance on a different device or laptop.**
Say it first, say it last. Everything else supports it.

## Set-up (2 minutes before)
- Open https://eshwarpotturi.github.io/keystroke-replication/ in a browser tab and share that tab only, so the meeting panel stays off screen.
- Zoom the page to about 150%.
- Keep the Dark mode button in reserve; light mode is the default.

## Flow (about 4 minutes)
1. **The claim (30 s).** A 2023 paper says a phone, or a Zoom call, can recover typed keys with 95% and 93% accuracy. Point at the three tiles.
2. **Try it (60 s).** In the panel, leave "password1" and click Listen. Read out what the phone "heard". Click one wrong key to show the sound picture and the model's top three guesses. Press Listen again: different presses, different result.
3. **We reproduced it (30 s).** Results row: phone 93.0%, Zoom 91.1% over three runs each. The claim holds in the paper's own setup.
4. **Where it bends (60 s).** Click "Stricter split": same panel, later presses, about 84 to 86%. Then the two transfer rows: trained on one device, tested on the other, about 4.5%. Chance is 2.8%.
5. **Close (30 s).** Repeat the one message. Ask for reactions.

## Questions you may get, and plain answers
| Question | Answer |
|---|---|
| Is the try-it live audio? | No. It replays held-out presses from the authors' recordings. The model never saw them in training. |
| Why is your number below the paper's? | Smaller model, 150 epochs on a CPU, and averages over several runs. The paper's single figure sits inside our range. |
| Is this a real-world risk then? | In these recordings the attack needs training data from the same device and room. We did not test a stranger's keyboard. |
| Does it fail across laptops too? | Yes in our test: 4.9% one way and 2.9% the other across two MacBook Pros, against 59% and 67% on each laptop's own recordings. Rooms and microphones also differ, so this is a hint, not proof. |
| Who is it for? | Nobody yet. It is a replication study. The honest use is as a reality check on a security headline. |

## What not to claim
- Do not say it works against your own audio.
- Do not call the transfer result exact: each direction is one run.
- Do not say a client has asked for this. None has.
