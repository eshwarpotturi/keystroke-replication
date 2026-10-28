# Demo notes: Can a phone hear what you type?

## The one message
**It works in the lab. It falls to chance on a different device or laptop.**
Say it first, say it last. Everything else supports it.

## Set-up (2 minutes before)
- Open https://eshwarpotturi.github.io/keystroke-replication/ in a browser tab and share that tab only, so the meeting panel stays off screen.
- Zoom the page to about 150%.
- Keep the Dark mode button in reserve; light mode is the default.

## Flow (about 4 minutes)
1. **Overview (45 s).** Read the headline. Point at the four tiles from left to right: 93% over a phone, 91% over Zoom, about 85% on later presses, about 4.5% on a different device. Chance is 2.8%.
2. **Listen (90 s).** Click Listen, leave "password1", press Listen. Read what the phone "heard". Click a red key to show its sound picture and the model's top three guesses. Press Listen again: different presses, different result. Switch to "Later presses only" to show the score falling.
3. **Results (60 s).** Open Results. Walk down the bars: the two blue bars near the top, then the short ones. Click the laptop row to show every run.
4. **Close (30 s).** Repeat the one message. Ask for reactions.

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
