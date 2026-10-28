# Demo notes: Can a phone hear what you type?

## The one message
**It works in the lab. It falls to chance on a different device or laptop.**
Say it first, say it last. Everything else supports it.

## Set-up (2 minutes before)
- Open https://eshwarpotturi.github.io/keystroke-replication/ in a browser tab and share that tab only, so the meeting panel stays off screen.
- Zoom the page to about 150%.
- Keep the Dark mode button in reserve; light mode is the default.

## Flow (about 4 minutes)
1. **The experiment tab, top (60 s).** Read the headline. Point at the grid: each square is a key press the model never heard; green is heard correctly. Click "Phone to Zoom" above the grid and watch the grid turn red. Then the four cards on the right.
2. **The example (45 s).** Scroll to "You type a password". Left card is typed, right card is what the phone wrote. Click a key to show its sound picture.
3. **How well and Another device (75 s).** The bar chart: easiest test at the top, each harder test lower. Then the pairs chart: dark bar same device, red bar different device. Click a red bar for the run.
4. **Listen to a word (45 s).** Open the tab, press Listen on "password1", then press it again for different presses. Switch to "Phone, later presses".
5. **Close (15 s).** Repeat the one message. "How to read a headline about keyboard sound" gives the three questions to leave them with.

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
