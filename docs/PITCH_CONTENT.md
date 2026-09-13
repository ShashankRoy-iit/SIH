# SAHYOG · SIH 2026 pitch content (speaker script)

Slide-by-slide speaking notes. The deck (`docs/SAHYOG_SIH2026_Winning_Deck.pptx`) carries the same notes on each slide, so this file and the deck cannot drift.

## Slide 1 · `s_title`

Open on the promise, not the title. 'In a flood, a survivor on a rooftop has about an hour before hypothermia or the water wins. Search teams cannot see her, GPS is unreliable and the network is down. SAHYOG is a drone that finds her by body heat — and the intelligence that does it flies on an ordinary Android phone.' Pause. Then slide 2.

## Slide 2 · `s_problem`

Rubric point 1. Do not rush this slide — the judges must feel the hour. 'Four numbers define our problem. Sixty minutes: after the golden hour, survival drops fast. Four to ten pixels: that is a human being in a thermal image from survey height — this is why naive AI fails. Zero: no network, no reliable GPS, no roads. And three lakh rupees: what the usual thermal-plus-computer payload costs per aircraft. We attack all four.'

## Slide 3 · `s_insight`

This is the slide that makes judges lean in. 'Two facts drive every design decision we made. First: at survey altitude a survivor is smaller than one grid cell of a standard detector — we measured 0.0009 mAP, effectively blind. One architecture change, a stride-4 head, took it to 0.234. Second: the most expensive subsystem in every competitor's budget is already in every rescuer's pocket. We fly it.'

## Slide 4 · `s_solution`

Rubric point 2, the overview. Walk the diagram left to right in one breath: cameras → physics gates → neural detector → fusion → belief map → decision → LoRa → ground station. Emphasise the last line: nothing in the loop needs the internet. Then: 'Six capabilities, one airframe, and every model runs on the device.'

## Slide 5 · `s_phone`

Your signature slide — spend a full minute here. 'Every team in this room will price a companion computer. We asked a different question: what does a drone need that a phone does not already have? The answer is nothing on this list. We measured it: 15 fps video, 30 Hz odometry, 69 milliseconds worst-case gap, link streaming — on a phone hanging off a USB cable. And because the phone's DSP is the same family as the Qualcomm board this problem statement targets, we train one model and deploy it twice.'

## Slide 6 · `s_physics`

Second USP. 'Anyone can bolt a YOLO onto a drone. Ours cannot lie: every detection passes three physics gates before it becomes a report — is the temperature a body's, is the footprint a body's, and at 22 metres, is it still there? That is why our measured precision is 1.000 while a detector without the gates produced a false alarm every sixth frame. In rescue, a false alarm is not an error metric — it is a team climbing the wrong rooftop.'

## Slide 7 · `s_search_see`

Keep this to forty seconds. Left: the search is planned so that coverage is provable — lane spacing comes from the sensor swath and a detection-probability target. Right: what the detector actually sees, thermal and RGB side by side with its boxes. Mention the honest ceiling: published aerial-thermal mAP50 tops out near 0.55; we quote it instead of inventing 0.95.

## Slide 8 · `s_denied`

Thirty seconds. 'Floods take GPS and the network first. So neither is load-bearing. Navigation falls back to visual-inertial odometry from the phone plus lidar, and the autopilot's EKF is told which source to trust, live. Reports go out as 200-byte verified packets over LoRa — store, forward, acknowledge. A saturated link means fewer reports. It never means silence.'

## Slide 9 · `s_rescue`

Twenty-five seconds. 'Finding her is half. The drone drops flotation on a confirmed survivor, and in the same packet the ground team gets an A* route across the hazard field to reach her. And above all of it sits a safety supervisor with twelve ways to say no — because a rescue drone that crashes into a crowd has added to the disaster.'

## Slide 10 · `s_proof`

Rubric point 6 — slow down here. 'We are not showing you a diagram of what we will build. One hundred eighty tests pass. Precision 1.000 is measured, not claimed. The phone bench ran on a real phone. The INT8 exports sit behind a gate that refuses them if they lose recall. And when something is not done — field flights — we say so on the slide.' Judges reward the team that knows exactly where its own truth ends.

## Slide 11 · `s_feasibility`

Rubric point 3. 'Nothing here is invented. Every line is a part you can order this week, and every line has a measured number next to it — hover power we measure on our airframe, not from a datasheet. And the design rule that makes it feasible: every layer has a fallback, so a missing component degrades the mission instead of ending it.'

## Slide 12 · `s_roadmap`

Rubric 3 continued. Thirty-second roadmap with named risks — judges trust a team that has already thought about what will go wrong. Close with the open release: 'whatever we fly, the next team inherits.'

## Slide 13 · `s_impact`

Rubric point 4. End on scale, not specs: 'India does not need one rescue drone. It needs one in every district control room — and that only happens at phone-money, not at research-money. Same airframe, same code, different disaster. And because it is Qualcomm silicon from the phone to the flight board, the ecosystem this problem statement wants is the ecosystem we already build on.'

## Slide 14 · `s_team`

Rubric point 5 — fifteen seconds, names and ownership only. The line that lands: 'every claim on these slides traces to an artefact in our repository.' Then move; do not linger on yourselves.

## Slide 15 · `s_ask`

Close in twenty seconds and stop talking. 'Three purchases and one permission. Everything else — the detectors, the physics gates, the phone bridge, the safety supervisor, the ground station — is in the repository and running on this laptop right now, if you would like to see it.' Then offer the demo. Silence is confidence.
