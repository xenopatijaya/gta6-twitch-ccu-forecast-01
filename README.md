# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/xenopatijaya/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/xenopatijaya/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

An ensemble of three size-conditioned log-ratio models that forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day (19 Nov 2026, UTC), as five quantiles an hour, from the first hour's viewer count and the past big launches bundled in the case.

## Model

Each member is fitted per hour on the case's own comparables (past launches' release days, hours 1–24):

1. **Local-linear, bandwidth 0.75** — Gaussian-kernel-weighted regression of log(viewers at hour h) on log(hour-1 print), slope capped at ±1.5, spread from the kernel-weighted residual std.
2. **Local-linear, bandwidth 1.5** — the same with a wider kernel.
3. **Kernel mean/std, bandwidth 1.0** — kernel-weighted mean and std of log(viewers_h / hour-1), turned into normal quantiles.

The three members' quantiles are combined by geometric mean. The shape the data actually shows: the fitted slope is about 1.0 in the first hours (a bigger launch stays proportionally bigger) and falls to about 0.65 by hour 24 (bigger launches decay relatively faster late in the day), so a plain ratio forecast understates the late-day regression to the mean while a global OLS over-extrapolates it. Ramp-shape matching on the pre-day trajectory and extra features (pre-launch level, channels) were tried offline and rejected — they made the leave-one-out loss worse.

## Local validation

Leave-one-out over the 37 usable past launches in the kit's practice cases, scored with the contest's own log-pinball loss: ensemble **0.384** overall, **0.203** on launches with hour-1 ≥ 50k viewers, **0.207** on hour-1 ≥ 100k. The house persistence baseline scores 0.930 / 0.251 / 0.261 on the same set. These are local backtest numbers, not server grades; the practice grade lands on submission.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13 standard library only, no network, deterministic — two runs on the same case write byte-identical output, well inside the 8-second cap.

---

<sub>Entered on [make.fun](https://make.fun/?ref=fmkt9ms6&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
