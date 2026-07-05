# Benchmark Submission

Thanks for adding a data point. The matrix is only as good as its provenance —
please confirm each item.

## Files

- [ ] One `benchresult/v1` JSON per run under `results/community/`
- [ ] The `plan/v1` file used, under `plans/` (or a link to the committed one)
- [ ] Your hardware profile under `hardware_profiles/` (detected preferred; manual template ok)

## Provenance checklist

- [ ] `pins.model_sha256` matches a pinned artifact in `model_profiles/` (or your PR adds the pin with source URL)
- [ ] `pins.runtime_commit` is the exact commit/build of the runtime you ran
- [ ] `pins.plan_hash` was produced by `frontier bench` (not hand-edited)
- [ ] `frontier validate .` passes locally
- [ ] Numbers are from the machine described — no borrowed or estimated values

## Reproduction status

- [ ] Single run (will be labeled `claimed`)
- [ ] I reproduced my own numbers and listed the prior `result_id` in `reproductions`
- [ ] This run reproduces someone else's result (`--repro-of <their result_id>`)

## Environment notes

Anything that affects interpretation: WSL2, thermal conditions, background
load, non-default runtime flags, power mode.
