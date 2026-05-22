# Model Residency

This package manages model residency state for inference workloads.

Current scope:

- track loaded GPT/SoVITS model pairs
- record `loaded_at` and `last_used_at`
- protect models with `active_requests`
- identify idle-expired and overflow eviction candidates

Planned integration points:

- `InferenceService.load_models()`
- `InferenceService.inference()`
- gateway cleanup / manual unload endpoints
- optional background cleanup loop
