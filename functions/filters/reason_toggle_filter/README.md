# Reason Toggle Filter
Temporarily routes a request to another model.

## Valves

- `REASONING_EFFORT`: "low", "medium", "high", or "not set" (default). When set
  to a value other than "not set", the filter adds a `reasoning_effort` field to
  the request body.

Copy `reason_toggle_filter.py` to Open WebUI under **Admin ▸ Filters** to enable.
