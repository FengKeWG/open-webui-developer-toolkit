# Filter Instructions

This guide applies to all files in the `functions/filters` directory.

## Structure
- Each filter lives in its own folder matching the Python file name.
- A filter folder contains `README.md`, `CHANGELOG.md` and `<filter>.py`.

## Versioning

Each filter's docstring must include a `version:` field. We follow **Semantic Versioning** (`MAJOR.MINOR.PATCH`).

## Changelog Updates
When a filter is modified, update its `version` only once per day and record the changes under the same version in `CHANGELOG.md` using the format:

```
## [x.y.z] - YYYY-MM-DD
```

Documentation-only edits do not require a version bump.

## Code Style
Run `pre-commit` before committing changes to ensure the repository tests and style checks pass.
