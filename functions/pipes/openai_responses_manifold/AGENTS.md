# Pipeline Instructions for OpenAI Responses Manifold

This guide applies to all files in this directory.

## Versioning

`openai_responses_manifold.py` contains a `version:` field in its top level docstring.
We follow **Semantic Versioning** (`MAJOR.MINOR.PATCH`):

- **MAJOR** – incompatible or breaking changes.
- **MINOR** – backward compatible feature additions.
- **PATCH** – bug fixes or internal changes that keep existing behaviour.

## Changelog Updates
When `openai_responses_manifold.py` is modified, update the `version` field only once per day. All changes made on the same day should be grouped under a single version in `CHANGELOG.md` using the format:

```
## [x.y.z] - YYYY-MM-DD
```

If additional changes are made later on the same day, update the existing entry instead of creating a new version. Only increment the version number when changes are made on a new day or when a significant milestone warrants it.


Documentation-only edits do not require a version bump.

## Code Style and Testing

Run `pre-commit` before committing changes. This executes **ruff** and the
pipeline's unit tests to ensure style and importability remain intact.

## README

If you add new valves or features, update the features table in `README.md` so
documentation stays in sync with the source file. Each feature row has a
`Last updated` column showing when that row's status or notes last changed.
Update that date whenever you modify a feature so we know exactly when each
capability was most recently reviewed.
