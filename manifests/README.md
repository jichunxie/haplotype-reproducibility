# Release manifests

- `code_inventory.csv`: one row per analysis script, ordered by stage and labeled by current status,
  environment, inputs, outputs, and role.
- `result_map.csv`: one row per manuscript display, linking its output to the display builder, locked
  artifacts, upstream analysis, compute job, and access class.
- `sha256.json`: immutable checksums used by `make verify` for production scripts, environments, and
  released aggregate/display inputs.

When a checksummed source file changes only in documentation or comments, update its digest and
record that the executable behavior is unchanged. When numerical logic changes, create a new
versioned script/artifact rather than overwriting the production record.

