# Portfolio figure

The stage diagram complements the real-scene video above; the paired bars show how pose correctness changes the end-to-end metrics.

## Reproduce

From the repository root:

```bash
python -m pip install -r docs/portfolio/requirements.txt
python docs/portfolio/render.py
```

The renderer verifies source SHA-256 hashes (CRLF normalized to LF) before plotting the reviewed values in `figure.json`. If a source changes, review and refresh the snapshot before regenerating. It writes SVG and PNG with matching content.

## Sources

- [release/v1.1.0/results.json](../../release/v1.1.0/results.json)

The flow/memory/timing illustrations are schematics. Only explicitly labeled measurements represent recorded experiments. Confidence intervals are copied from source reports, not recomputed.
