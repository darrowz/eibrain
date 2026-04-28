# eiskills Integration

`eibrain` no longer owns skill semantics. The canonical skill layer is the
standalone `eiskills` package.

Local development:

```bash
python -m pip install -e ../eiskills-main
python -m pip install -e .
```

Production deployment should install `/dev-project/eiskills` into the same
runtime environment before starting `eibrain`.

Boundary:

- `eiskills`: intent-to-action skill compilation.
- `eibrain`: embodied IO, runtime orchestration, and hardware adapters.
- `eimemory`: unified long-term memory.

The legacy `eibrain.skills.*` import paths remain as compatibility adapters.
