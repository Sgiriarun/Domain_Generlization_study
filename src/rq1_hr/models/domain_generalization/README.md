# Domain-generalisation model components

This package will contain regression-compatible, source-only implementations of
the representative Phase-13 methods.

Shared interfaces must keep the encoder and HR head comparable across methods.
Every objective receives explicitly separated source-domain mini-batches and
must never load the held-out target domain.

Implement methods only after their definitions and adaptations are frozen in
`reports/phase13_dg_benchmark/00_method_audit/` and `01_protocol/`.

