#!/usr/bin/env python3
"""Compatibility wrapper for the deterministic Phase 0 audit CLI."""

from temporal_ehr.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
