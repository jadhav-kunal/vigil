"""Env-gated sponsor adapters (spec 4.9). Each is a small, isolated, individually testable
module with a factory that returns None when its key is absent, so the proxy runs identically
with or without any of them (Invariant I2). No adapter is imported at module load of the hot
path; SDK imports are guarded so an absent SDK can never break local mode."""
