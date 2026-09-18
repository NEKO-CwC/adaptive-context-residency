"""Engine-side artifacts (loaded *inside* the vLLM container, not by the simulator).

* `policy.py` — out-of-tree `CachePolicy`, loadable via
  `eviction_policy` + `cache_policy_module_path` in `kv_connector_extra_config`.
* `launch.py` — renders the `--kv-transfer-config` JSON for a given host/engine sizing.

Nothing here imports vLLM at module load time except `policy.py`'s guarded import, so the
package stays importable on a laptop without the engine installed.
"""
