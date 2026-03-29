# Examples

This directory is for small, compile-checked API demonstrations and a small number of
repo-local utilities that are easier to keep as Rust entrypoints than shell scripts.

Keep examples here only if they do at least one of:

- demonstrate a distinct public API surface
- show a minimal model/data shape that actually runs
- provide a maintained artifact-generation utility tied to repo code

Do not add near-identical stubs for every module. If multiple surfaces share the same
shape, prefer one consolidated example.

Current examples:

- `core_bdh_api.rs`: minimal BDH language forward pass
- `graph_compiled_executor_api.rs`: graph topology + compiled executor usage
- `stream_api.rs`: streaming/TBPTT metadata surface
- `vision_pyramid_api.rs`: minimal vision pyramid forward pass
- `multimodal_vl_jepa_api.rs`: minimal multimodal VL-JEPA forward pass
- `checkpoint_export_api.rs`: consolidated burnpack export API surface
- `fovea_artifacts.rs`: artifact-generation utility for fovea render comparisons
