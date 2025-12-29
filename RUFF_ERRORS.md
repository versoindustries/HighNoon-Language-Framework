# Ruff Lint Errors Report

**Total errors: 182**

## Summary by Error Code

| Code | Count | Description |
|------|-------|-------------|
| B007 | 10 | Loop control variable `name` not used within loop body |
| B023 | 2 | Function definition does not bind loop variable `y_batch` |
| B904 | 28 | Within an `except` clause, raise exceptions with `raise ...  |
| E402 | 18 | Module level import not at top of file |
| E722 | 1 | Do not use bare `except` |
| E741 | 6 | Ambiguous variable name |
| F401 | 92 | `re` imported but unused |
| F402 | 1 | Import `field` from line 18 shadowed by loop variable |
| F841 | 9 | Local variable `l1_miss_rate` is assigned to but never used |
| UP035 | 15 | `typing.Tuple` is deprecated, use `tuple` instead |

## Detailed Errors

### B007 (10 errors)

- `benchmarks/bench_comparison.py:425` - Loop control variable `name` not used within loop body
- `highnoon/quantum/unified_bus.py:418` - Loop control variable `i` not used within loop body
- `highnoon/services/hpo_manager.py:1486` - Loop control variable `i` not used within loop body
- `highnoon/services/hpo_manager.py:1493` - Loop control variable `i` not used within loop body
- `highnoon/services/hpo_manager.py:1500` - Loop control variable `i` not used within loop body
- `highnoon/training/fisher_layer_grouper.py:535` - Loop control variable `group_id` not used within loop body
- `highnoon/training/inline_qpbt.py:216` - Loop control variable `i` not used within loop body
- `highnoon/training/quantum_loss.py:619` - Loop control variable `key` not used within loop body
- `highnoon/webui/app.py:3321` - Loop control variable `i` not used within loop body
- `scripts/analyze_codebase.py:148` - Loop control variable `h` not used within loop body

### B023 (2 errors)

- `benchmarks/optimizer_benchmark.py:348` - Function definition does not bind loop variable `y_batch`
- `benchmarks/optimizer_benchmark.py:348` - Function definition does not bind loop variable `x_batch`

### B904 (28 errors)

- `benchmarks/bench_perplexity.py:134` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `benchmarks/datasets.py:511` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/models/reasoning/latent_reasoning.py:548` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/retrieval/rag_module.py:192` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:1387` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:1389` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:1426` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:1428` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:1475` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:1477` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:1542` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:1544` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:3489` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:3603` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:3605` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:3621` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:3623` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:3741` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:3816` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:3891` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:3958` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:3960` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:3981` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:3983` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:3985` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:3987` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:4002` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
- `highnoon/webui/app.py:4410` - Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling

### E402 (18 errors)

- `highnoon/_native/ops/fused_flash_attention_op.py:43` - Module level import not at top of file
- `highnoon/_native/ops/fused_hnn_sequence/__init__.py:63` - Module level import not at top of file
- `highnoon/_native/ops/fused_hnn_step/__init__.py:68` - Module level import not at top of file
- `highnoon/models/layers/latent_kv_attention.py:48` - Module level import not at top of file
- `highnoon/models/tensor_layers.py:358` - Module level import not at top of file
- `highnoon/quantum/layers.py:591` - Module level import not at top of file
- `highnoon/services/hpo_trial_runner.py:49` - Module level import not at top of file
- `highnoon/services/hpo_trial_runner.py:52` - Module level import not at top of file
- `highnoon/services/hpo_trial_runner.py:71` - Module level import not at top of file
- `highnoon/webui/app.py:32` - Module level import not at top of file
- `scripts/debug_hpo_nan.py:53` - Module level import not at top of file
- `scripts/debug_hpo_nan.py:56` - Module level import not at top of file
- `scripts/debug_hpo_sweep_webui.py:48` - Module level import not at top of file
- `scripts/debug_hpo_sweep_webui.py:51` - Module level import not at top of file
- `scripts/debug_hpo_sweep_webui.py:54` - Module level import not at top of file
- `scripts/debug_hpo_sweep_webui.py:55` - Module level import not at top of file
- `scripts/debug_hpo_sweep_webui.py:58` - Module level import not at top of file
- `scripts/debug_hpo_sweep_webui.py:61` - Module level import not at top of file

### E722 (1 errors)

- `benchmarks/native_ops_bridge.py:466` - Do not use bare `except`

### E741 (6 errors)

- `benchmarks/bench_quantum.py:83` - Ambiguous variable name: `I`
- `benchmarks/optimizer_benchmark.py:369` - Ambiguous variable name: `l`
- `highnoon/models/layers/cayley_weights.py:155` - Ambiguous variable name: `I`
- `highnoon/services/hpo_multi_fidelity.py:262` - Ambiguous variable name: `l`
- `highnoon/training/fisher_layer_grouper.py:418` - Ambiguous variable name: `l`
- `highnoon/training/tensor_budget_controller.py:415` - Ambiguous variable name: `l`

### F401 (92 errors)

- `benchmarks/benchmark_cache.py:39` - `re` imported but unused
- `benchmarks/benchmark_cache.py:52` - `numpy` imported but unused
- `benchmarks/native_ops_bridge.py:439` - `highnoon.config.QMR_CHECKPOINT_STRATEGY` imported but unused; consider using `importlib.util.find_spec` to test for availability
- `highnoon/_native/ops/dtc_ops.py:33` - `typing.Optional` imported but unused
- `highnoon/_native/ops/entropy_regularization_ops.py:29` - `typing.Tuple` imported but unused
- `highnoon/_native/ops/fused_coconut_ops.py:29` - `typing.Tuple` imported but unused
- `highnoon/_native/ops/hyperdimensional_embedding.py:32` - `typing.Tuple` imported but unused
- `highnoon/_native/ops/lmwt_ops.py:30` - `typing.Optional` imported but unused
- `highnoon/_native/ops/lmwt_ops.py:30` - `typing.Tuple` imported but unused
- `highnoon/_native/ops/optimizers.py:29` - `typing.Tuple` imported but unused
- `highnoon/_native/ops/q_ssm_ops.py:34` - `typing.Tuple` imported but unused
- `highnoon/_native/ops/qmamba_ops.py:30` - `typing.Optional` imported but unused
- `highnoon/_native/ops/quantum_coherence_bus_ops.py:37` - `typing.Tuple` imported but unused
- `highnoon/_native/ops/quantum_dropout_ops.py:30` - `typing.Tuple` imported but unused
- `highnoon/_native/ops/quantum_galore_ops.py:33` - `typing.Tuple` imported but unused
- `highnoon/_native/ops/quantum_teleport_bus_ops.py:29` - `typing.Tuple` imported but unused
- `highnoon/_native/ops/specialized_quantum_ops.py:11` - `typing.Tuple` imported but unused
- `highnoon/data/hd_corpus.py:56` - `dataclasses.field` imported but unused
- `highnoon/data/streaming_tokenizer.py:50` - `collections.abc.Iterator` imported but unused
- `highnoon/models/hsmn.py:41` - `highnoon.config.MAX_CONTEXT_LEN` imported but unused
- `highnoon/models/layers/flash_linear_attention.py:64` - `highnoon._native.ops.fused_flash_attention_op.fused_quantum_inspired_features` imported but unused; consider using `importlib.util.find_spec` to test for availability
- `highnoon/models/layers/flash_linear_attention.py:66` - `highnoon._native.ops.fused_flash_attention_op.fused_random_maclaurin_features` imported but unused; consider using `importlib.util.find_spec` to test for availability
- `highnoon/models/layers/hd_superposition_embedding.py:52` - `highnoon.config.USE_HD_SUPERPOSITION_EMBEDDING` imported but unused
- `highnoon/models/layers/hyperdimensional_layer.py:60` - `highnoon._native.ops.hyperdimensional_embedding.holographic_bundle` imported but unused
- `highnoon/models/layers/hyperdimensional_layer.py:61` - `highnoon._native.ops.hyperdimensional_embedding.hyperdimensional_embedding_available` imported but unused
- `highnoon/models/moe.py:30` - `math` imported but unused
- `highnoon/services/hpo_trial_runner.py:41` - `highnoon.training.control_bridge.EvolutionTimeControlBridge` imported but unused; consider using `importlib.util.find_spec` to test for availability
- `highnoon/services/hpo_trial_runner.py:53` - `highnoon.config.GALORE_RANK` imported but unused
- `highnoon/services/hpo_trial_runner.py:54` - `highnoon.config.LITE_MAX_CONTEXT_LENGTH` imported but unused
- `highnoon/services/hpo_trial_runner.py:55` - `highnoon.config.LITE_MAX_MOE_EXPERTS` imported but unused
- `highnoon/services/hpo_trial_runner.py:56` - `highnoon.config.LITE_MAX_PARAMS` imported but unused
- `highnoon/services/hpo_trial_runner.py:57` - `highnoon.config.LITE_MAX_REASONING_BLOCKS` imported but unused
- `highnoon/services/hpo_trial_runner.py:60` - `highnoon.config.USE_HYPERDIMENSIONAL_EMBEDDING` imported but unused
- `highnoon/services/hpo_trial_runner.py:62` - `highnoon.config.USE_NEURAL_QEM` imported but unused
- `highnoon/services/hpo_trial_runner.py:63` - `highnoon.config.USE_NEURAL_ZNE` imported but unused
- `highnoon/services/hpo_trial_runner.py:65` - `highnoon.config.USE_QUANTUM_LM_HEAD` imported but unused
- `highnoon/services/hpo_trial_runner.py:66` - `highnoon.config.USE_QUANTUM_LR_CONTROLLER` imported but unused
- `highnoon/services/hpo_trial_runner.py:68` - `highnoon.config.USE_TENSOR_GALORE` imported but unused
- `highnoon/services/hpo_trial_runner.py:71` - `highnoon.training.quantum_lr_controller.QuantumAdaptiveLRController` imported but unused
- `highnoon/services/hpo_trial_runner.py:80` - `highnoon.training.quantum_loss.QuantumUnifiedLoss` imported but unused; consider using `importlib.util.find_spec` to test for availability
- `highnoon/services/hpo_trial_runner.py:81` - `highnoon.training.quantum_loss.QULSConfig` imported but unused; consider using `importlib.util.find_spec` to test for availability
- `highnoon/services/hpo_trial_runner.py:82` - `highnoon.training.quantum_loss.create_quls_from_hpo_config` imported but unused; consider using `importlib.util.find_spec` to test for availability
- `highnoon/services/hpo_trial_runner.py:1080` - `highnoon.config.SYMPFLOW_NUM_LEAPFROG_STEPS` imported but unused; consider using `importlib.util.find_spec` to test for availability
- `highnoon/services/quantum_hpo_scheduler.py:62` - `dataclasses.field` imported but unused
- `highnoon/services/quantum_hpo_scheduler.py:72` - `highnoon.services.hpo_metrics.HPOMetricsCollector` imported but unused
- `highnoon/services/quantum_hpo_scheduler.py:72` - `highnoon.services.hpo_metrics.TrialStatus` imported but unused
- `highnoon/services/quantum_hpo_scheduler.py:74` - `highnoon.services.hpo_utils.convert_numpy_types` imported but unused
- `highnoon/services/scheduler_factory.py:12` - `collections.abc.Callable` imported but unused
- `highnoon/services/sweep_executor.py:31` - `highnoon.services.hpo_utils.convert_numpy_types` imported but unused
- `highnoon/tokenization/vocab_controller.py:45` - `json` imported but unused
- `highnoon/training/barren_plateau_v2.py:48` - `dataclasses.field` imported but unused
- `highnoon/training/barren_plateau_v2.py:55` - `highnoon.config` imported but unused
- `highnoon/training/control_bridge.py:30` - `typing.Optional` imported but unused
- `highnoon/training/tensor_budget_controller.py:36` - `dataclasses.field` imported but unused
- `highnoon/training/tensor_budget_controller.py:40` - `tensorflow` imported but unused
- `highnoon/training/thought_distillation.py:49` - `highnoon.config.COCONUT_NUM_PATHS` imported but unused
- `highnoon/training/thought_distillation.py:49` - `highnoon.config.USE_CONTINUOUS_THOUGHT` imported but unused
- `highnoon/training/training_engine.py:58` - `abc.ABC` imported but unused
- `highnoon/training/training_engine.py:58` - `abc.abstractmethod` imported but unused
- `highnoon/training/training_loop.py:30` - `highnoon.config.BARREN_PLATEAU_MONITOR` imported but unused
- `highnoon/training/training_loop.py:31` - `highnoon.config.BARREN_PLATEAU_RECOVERY_LR_SCALE` imported but unused
- `highnoon/training/training_loop.py:38` - `highnoon.config.LITE_MAX_CONTEXT_LENGTH` imported but unused
- `highnoon/training/training_loop.py:39` - `highnoon.config.LITE_MAX_MOE_EXPERTS` imported but unused
- `highnoon/training/training_loop.py:40` - `highnoon.config.LITE_MAX_PARAMS` imported but unused
- `highnoon/training/training_loop.py:41` - `highnoon.config.LITE_MAX_REASONING_BLOCKS` imported but unused
- `highnoon/training/training_loop.py:43` - `highnoon.config.QNG_DAMPING` imported but unused
- `highnoon/training/training_loop.py:44` - `highnoon.config.USE_META_CONTROLLER` imported but unused
- `highnoon/training/training_loop.py:45` - `highnoon.config.USE_NEURAL_QEM` imported but unused
- `highnoon/training/training_loop.py:46` - `highnoon.config.USE_NEURAL_ZNE` imported but unused
- `highnoon/training/training_loop.py:47` - `highnoon.config.USE_QUANTUM_NATURAL_GRADIENT` imported but unused
- `highnoon/training/training_loop.py:59` - `highnoon.training.gradient_compression.GaLoreOptimizerWrapper` imported but unused
- `highnoon/training/unified_smart_tuner.py:543` - `highnoon.training.vqc_meta_optimizer.VQCTuningDecisions` imported but unused
- `highnoon/training/vqc_meta_optimizer.py:48` - `math` imported but unused
- `scripts/analyze_codebase.py:14` - `json` imported but unused
- `scripts/analyze_codebase.py:20` - `dataclasses.field` imported but unused
- `scripts/analyze_codebase.py:22` - `typing.Any` imported but unused
- `scripts/analyze_codebase.py:22` - `typing.Dict` imported but unused
- `scripts/analyze_codebase.py:22` - `typing.List` imported but unused
- `scripts/analyze_codebase.py:22` - `typing.Optional` imported but unused
- `scripts/analyze_codebase.py:22` - `typing.Set` imported but unused
- `scripts/analyze_codebase.py:22` - `typing.Tuple` imported but unused
- `scripts/debug_hpo_nan.py:30` - `dataclasses.field` imported but unused
- `scripts/debug_hpo_nan.py:56` - `highnoon.config` imported but unused
- `scripts/debug_hpo_sweep_webui.py:26` - `os` imported but unused
- `scripts/debug_hpo_sweep_webui.py:51` - `highnoon.config` imported but unused
- `scripts/debug_hpo_sweep_webui.py:62` - `highnoon.training.training_engine.EnterpriseTrainingConfig` imported but unused
- `scripts/debug_hpo_sweep_webui.py:302` - `highnoon.services.hpo_trial_runner.build_hsmn_model` imported but unused
- `scripts/debug_hpo_sweep_webui.py:460` - `datasets.IterableDataset` imported but unused; consider using `importlib.util.find_spec` to test for availability
- `scripts/validate_control_configs.py:23` - `json` imported but unused
- `scripts/validate_control_configs.py:26` - `os` imported but unused
- `scripts/validate_control_configs.py:28` - `tempfile` imported but unused
- `scripts/validate_control_configs.py:102` - `highnoon._native.ops.meta_controller_op.trigger_meta_controller` imported but unused; consider using `importlib.util.find_spec` to test for availability

### F402 (1 errors)

- `highnoon/telemetry/full_spectrum.py:152` - Import `field` from line 18 shadowed by loop variable

### F841 (9 errors)

- `benchmarks/benchmark_cache.py:842` - Local variable `l1_miss_rate` is assigned to but never used
- `highnoon/_native/ops/quantum_coherence_bus_ops.py:311` - Local variable `batch` is assigned to but never used
- `highnoon/_native/ops/specialized_quantum_ops.py:93` - Local variable `batch_size` is assigned to but never used
- `highnoon/_native/ops/specialized_quantum_ops.py:94` - Local variable `neurons` is assigned to but never used
- `highnoon/data/create_tfrecords.py:129` - Local variable `shard_writer` is assigned to but never used
- `highnoon/models/layers/local_attention.py:487` - Local variable `batch_size` is assigned to but never used
- `highnoon/services/hpo_importance.py:532` - Local variable `template` is assigned to but never used
- `highnoon/training/hd_activation_checkpoint.py:469` - Local variable `indices` is assigned to but never used
- `highnoon/training/quantum_loss.py:416` - Local variable `batch_size` is assigned to but never used

### UP035 (15 errors)

- `highnoon/_native/ops/entropy_regularization_ops.py:29` - `typing.Tuple` is deprecated, use `tuple` instead
- `highnoon/_native/ops/fused_coconut_ops.py:29` - `typing.Tuple` is deprecated, use `tuple` instead
- `highnoon/_native/ops/hyperdimensional_embedding.py:32` - `typing.Tuple` is deprecated, use `tuple` instead
- `highnoon/_native/ops/lmwt_ops.py:30` - `typing.Tuple` is deprecated, use `tuple` instead
- `highnoon/_native/ops/optimizers.py:29` - `typing.Tuple` is deprecated, use `tuple` instead
- `highnoon/_native/ops/q_ssm_ops.py:34` - `typing.Tuple` is deprecated, use `tuple` instead
- `highnoon/_native/ops/quantum_coherence_bus_ops.py:37` - `typing.Tuple` is deprecated, use `tuple` instead
- `highnoon/_native/ops/quantum_dropout_ops.py:30` - `typing.Tuple` is deprecated, use `tuple` instead
- `highnoon/_native/ops/quantum_galore_ops.py:33` - `typing.Tuple` is deprecated, use `tuple` instead
- `highnoon/_native/ops/quantum_teleport_bus_ops.py:29` - `typing.Tuple` is deprecated, use `tuple` instead
- `highnoon/_native/ops/specialized_quantum_ops.py:11` - `typing.Tuple` is deprecated, use `tuple` instead
- `scripts/analyze_codebase.py:22` - `typing.Dict` is deprecated, use `dict` instead
- `scripts/analyze_codebase.py:22` - `typing.List` is deprecated, use `list` instead
- `scripts/analyze_codebase.py:22` - `typing.Set` is deprecated, use `set` instead
- `scripts/analyze_codebase.py:22` - `typing.Tuple` is deprecated, use `tuple` instead

