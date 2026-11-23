# Offline experiment validation

The reusable analyzer and the complete q8 scale-register evidence case live in
[wangtianwen07/inference-validation-tools](https://github.com/wangtianwen07/inference-validation-tools).

It checks complete A0/B/A1 call matrices, recorded numerical checks, independent service measurements, and interval attribution. The q8 case preserves a positive local screen and an unconfirmed end-to-end result; it does not propose a kernel or serving change.

The bundled results are historical extracts. Original logs, checkpoint inputs and weights are not included; the analyzer does not rerun GPU checks or qualify the current TensorFold recipe.
