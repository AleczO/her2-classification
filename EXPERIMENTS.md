

### Phase 1: Baseline Results (ResNet50)
| Epoch | Train Acc | Val Acc | Train Loss | Val Loss | Status/Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | 86.39% | 72.23% | 0.3601 | 0.9006 | Start |
| 7 | 93.32% | 93.77% | 0.1752 | 0.1533 | Stabilization |
| 11 | 94.13% | 94.68% | 0.1510 | 0.1505 | High Accuracy |
| 16 | 94.53% | 75.55% | 0.1344 | 1.1335 | **Instability (LR Spike)** |
| 22 | 95.32% | **95.50%** | 0.1215 | 0.1381 | **Best Baseline** |
| 29 | 95.94% | 82.68% | 0.1077 | 0.7143 | Divergence |
| 30 | 96.36% | 93.64% | 0.0933 | 0.2061 | End of Phase 1 |

<br>

__Phase 1: Baseline__ (ResNet50, LR=0.001)

- __Max Val Acc__: 95,50% (Epoch 22).
- __Final State (Ep 28)__: Train Acc 96,03% | Val Acc 94,05%.
- __Observations__: Model reached a plateau. Further training at $LR=0.001$ results in oscillations without accuracy gain.
- __Conclusion__: Transitioning to Fine-tuning ($LR=0.0001$) using the checkpoint from Epoch 22.