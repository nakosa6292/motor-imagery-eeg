# Motor-Imagery EEG Classification & Prosthetic Simulation

**Project status:** completed, confirmed by Noah on November 6, 2026. The project was tested and worked with online recorded EEG data. Noah’s lower-cost NeuroSky MindWave headset was not compatible with the motor-imagery pipeline; live control from that device is not claimed.

Python experiments comparing log-variance/LDA, CSP/SVM, and a TensorFlow/Keras EEGNet-style convolutional network for four-class motor-imagery EEG. Saved predictions drive Matplotlib arm and stick-figure animations.

## Supplied source

| File | Role |
| --- | --- |
| `load_data.py` | Loads a local GDF recording and displays EEG channels. |
| `load_from_moabb.py` | Loads subject 1 through MOABB, applies an 8–30 Hz filter, extracts log-variance features, and trains/evaluates LDA. |
| `train_csp_model.py` | Six-component CSP with an RBF SVM in a scikit-learn pipeline. |
| `csp_feature_select.py` | Bandpass filtering, 12 CSP components, selection of six features, and linear SVM. |
| `cnn_eegnet.py` | EEGNet-style CNN training/evaluation using TensorFlow/Keras; saves the model and held-out input array. |
| `cnn_visualizer.py` | Displays model predictions as four screen positions. |
| `prosthetic_simulator.py` | Single-segment arm response animation. |
| `prosthetic_simulator2.py` | Two-link arm with smoothed joint-angle transitions. |
| `prosthetic_stick_figure.py` | Maps predicted classes to stick-figure actions. |
| `realtime_visualizer.py` | Displays live MindWave attention, meditation, and signal readings. |
| `stream_mindwave.py` | Serial packet-stream diagnostic. |
| `watch_ports.py` | Decodes and prints headset values, raw EEG, and band-power fields. |

The three live-headset utilities belong to this project as directed by Noah, including `stream_mindwave.py`. Their presence does not establish that the trained multichannel model processes the live headset stream.

## Data and model artifacts

The full private bundle includes `eegnet_model.h5`, `X_test.npy`, `data/A01E.gdf`, and `data/A09T.gdf`. The GDF files are examples from a larger dataset collection, not the complete data used in the project. The smaller website download includes source and documentation; obtain the binary artifacts through the full bundle linked on the review page.

`X_test.npy` was read without pickle: shape `(116, 22, 1001, 1)`, dtype `float32`, all entries finite. Both sample GDF headers identify GDF 1.99. The model was preserved and hashed but was not loaded or evaluated.

## Environment and running

Imports identify NumPy, SciPy, scikit-learn, MNE, MOABB, TensorFlow, Matplotlib, and pyserial. Original Python/package versions were not supplied; this is a dependency inventory, not a tested environment specification.

From this project directory, the training scripts can be invoked individually, e.g. `python train_csp_model.py` or `python cnn_eegnet.py`. They request subject 1 from `BNCI2014001` through MOABB and may download data. The included GDF examples are not wired into those training scripts.

To replay a saved model, keep `eegnet_model.h5` and `X_test.npy` in the current working directory and run a visualizer such as `python prosthetic_simulator2.py`. This procedure follows the code's paths; runtime compatibility remains unverified. Hardware diagnostics require a connected headset and configured serial port. Scripts execute their work immediately when run or imported.

## Evaluation and review findings

- The submitted CNN uses TensorFlow/Keras, not PyTorch. It runs 20 epochs with batch size 16 and a 10% validation fraction from its training set.
- Classification scripts use subject 1 with a random 80/20 trial split and `random_state=42`. This does not establish held-out-subject or held-out-session performance.
- Training uses `LabelEncoder`; visualizers hard-code class meanings. The fitted class order is not saved. Save/reuse `encoder.classes_` before relying on the displayed movement labels.
- No held-out labels, evaluation report, or confusion matrix was supplied, so the model and test input alone cannot verify accuracy.
- `load_data.py` expects `data/A01T.gdf`, which is not included. The uploaded `A01E.gdf` and `A09T.gdf` are retained under their original names.
- `prosthetic_simulator2.py` smooths joint angles. The supplied animations precompute predictions from saved trials; confidence gating and measured live inference latency are not established by these files.
- Original dataset source/attribution and redistribution terms should accompany any eventual public data release. The sample files remain private in this review.

## Verification and release

All 12 Python files passed syntax parsing on November 6, 2026. Training, inference, graphics, and hardware paths were not executed; this environment lacks the project ML dependencies and connected hardware. All uploaded bytes are preserved with SHA-256 checksums in the manifest. No code fixes, license assignment, or public publication have been made.
