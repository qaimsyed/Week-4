# Part B — Deep Learning Architecture Comparison

| Model | Task | Dataset | Test Metric | Params | Training Time* |
|---|---|---|---|---|---|
| ANN | Binary classification (structured/tabular) | Breast Cancer Wisconsin (569 rows, 30 features) | Accuracy 0.965 | ~1.6K | Seconds |
| CNN | Multi-class image classification | Digits (1,797 8×8 grayscale images) | Accuracy 0.969 | ~40K | Under a minute |
| SimpleRNN | Binary sequence classification | 30-day temperature windows | Accuracy 0.699 | ~1.2K | Under a minute |
| LSTM | Binary sequence classification | 30-day temperature windows | Accuracy 0.704 | ~4.5K | ~1–2 minutes |
| YOLOv8n | Object detection (pretrained, inference only) | 4 sample photos (COCO classes) | 9 objects detected, confidences 0.25–0.87 | 3.2M (pretrained) | No training — inference only, ~0.1s/image after warm-up |

\* Approximate, CPU-only, this sandbox's hardware — not comparable across
very different environments, but useful for relative ordering.

## Complexity and why each architecture fit its problem

**ANN** is the simplest architecture here and also solved its problem best,
because the breast-cancer features are already meaningful, hand-engineered
numeric measurements with no spatial or sequential structure to exploit — a
fully connected network is exactly matched to that kind of "flat table"
input.

**CNN** added convolution and pooling layers to exploit the 2D spatial
structure of pixel data (nearby pixels are related; a digit's shape is what
matters, not any one pixel's exact value). That structural assumption is
why it reached ~97% accuracy on a genuinely harder problem (10-way image
classification vs. 2-way tabular classification) with a similarly small
amount of training data.

**SimpleRNN and LSTM** tackled the hardest problem of the four: predicting
the *direction* of a noisy real-world time series seven days ahead, which is
inherently harder than either of the other tasks because daily temperature
has a large unpredictable (weather) component on top of its seasonal trend.
Both models landed close to 70% accuracy, with LSTM only marginally ahead of
SimpleRNN. That small a gap suggests the extra gating machinery in the LSTM
wasn't needed for this window length (30 days) — the useful signal for a
7-day-ahead forecast lives mostly in the recent temperature trend, which a
plain RNN can already capture. The gap would likely widen on longer
sequences (say, 100+ timesteps) where SimpleRNN's vanishing-gradient problem
becomes the limiting factor and LSTM's gates start to actually help.

**YOLOv8n** is architecturally the most complex model of the five (a
convolutional backbone plus detection heads that regress bounding boxes and
classify content in a single forward pass) but required zero training here
— it shipped pretrained on COCO and correctly found people, a bus, and a tie
across the sample images in well under a second per image on CPU. That
speed is the entire value proposition of a single-stage detector: it trades
a bit of raw accuracy (vs. two-stage detectors like Faster R-CNN) for the
real-time throughput that applications like video surveillance or robotics
need.

## Use-case fit, one line each

- **ANN** → any tabular business problem with engineered features: churn,
  credit risk, fraud flags.
- **CNN** → anything where "shape/pattern in a grid" is the signal: images,
  spectrograms, even board-game states.
- **RNN/LSTM** → anything ordered where earlier values inform later ones:
  time series, text, sensor streams.
- **YOLO** → anything needing fast, localized detection rather than a
  single whole-image label: counting objects, tracking, real-time
  monitoring.
