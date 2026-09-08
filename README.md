# Week 4 — Forecasting Models & Deep Learning Architectures

Career Launchpad Program — AI/DS Internship Track
Instructor: Zaheer Ahmad, Lecturer, Computer Science

## What's in this package

```
Week4/
├── README.md                  <- you are here
├── requirements.txt            <- exact library versions used
│
├── Part_A/                     <- ARIMA vs Prophet forecasting
│   ├── data/
│   │   └── daily-min-temperatures.csv
│   ├── ARIMA_Prophet_Forecasting.ipynb   <- main notebook, fully executed
│   ├── part_a_forecasting.py             <- same logic as a plain script
│   ├── part_a_nb.py                      <- notebook source (jupytext format)
│   ├── comparison_table.csv              <- MAE / RMSE / MAPE for both models
│   ├── adf_test_result.txt               <- stationarity test output
│   ├── arima_summary.txt                 <- statsmodels ARIMA fit summary
│   ├── analysis.md                       <- 250-300 word written comparison
│   └── plots/                            <- all EDA + forecast plots (PNG)
│
└── Part_B/                     <- ANN, CNN, RNN, YOLO
    ├── ann/
    │   ├── ANN_BreastCancer.ipynb
    │   ├── ann_report.txt
    │   └── ann_training_curves.png
    ├── cnn/
    │   ├── CNN_Digits.ipynb
    │   ├── cnn_report.txt
    │   ├── cnn_sample_images.png
    │   └── cnn_training_curves.png
    ├── rnn/
    │   ├── RNN_LSTM_Temperature.ipynb
    │   ├── rnn_report.txt
    │   ├── rnn_comparison_curves.png
    │   └── rnn_confusion_matrices.png
    ├── yolo/
    │   ├── YOLO_Inference.ipynb
    │   ├── images/              <- 4 sample input photos
    │   ├── results/              <- annotated (bounding-box) output images
    │   ├── detections.csv
    │   └── inference_times.csv
    └── summary_report.md         <- architecture comparison across all 5 models
```

Every notebook in this package was actually run top-to-bottom (not just
written) — the outputs, plots, and metrics you see inside them are real
results from this run, not placeholders.

## Datasets used, and why

| Part | Dataset | Why this one |
|---|---|---|
| A (ARIMA/Prophet) | [Daily min temperatures, Melbourne 1981-1990](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv) — 3,650 rows | Real univariate time series, well over the 500-row/2-year minimum, with a clean yearly seasonal cycle that makes the ARIMA-vs-Prophet comparison meaningful. |
| B1 (ANN) | scikit-learn's Breast Cancer Wisconsin dataset — 569 rows, 30 features | Same shape of problem as the suggested churn/loan/diabetes datasets (structured numeric features → binary label), built into scikit-learn with no download needed. |
| B2 (CNN) | scikit-learn's `digits` dataset — 1,797 8×8 grayscale images | The dev sandbox this was built in only has network access to a short allow-list of domains (PyPI, GitHub, npm) for security, so it can't reach the servers CIFAR-10/Fashion-MNIST auto-download from. `digits` is a genuine image dataset with zero download required. The CNN architecture (Conv2D → pool → dropout) is identical to what you'd use on CIFAR-10 or Fashion-MNIST — only the input shape and training time change if you swap datasets. |
| B3 (RNN/LSTM) | The same Melbourne temperature series, reframed as 30-day sliding windows → "will the temperature 7 days later be higher or lower?" | Same reachability constraint ruled out auto-downloading IMDB sentiment data. This keeps the RNN task on a real, already-verified dataset while still being a genuine sequence-in/label-out classification problem. |
| B4 (YOLO) | 4 real photos from Ultralytics' own public sample-asset repo (bus, zidane, airport, boats) | Pretrained YOLOv8n weights download automatically from GitHub Releases (reachable in the sandbox); the sample photos come from the same trusted source ultralytics ships in its own quick-start docs. |

**One limitation worth flagging honestly:** the YOLO task also asks for a
short video/webcam clip. This sandbox has no camera and no reachable
video-hosting service, so only image inference was run. The inference code
is identical for video (Ultralytics treats a video as a sequence of frames
through the same call) — the exact drop-in code is in a comment at the end
of `YOLO_Inference.ipynb`.

## Results at a glance

**Part A:**

| Model | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| ARIMA(3,0,1) | 3.31 | 4.02 | 45.5 |
| Prophet | 2.36 | 2.90 | 32.1 |

Prophet wins because it explicitly models the yearly seasonal cycle, which
matters a lot at this forecast horizon (730 days out) — full reasoning in
`Part_A/analysis.md`.

**Part B:**

| Model | Task | Test Accuracy |
|---|---|---|
| ANN | Breast cancer classification | 96.5% |
| CNN | Digit image classification | 96.9% |
| SimpleRNN | Temperature direction (7-day) | 69.9% |
| LSTM | Temperature direction (7-day) | 70.4% |
| YOLOv8n | Object detection (COCO) | 9 objects across 4 images, ~0.1s/image |

Full comparison and reasoning in `Part_B/summary_report.md`.

## How to reproduce

```bash
pip install -r requirements.txt
jupyter notebook   # open any .ipynb and Run All
```

Each notebook is self-contained except `RNN_LSTM_Temperature.ipynb`, which
reads the CSV from `../../Part_A/data/` — keep the folder structure intact
when re-running.

## Before you submit

The task manual asks for the ZIP to be named `FirstName_LastName_Week4`.
This package is named `Qaim_Week4` — rename the folder/zip to include your
surname before uploading to the LMS/Drive link.
