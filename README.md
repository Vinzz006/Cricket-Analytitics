# CrickIQ — AI-Powered Cricket Analytics Platform

CrickIQ is a computer vision and language model system that analyzes batting and bowling technique from a single uploaded video or image. It extracts body landmarks, computes biomechanical metrics, and feeds them into **CrickLM** — a custom-trained language model built specifically for this project — to generate a structured technical report: weaknesses, tactical vulnerabilities, and corrective drills.

Most technique-analysis tools either stop at raw pose data with no interpretation, or hand an image to a general-purpose model with no biomechanical grounding. CrickIQ was built to avoid both: the numeric pose features are computed deterministically, and the interpretation layer is a model trained from scratch for exactly this domain rather than a general vision-language model repurposed for it.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [CrickLM](#cricklm)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Folder Structure](#folder-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Design Decisions](#design-decisions)
- [Roadmap](#roadmap)
- [Author](#author)

---

## Overview

A player uploads a short clip or image of a batting shot or bowling action. The backend extracts pose landmarks frame by frame using MediaPipe, derives quantitative biomechanical features from them, and passes the structured data to CrickLM, an in-house model trained to reason over pose metrics and produce coaching-grade feedback.

Built for three audiences:

- **Players**, who want specific, actionable feedback on their technique
- **Coaches**, who want a fast way to flag mechanical issues worth reviewing
- **Analysts**, who want quantified comparisons against professional benchmarks

---

## How It Works

1. A batting or bowling video/image is uploaded through the React frontend.
2. OpenCV samples the relevant frames from the input.
3. MediaPipe's pose landmarker (`pose_landmarker_full.task`) extracts 33 body landmarks per frame.
4. A feature extraction layer computes derived metrics: joint angles, bat swing arc, release point, head position, knee bend, foot placement, and overall balance.
5. A skeleton overlay is rendered on top of the original frame for visual reference.
6. The structured pose metrics are tokenized and passed to CrickLM for inference.
7. CrickLM returns a weakness report, tactical vulnerability zones, and corrective drills grounded in the specific metrics observed.
8. FastAPI serves the structured response to the frontend.
9. The full report can be exported as a PDF.

---

## CrickLM

CrickLM is the custom language model at the core of the analysis pipeline, trained specifically to reason over cricket biomechanics data rather than adapted from a general-purpose model.

```
Backend/cricklm/
├── model.py          model architecture
├── dataset.py         training data pipeline
├── tokenizer.py       domain-specific tokenizer for pose/biomechanics tokens
├── inference.py        inference-time loading and generation
├── checkpoints/        trained model weights
├── data/               training and evaluation data
└── logs/               training run logs
```

Training a domain-specific model rather than prompting a general one means the model's outputs are shaped entirely by cricket biomechanics data — joint angle ranges, swing mechanics, release points — instead of general-purpose priors. It also means inference runs locally without a dependency on a third-party API.

---

## Features

### Batting Analysis
Flags backlift issues, playing across the line, head imbalance, poor footwork, and weak technique on either side of the wicket.

### Bowling Analysis
Flags run-up alignment problems, front-arm inefficiency, inconsistent release points, knee bend issues, and flaws in side-on or front-on bowling actions.

### Comparison Mode
Benchmarks a player's mechanics against reference data modeled on professional players (Virat Kohli, Babar Azam, Jasprit Bumrah, Pat Cummins), using Euclidean distance, cosine similarity, and joint-angle variance to quantify the gap.

### Session History
Every analysis is stored, so players and coaches can track technique changes over time rather than looking at a single isolated report.

### PDF Export
Full reports — metrics, visuals, and model-generated recommendations — can be exported and shared outside the platform.

---

## Tech Stack

**Frontend**
React, TypeScript, TailwindCSS, Framer Motion, GSAP, Lenis, Shadcn UI

**Backend**
FastAPI, Python, OpenCV, MediaPipe, NumPy, Pydantic

**AI Layer**
CrickLM — a custom-trained language model for biomechanical analysis (see [CrickLM](#cricklm))

**Database**
SQLite (local/dev), Firebase (production)

**Deployment**
Vercel (frontend), Render (backend)

---

## System Architecture

```
Input Layer (video/image upload)
        │
        ▼
Preprocessing (frame extraction, normalization)
        │
        ▼
Pose Estimation (MediaPipe — 33 landmarks)
        │
        ▼
Feature Engine (joint angles, swing arc, release point, balance)
        │
        ▼
CrickLM Inference (custom-trained model — tokenized pose data in, report out)
        │
        ▼
Report Generation (weaknesses, drills, comparisons)
        │
        ▼
Frontend Visualization (dashboard, overlays, PDF export)
```

Each stage is decoupled: the pose estimation and feature engine can be tested and improved independently of CrickLM, and CrickLM itself can be retrained or swapped without touching the CV pipeline.

---

## Folder Structure

```
Cric-analytics/
├── .vscode/
│   ├── launch.json
│   └── settings.json
│
├── Backend/
│   ├── cricklm/
│   │   ├── checkpoints/
│   │   ├── data/
│   │   ├── logs/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── inference.py
│   │   ├── model.py
│   │   ├── pose_landmarker_full.task
│   │   └── tokenizer.py
│   ├── main.py
│   ├── pyrightconfig.json
│   └── requirements.txt
│
└── Frontend/
    ├── dist/
    ├── node_modules/
    ├── src/
    │   ├── App.tsx
    │   └── main.tsx
    ├── .gitignore
    ├── index.html
    ├── package-lock.json
    └── package.json
```

---

## Getting Started

### Backend

```bash
cd Backend
pip install -r requirements.txt
uvicorn main:app --reload
```

CrickLM checkpoints are loaded automatically from `Backend/cricklm/checkpoints/` at startup. See `Backend/cricklm/` for training and inference scripts if you want to retrain on your own data.

### Frontend

```bash
cd Frontend
npm install
npm run dev
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/analyze/image` | Run full analysis on a single uploaded image |
| POST | `/analyze/video` | Run full analysis on an uploaded video clip |
| POST | `/compare` | Compare a player's metrics against a professional benchmark |
| GET | `/report/{id}` | Retrieve a previously generated report |
| GET | `/history` | Retrieve a user's past analysis sessions |

---

## Design Decisions

- **Why train CrickLM instead of calling a general-purpose vision-language API.** General models are fluent but not grounded in cricket-specific biomechanics — they'll describe what they see without necessarily knowing what a good backlift or a clean release point actually looks like in numeric terms. Training on structured pose data and biomechanics labels means the model's outputs are shaped by domain data rather than general priors, and inference doesn't depend on a third-party API or its rate limits.
- **Why separate the feature engine from CrickLM.** Biomechanics metrics (joint angles, balance, release point) are useful on their own, independent of any model interpretation. Keeping them as a standalone module means the platform still produces measurable value even while CrickLM is being retrained or improved.
- **Why comparison mode uses distance metrics instead of a model call.** Euclidean distance and cosine similarity on joint-angle vectors are fast, deterministic, and explainable — important for a feature where users need to understand exactly why a gap score was assigned.

---

## Roadmap

- Real-time webcam analysis
- Ball trajectory prediction
- Injury risk detection based on mechanical strain patterns
- Automated shot classification
- Bowling speed estimation from frame timing
- Mobile app
- Expanding CrickLM's training data with more professional reference footage

---

## Author

Built by **A Rohan**
AI Engineer, ML Enthusiast, Full Stack Developer