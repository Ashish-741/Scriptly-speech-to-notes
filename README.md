# Scriptly-speech-to-notes
this is the jovac/mini project of group 9


# 🎤 Scriptly – Speech to Notes

Scriptly converts spoken words into structured, searchable notes.  
---

## 📚 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture & Tech Stack](#architecture--tech-stack)
- [Team Contributions](#team-contributions)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Future Work](#future-work)

---

## Overview
Scriptly is our college mini-project demonstrating a complete pipeline:
1. Real-time speech transcription using **Faster-Whisper**.
2. Keyword extraction and classification using **TF-IDF + ML models**.
3. A clean front-end to display notes to users.

---

## ✨ Features
- 🗣 **Speech to Text** – Pre-trained Faster-Whisper model for fast transcription.
- 🔑 **Keyword Extraction** – Balanced dataset, TF-IDF vectorization, Logistic Regression & Random Forest.
- 🖥 **Catchy UI** – HTML & CSS interface to view and copy notes.
- 📊 **Classification Report** – Model evaluation on test data.

---

## 🛠 Architecture & Tech Stack
| Layer                | Technology                                   |
|----------------------|----------------------------------------------|
| Speech to Text        | Python, [Faster-Whisper]                    |
| Keyword Extraction    | Python, scikit-learn, pandas                |
| Front-end UI          | HTML5, CSS3                                 |

---

## 👥 Team Contributions
| Team Member               | Work Done |
|---------------------------|-----------|
| **Ashish Kumar Chahar**    | Developed the **keyword extraction and classification model** (TF-IDF, Logistic Regression, Random Forest, balanced dataset). |
| **Gopal Varshney**         | Integrated and configured the **Faster-Whisper speech-to-text engine** with pre and post processing. |
| **Kush Gupta**             | Designed and built a **modern, responsive landing page** and the initial HTML/CSS layout to showcase the app. |
| **Anant Seth**             | Enhanced the **UI styling and user experience**, refined the HTML/CSS components into a **clean, accessible interface**. |

