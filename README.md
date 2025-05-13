# Sign Language to Text Recognition

A live sign language to text recognition system using machine learning and Python. This project recognizes American Sign Language (ASL) letters A–Z (and optionally numbers 1–9) from webcam input, converting them to text in real-time.

## Project Overview
- **Objective**: Recognize ASL gestures from live video and convert them to text.
- **Technologies**:
  - **MediaPipe**: For hand landmark detection.
  - **Random Forest**: For gesture classification (can be upgraded to MLP or LSTM).
  - **OpenCV**: For webcam video capture.
  - **Python**: Core programming language.
- **Components**:
  - Data collection using webcam.
  - Model training with extracted hand landmarks.
  - Live recognition with real-time text output.


## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/sign-language-to-text.git
   cd sign-language-to-text
   
