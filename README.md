⚡ NeuralVibe AI — Emotion-Based Music Player

Real-time face emotion detection that automatically plays music matching your mood.

NeuralVibe AI uses your webcam to detect facial emotions and instantly searches and plays mood-matched music from YouTube. Built with DeepFace, OpenCV, and Pygame — no manual input needed, just sit back and let your face control the vibe.

✨ Features

🎭 7 Emotion Detection — happy, sad, angry, neutral, fear, disgust, surprise
🎵 Auto Music Search — searches YouTube and streams mood-matched songs instantly
🎼 Genre Selection — Bollywood, Pop, Lo-Fi, Rock, Jazz, Hip-Hop, Classical
🔒 Emotion Stability Lock — confirms the same emotion twice before switching tracks to avoid false triggers
⏱️ Min Play Timer — waits at least 25 seconds before reacting to a new emotion
⏭️ Skip Song — manually skip to a new track anytime
🔊 Volume Control — real-time volume slider
✨ AI Mood Insights (test.py only) — Claude AI gives personalised mood tips, artist picks, and activity suggestions every time your emotion changes
📋 Emotion History Log (test.py only) — timestamped log of every detected emotion


🗂️ Project Structure
NeuralVibe-AI/
│
├── main.py          # Core app — emotion detection + music player
├── test.py          # Extended app — adds Claude AI suggestion panel
├── requirements.txt # All dependencies
└── README.md

🛠️ Tech Stack
LibraryPurposeOpenCVWebcam feed & face detection (Haar Cascade)DeepFaceFacial emotion recognitionyt-dlpYouTube search & audio streamPygameAudio playbackCustomTkinterModern dark-themed UIAnthropic Claude APIAI mood tips (test.py only)PillowImage processing for video feed

⚙️ Requirements

Python 3.9 or higher
A working webcam
ffmpeg installed and available in your system PATH
Internet connection (for YouTube search)
Anthropic API key (only for test.py)


🚀 Installation
1. Clone the repository
bashgit clone https://github.com/your-username/neuralvibe-ai.git
cd neuralvibe-ai
2. Create and activate a virtual environment (recommended)
bashpython -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
3. Install dependencies
bashpip install -r requirements.txt
4. Install ffmpeg

Windows: Download from ffmpeg.org and add to PATH
macOS: brew install ffmpeg
Linux: sudo apt install ffmpeg


▶️ Usage
Run the core player
bashpython main.py
Run with Claude AI suggestions
bash# First, open test.py and add your Anthropic API key:
# ANTHROPIC_API_KEY = "sk-ant-..."

python test.py
Once running:

Allow webcam access
Sit in front of your camera in good lighting
The app detects your emotion after 2 consecutive confirmations
Music automatically starts playing to match your mood
Use the Genre dropdown to change music style
Use Skip Song to move to the next track


🎛️ Configuration
You can tweak these constants at the top of either file:
ConstantDefaultDescriptionDETECT_INTERVAL2Seconds between each emotion checkEMOTION_CONFIRM2Consecutive detections needed to confirm an emotionMIN_PLAY_SECONDS25Minimum seconds before reacting to an emotion changeCONFIDENCE_THRESH35.0Minimum DeepFace confidence % to accept a detectionSEARCH_RETRIES3YouTube search/download retry attemptsSOCKET_TIMEOUT20Network timeout in seconds

🧠 How It Works
Webcam Frame
     │
     ▼
Face Detection (Haar Cascade)
     │
     ▼
Face Crop + Padding
     │
     ▼
DeepFace Emotion Analysis
     │
     ├── confidence < 35%? → Skip
     │
     ▼
Emotion Stability Engine
  (needs 2× same emotion)
     │
     ▼
YouTube Search (mood-mapped query)
     │
     ▼
Audio Download & Playback (Pygame)
     │
     ▼  [test.py only]
Claude AI → Mood Tip + Artist Pick + Activity Idea

🎭 Emotion → Music Mood Mapping
Detected EmotionSearch Mood UsedHappyhappy upbeat feel goodSadsad emotional heartbreakAngryangry intense energeticNeutralcalm relaxed chillFeardark tense anxiousDisgustdark moody intenseSurpriseexciting energetic upbeat

📦 requirements.txt
opencv-python
deepface
yt-dlp
pygame
customtkinter
Pillow
tf-keras
anthropic

Note: DeepFace will automatically download its model weights (~100 MB) on first run.


🐛 Known Issues & Tips

No face detected: Ensure good front-facing lighting. Avoid backlighting.
Slow music start: First song takes longer due to model warm-up. Subsequent detections are faster.
Download fails: Check internet connection. The app retries automatically up to 3 times.
Wrong emotion detected: Try CONFIDENCE_THRESH = 40.0 in poor lighting conditions.
ffmpeg not found: Make sure ffmpeg is installed and available in your system PATH.


🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

Fork the repo
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request


📄 License
Distributed under the MIT License. See LICENSE for more information.

🙏 Acknowledgements

DeepFace — facial analysis library
yt-dlp — YouTube downloader
CustomTkinter — modern Tkinter UI
Anthropic Claude — AI suggestions engine

