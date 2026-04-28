import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FIX 1: Must be before ALL other imports

import cv2
import threading
import time
import tempfile
import warnings
warnings.filterwarnings("ignore")

import customtkinter as ctk
from PIL import Image, ImageTk
from deepface import DeepFace
import yt_dlp
import pygame


# ── Tuneable constants ─────────────────────────────────────────────────────────
DETECT_INTERVAL   = 2      # checks every 2 seconds (real-time feel)
EMOTION_CONFIRM   = 2      # only needs 2 consecutive detections
MIN_PLAY_SECONDS  = 25     # reacts faster to emotion changes
CONFIDENCE_THRESH = 55.0   # ignore detections below this % confidence
SEARCH_RETRIES    = 3
SOCKET_TIMEOUT    = 20

# OpenCV face detector (Haar Cascade — ships with OpenCV, no download needed)
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# ──────────────────────────────────────────────────────────────────────────────


class NeuralVibeAI:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuralVibe AI — Emotion Music Player")
        self.root.geometry("1100x820")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # ── App state ──
        self.running          = True
        self.current_genre    = "Bollywood"
        self.is_loading       = False
        self.last_detect_t    = 0
        self.temp_file        = None

        # ── Emotion stability ──
        self.confirmed_emotion   = ""
        self.candidate_emotion   = ""
        self.candidate_count     = 0
        self.song_started_at     = 0

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        self.setup_ui()
        self.start_system()

    # ──────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────
    def setup_ui(self):
        ctk.set_appearance_mode("dark")

        # ── Sidebar ──
        self.sidebar = ctk.CTkFrame(self.root, width=220, fg_color="#090a12")
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(self.sidebar, text="⚡ NEURAL VIBE",
                     font=("Courier", 18, "bold"), text_color="#00ffff").pack(pady=30)

        ctk.CTkLabel(self.sidebar, text="Genre", text_color="#888").pack()
        self.genre_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["Bollywood", "Pop", "Lo-Fi", "Rock", "Jazz", "Hip-Hop", "Classical"],
            command=self.set_genre)
        self.genre_menu.pack(pady=8, padx=14, fill="x")

        ctk.CTkLabel(self.sidebar, text="Volume", text_color="#888").pack(pady=(18, 4))
        self.vol_slider = ctk.CTkSlider(self.sidebar, from_=0, to=1, command=self.set_volume)
        self.vol_slider.set(0.7)
        self.vol_slider.pack(padx=20, fill="x")

        ctk.CTkLabel(self.sidebar, text=f"Confirm after {EMOTION_CONFIRM}× detections",
                     text_color="#555", font=("Courier", 10)).pack(pady=(6, 0))
        ctk.CTkLabel(self.sidebar, text=f"Min play time: {MIN_PLAY_SECONDS}s",
                     text_color="#555", font=("Courier", 10)).pack()

        ctk.CTkButton(self.sidebar, text="⏭  Skip Song", command=self.skip_song,
                      fg_color="#1a1a2e", hover_color="#16213e").pack(pady=20, padx=14, fill="x")

        ctk.CTkLabel(self.sidebar, text="Status", text_color="#888").pack(pady=(10, 4))
        self.status_lbl = ctk.CTkLabel(self.sidebar, text="● Starting…",
                                       text_color="#ffff00", font=("Courier", 12))
        self.status_lbl.pack()

        # ── Stability indicator ──
        ctk.CTkLabel(self.sidebar, text="Emotion Lock", text_color="#888").pack(pady=(14, 2))
        self.confirm_bar = ctk.CTkProgressBar(self.sidebar, width=160)
        self.confirm_bar.pack(padx=14)
        self.confirm_bar.set(0)

        ctk.CTkLabel(self.sidebar, text="Play timer", text_color="#888").pack(pady=(10, 2))
        self.timer_lbl = ctk.CTkLabel(self.sidebar, text=f"0 / {MIN_PLAY_SECONDS}s",
                                      text_color="#aaa", font=("Courier", 11))
        self.timer_lbl.pack()

        # ── Main area ──
        self.main = ctk.CTkFrame(self.root, fg_color="#0d0f1e")
        self.main.pack(side="right", expand=True, fill="both")

        self.video_label = ctk.CTkLabel(self.main, text="")
        self.video_label.pack(pady=18)

        self.emo_lbl = ctk.CTkLabel(self.main, text="STARTING AI…",
                                    font=("Courier", 34, "bold"), text_color="#00ffff")
        self.emo_lbl.pack()

        self.candidate_lbl = ctk.CTkLabel(self.main, text="",
                                          font=("Courier", 12), text_color="#888")
        self.candidate_lbl.pack()

        self.song_lbl = ctk.CTkLabel(self.main, text="Initialising…",
                                     font=("Courier", 13), text_color="#ff00ff", wraplength=620)
        self.song_lbl.pack(pady=8)

        self.progress_bar = ctk.CTkProgressBar(self.main, width=520)
        self.progress_bar.pack(pady=6)
        self.progress_bar.set(0)

        self._tick_timer()

    # ──────────────────────────────────────────────
    # PLAY TIMER TICK
    # ──────────────────────────────────────────────
    def _tick_timer(self):
        if self.song_started_at > 0:
            elapsed = int(time.time() - self.song_started_at)
            self.timer_lbl.configure(text=f"{elapsed} / {MIN_PLAY_SECONDS}s")
        else:
            self.timer_lbl.configure(text=f"0 / {MIN_PLAY_SECONDS}s")
        if self.running:
            self.root.after(1000, self._tick_timer)

    # ──────────────────────────────────────────────
    # CONTROLS
    # ──────────────────────────────────────────────
    def set_genre(self, g):
        self.current_genre = g
        self.confirmed_emotion = ""

    def set_volume(self, v):
        pygame.mixer.music.set_volume(float(v))

    def skip_song(self):
        pygame.mixer.music.stop()
        self.confirmed_emotion = ""
        self.candidate_emotion = ""
        self.candidate_count   = 0
        self.song_started_at   = 0
        self.is_loading        = False  # FIX 4: reset lock on manual skip
        self.ui(status="● Skipped — waiting")
        self.confirm_bar.set(0)

    # ──────────────────────────────────────────────
    # THREAD-SAFE UI
    # ──────────────────────────────────────────────
    def ui(self, **kw):
        def _apply():
            if "emotion"    in kw: self.emo_lbl.configure(text=kw["emotion"])
            if "candidate"  in kw: self.candidate_lbl.configure(text=kw["candidate"])
            if "song"       in kw: self.song_lbl.configure(text=kw["song"])
            if "status"     in kw: self.status_lbl.configure(text=kw["status"])
            if "progress"   in kw: self.progress_bar.set(kw["progress"])
            if "lock"       in kw: self.confirm_bar.set(kw["lock"])
        self.root.after(0, _apply)

    # ──────────────────────────────────────────────
    # YOUTUBE SEARCH
    # ──────────────────────────────────────────────
    def search_and_play(self, emotion):
        # FIX 3: Lock immediately here, before spawning thread,
        # so no second emotion can trigger a parallel search.
        if self.is_loading:
            return
        self.is_loading = True

        def _task():
            query = f"{emotion} mood {self.current_genre} songs"
            self.ui(song=f"🔍 Searching: {query}", status="● Searching", progress=0.1)

            ydl_opts = {
                "quiet":          True,
                "no_warnings":    True,
                "extract_flat":   True,
                "socket_timeout": SOCKET_TIMEOUT,
            }

            info = None
            for attempt in range(1, SEARCH_RETRIES + 1):
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(f"ytsearch1:{query}", download=False)
                    break
                except Exception as exc:
                    print(f"[Search attempt {attempt}/{SEARCH_RETRIES}] {exc}")
                    if attempt < SEARCH_RETRIES:
                        self.ui(status=f"● Retrying search ({attempt})…")
                        time.sleep(3 * attempt)

            # FIX 4: Always release the lock on early-exit failure paths
            if not info:
                self.ui(song="❌ Search failed after retries — check internet",
                        status="● No network", progress=0)
                self.confirmed_emotion = ""
                self.is_loading = False  # FIX 4
                return

            entries = info.get("entries", [])
            if not entries:
                self.ui(song="❌ No results found", status="● Idle", progress=0)
                self.confirmed_emotion = ""
                self.is_loading = False  # FIX 4
                return

            video = entries[0]
            url   = f"https://www.youtube.com/watch?v={video['id']}"
            title = video.get("title", "Unknown Title")
            # Note: is_loading stays True — download_and_play will release it
            self.download_and_play(url, title)

        threading.Thread(target=_task, daemon=True).start()

    # ──────────────────────────────────────────────
    # DOWNLOAD → PLAY
    # ──────────────────────────────────────────────
    def download_and_play(self, url, title):
        def _task():
            try:
                self.ui(song=f"⏳ Loading: {title[:45]}…",
                        status="● Downloading", progress=0.2)
                pygame.mixer.music.stop()

                if self.temp_file and os.path.exists(self.temp_file):
                    try: os.remove(self.temp_file)
                    except OSError: pass

                fd, base_path = tempfile.mkstemp()
                os.close(fd)
                os.remove(base_path)

                ydl_opts = {
                    "format": "bestaudio/best",
                    "outtmpl": base_path + ".%(ext)s",
                    "postprocessors": [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "128",
                    }],
                    "quiet":            True,
                    "no_warnings":      True,
                    "socket_timeout":   SOCKET_TIMEOUT,
                    "retries":          5,
                    "fragment_retries": 5,
                }

                self.ui(progress=0.5)
                for attempt in range(1, SEARCH_RETRIES + 1):
                    try:
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([url])
                        break
                    except Exception as exc:
                        print(f"[Download attempt {attempt}/{SEARCH_RETRIES}] {exc}")
                        if attempt < SEARCH_RETRIES:
                            self.ui(status=f"● Retrying download ({attempt})…")
                            time.sleep(3 * attempt)
                        else:
                            raise

                mp3_path = base_path + ".mp3"
                if not os.path.exists(mp3_path):
                    raise FileNotFoundError(f"Audio file missing: {mp3_path}")

                self.temp_file = mp3_path
                self.ui(progress=0.85)

                pygame.mixer.music.load(self.temp_file)
                pygame.mixer.music.set_volume(self.vol_slider.get())
                pygame.mixer.music.play()

                self.song_started_at = time.time()
                self.ui(song=f"🎵 {title[:55]}",
                        status="● Playing", progress=1.0)

            except Exception as exc:
                print(f"[Playback Error] {exc}")
                self.ui(song="❌ Download failed — will retry next detection",
                        status="● Error", progress=0)
                self.confirmed_emotion = ""

            finally:
                self.is_loading = False  # Always released here

        threading.Thread(target=_task, daemon=True).start()

    # ──────────────────────────────────────────────
    # EMOTION STABILITY ENGINE
    # ──────────────────────────────────────────────
    def handle_emotion(self, emotion):
        if emotion == self.candidate_emotion:
            self.candidate_count += 1
        else:
            self.candidate_emotion = emotion
            self.candidate_count   = 1

        lock_ratio = min(self.candidate_count / EMOTION_CONFIRM, 1.0)
        time_since = time.time() - self.song_started_at if self.song_started_at > 0 else 0
        play_ok    = time_since >= MIN_PLAY_SECONDS or self.song_started_at == 0

        self.ui(
            candidate=f"Candidate: {self.candidate_emotion} "
                      f"({self.candidate_count}/{EMOTION_CONFIRM})"
                      + ("  ⏱ waiting…" if not play_ok else ""),
            lock=lock_ratio,
        )

        confirmed = self.candidate_count >= EMOTION_CONFIRM
        different = emotion != self.confirmed_emotion

        if confirmed and different and play_ok and not self.is_loading:
            self.confirmed_emotion = emotion
            self.candidate_count   = 0
            self.ui(emotion=emotion.upper(), status="● New emotion detected")
            self.search_and_play(emotion)

    # ──────────────────────────────────────────────
    # FACE CROP HELPER
    # ──────────────────────────────────────────────
    def crop_face(self, frame):
        """
        Returns (face_crop, True) if a face is found,
        or (None, False) if no face is detected.
        The face region is expanded slightly for better DeepFace accuracy.
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            return None, False

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        pad   = int(0.20 * max(w, h))
        fh, fw = frame.shape[:2]
        x1 = max(0,  x - pad)
        y1 = max(0,  y - pad)
        x2 = min(fw, x + w + pad)
        y2 = min(fh, y + h + pad)

        return frame[y1:y2, x1:x2], True

    # ──────────────────────────────────────────────
    # DEEPFACE ANALYSIS
    # ──────────────────────────────────────────────
    def analyse_frame(self, frame):
        def _task():
            face_crop, found = self.crop_face(frame)
            if not found:
                self.ui(candidate="No face detected — skipping frame",
                        status="● Waiting for face…")
                return

            try:
                result = DeepFace.analyze(
                    face_crop,
                    actions=["emotion"],
                    enforce_detection=True,
                    silent=True
                )

                emotions   = result[0]["emotion"]
                dominant   = result[0]["dominant_emotion"]
                confidence = emotions[dominant]

                if confidence < CONFIDENCE_THRESH:
                    self.ui(
                        candidate=f"Low confidence ({confidence:.1f}%) "
                                  f"for '{dominant}' — skipping",
                        status="● Uncertain detection"
                    )
                    return

                emotion = dominant.capitalize()
                self.root.after(0, lambda e=emotion: self.handle_emotion(e))

            except ValueError:
                self.ui(candidate="Face crop too small or unclear — skipping",
                        status="● Waiting for face…")
            except Exception as exc:
                print(f"[Detection Error] {exc}")

        threading.Thread(target=_task, daemon=True).start()

    # ──────────────────────────────────────────────
    # CAMERA LOOP
    # ──────────────────────────────────────────────
    def vision_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.ui(emotion="NO CAMERA FOUND", status="● Camera Error")
            return

        self.ui(status="● Camera Active")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces    = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            rgb_draw = rgb.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(rgb_draw, (x, y), (x + w, y + h), (0, 255, 200), 2)

            img   = Image.fromarray(rgb_draw).resize((460, 340))
            imgtk = ImageTk.PhotoImage(image=img)

            def _show(tk_img=imgtk):
                self.video_label.configure(image=tk_img)
                self.video_label.imgtk = tk_img

            self.root.after(0, _show)

            now = time.time()
            if (now - self.last_detect_t) >= DETECT_INTERVAL and not self.is_loading:
                self.last_detect_t = now
                self.analyse_frame(frame.copy())

            time.sleep(0.033)

        cap.release()

    # ──────────────────────────────────────────────
    # START / STOP
    # ──────────────────────────────────────────────
    def start_system(self):
        threading.Thread(target=self.vision_loop, daemon=True).start()

    def on_close(self):
        self.running = False
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        if self.temp_file and os.path.exists(self.temp_file):
            try: os.remove(self.temp_file)
            except OSError: pass
        self.root.destroy()


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    NeuralVibeAI(root)
    root.mainloop()