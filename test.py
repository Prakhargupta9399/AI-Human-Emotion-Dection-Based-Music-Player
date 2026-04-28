import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Must be before ALL other imports

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
import anthropic   # pip install anthropic


# ── Tuneable constants ─────────────────────────────────────────────────────────
DETECT_INTERVAL   = 2
EMOTION_CONFIRM   = 2
MIN_PLAY_SECONDS  = 25
CONFIDENCE_THRESH = 55.0
SEARCH_RETRIES    = 3
SOCKET_TIMEOUT    = 20

ANTHROPIC_API_KEY = ""   # ← paste your key here

import cv2.data
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# ──────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# AI SUGGESTION ENGINE
# Calls Claude to generate contextual tips whenever a new emotion is confirmed.
# Runs in a background thread — never blocks the UI.
# ═══════════════════════════════════════════════════════════════════════════════
class AISuggestionEngine:
    """
    Generates three types of suggestion when called:
      🎯 Mood Tip      — a short emotional-wellness insight
      🎵 Artist Pick   — a recommended artist/album for the emotion + genre
      💡 Activity Idea — something to do right now that matches the mood
    """

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self._emotion_start: dict = {}   # tracks how long each emotion has persisted

    def fetch(self, emotion: str, genre: str, callback) -> None:
        """Spin up a daemon thread; calls callback(lines: list[str]) on result."""
        def _worker():
            now = time.time()
            if emotion not in self._emotion_start:
                self._emotion_start[emotion] = now
            duration_min = int((now - self._emotion_start[emotion]) / 60)

            duration_note = ""
            if duration_min >= 5:
                duration_note = (
                    f" The user has been feeling {emotion} for about "
                    f"{duration_min} minutes — factor that into your advice."
                )

            prompt = (
                f"The user's face-recognition system just confirmed their emotion "
                f"is: {emotion}. Their current music genre preference is: {genre}.{duration_note}\n\n"
                "Reply with EXACTLY 3 short lines. Each line starts with its emoji label:\n"
                "🎯 Mood Tip: <one sentence wellness insight about this emotion>\n"
                "🎵 Artist Pick: <one specific artist or album perfect for this emotion + genre>\n"
                "💡 Activity Idea: <one concrete activity that matches this mood right now>\n\n"
                "Be specific, warm, and concise. No extra text."
            )

            try:
                msg = self.client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = next(
                    (getattr(block, "text", "") for block in msg.content if hasattr(block, "text")),
                    ""
                ).strip()
                lines = [l.strip() for l in raw.splitlines() if l.strip()]
            except Exception as exc:
                lines = [f"⚠️ AI tip unavailable ({exc})"]

            callback(lines)

        threading.Thread(target=_worker, daemon=True).start()

    def reset_emotion(self, emotion: str) -> None:
        """Call when an emotion is no longer confirmed so duration resets."""
        self._emotion_start.pop(emotion, None)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════
class NeuralVibeAI:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuralVibe AI — Emotion Music Player")
        self.root.geometry("1280x820")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.running          = True
        self.current_genre    = "Bollywood"
        self.is_loading       = False
        self.last_detect_t    = 0
        self.temp_file        = None

        self.confirmed_emotion   = ""
        self.candidate_emotion   = ""
        self.candidate_count     = 0
        self.song_started_at     = 0
        self.video_imgtk       = None

        self.ai = AISuggestionEngine(ANTHROPIC_API_KEY)

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        self.setup_ui()
        self.start_system()

    # ──────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────
    def setup_ui(self):
        ctk.set_appearance_mode("dark")

        # ── Sidebar (left) ──────────────────────────────────────────────────
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

        ctk.CTkLabel(self.sidebar, text="Emotion Lock", text_color="#888").pack(pady=(14, 2))
        self.confirm_bar = ctk.CTkProgressBar(self.sidebar, width=160)
        self.confirm_bar.pack(padx=14)
        self.confirm_bar.set(0)

        ctk.CTkLabel(self.sidebar, text="Play timer", text_color="#888").pack(pady=(10, 2))
        self.timer_lbl = ctk.CTkLabel(self.sidebar, text=f"0 / {MIN_PLAY_SECONDS}s",
                                      text_color="#aaa", font=("Courier", 11))
        self.timer_lbl.pack()

        # ── Main area (centre) ──────────────────────────────────────────────
        self.main = ctk.CTkFrame(self.root, fg_color="#0d0f1e")
        self.main.pack(side="left", expand=True, fill="both")

        self.video_label = ctk.CTkLabel(self.main, text="")
        self.video_label.pack(pady=18)

        self.emo_lbl = ctk.CTkLabel(self.main, text="STARTING AI…",
                                    font=("Courier", 34, "bold"), text_color="#00ffff")
        self.emo_lbl.pack()

        self.candidate_lbl = ctk.CTkLabel(self.main, text="",
                                          font=("Courier", 12), text_color="#888")
        self.candidate_lbl.pack()

        self.song_lbl = ctk.CTkLabel(self.main, text="Initialising…",
                                     font=("Courier", 13), text_color="#ff00ff", wraplength=560)
        self.song_lbl.pack(pady=8)

        self.progress_bar = ctk.CTkProgressBar(self.main, width=520)
        self.progress_bar.pack(pady=6)
        self.progress_bar.set(0)

        # ── AI Suggestion panel (right) ─────────────────────────────────────
        self.ai_panel = ctk.CTkFrame(self.root, width=260, fg_color="#07080f",
                                     corner_radius=0)
        self.ai_panel.pack(side="right", fill="y")
        self.ai_panel.pack_propagate(False)

        ctk.CTkLabel(self.ai_panel, text="✨ AI INSIGHTS",
                     font=("Courier", 14, "bold"), text_color="#cc88ff").pack(pady=(24, 6))
        ctk.CTkLabel(self.ai_panel, text="Powered by Claude",
                     font=("Courier", 9), text_color="#444").pack()

        ctk.CTkFrame(self.ai_panel, height=1, fg_color="#222").pack(
            fill="x", padx=16, pady=10)

        self.ai_loading_lbl = ctk.CTkLabel(
            self.ai_panel, text="Waiting for\nemotion…",
            font=("Courier", 11), text_color="#555", justify="center")
        self.ai_loading_lbl.pack(pady=10)

        # Three colour-coded suggestion cards
        self._suggestion_cards = []
        for _ in range(3):
            card = ctk.CTkLabel(
                self.ai_panel, text="",
                font=("Courier", 11), text_color="#ddd",
                wraplength=230, justify="left",
                fg_color="#111420", corner_radius=8,
                padx=10, pady=10,
            )
            card.pack(padx=14, pady=6, fill="x")
            card.pack_forget()
            self._suggestion_cards.append(card)

        ctk.CTkFrame(self.ai_panel, height=1, fg_color="#222").pack(
            fill="x", padx=16, pady=10)
        self.ai_refresh_btn = ctk.CTkButton(
            self.ai_panel, text="🔄  Refresh Tips",
            command=self._refresh_ai_tips,
            fg_color="#1a0a2e", hover_color="#2a1040",
            state="disabled",
        )
        self.ai_refresh_btn.pack(padx=14, fill="x")

        # Emotion history log
        ctk.CTkLabel(self.ai_panel, text="Emotion log",
                     text_color="#444", font=("Courier", 9)).pack(pady=(18, 2))
        self.emotion_log = ctk.CTkTextbox(
            self.ai_panel, height=140, width=230,
            font=("Courier", 9), text_color="#555",
            fg_color="#0a0b14", state="disabled",
        )
        self.emotion_log.pack(padx=14, pady=(0, 14))

        self._tick_timer()

    # ──────────────────────────────────────────────
    # AI SUGGESTION HELPERS
    # ──────────────────────────────────────────────
    def _request_ai_tips(self, emotion: str) -> None:
        self.ai_loading_lbl.configure(text="🤖 Claude is thinking…", text_color="#cc88ff")
        for card in self._suggestion_cards:
            card.pack_forget()
        self.ai_refresh_btn.configure(state="disabled")

        def _on_result(lines):
            self.root.after(0, lambda: self._display_ai_tips(lines))

        self.ai.fetch(emotion, self.current_genre, _on_result)

    def _display_ai_tips(self, lines) -> None:
        self.ai_loading_lbl.configure(text="", text_color="#555")

        colours      = {"🎯": "#1a2a1a", "🎵": "#1a1a2a", "💡": "#2a1a10"}
        text_colours = {"🎯": "#88ffaa", "🎵": "#aaaaff", "💡": "#ffcc88"}

        for i, card in enumerate(self._suggestion_cards):
            if i < len(lines):
                line  = lines[i]
                emoji = line[:2] if line else ""
                card.configure(
                    text=line,
                    fg_color=colours.get(emoji, "#111420"),
                    text_color=text_colours.get(emoji, "#ddd"),
                )
                card.pack(padx=14, pady=6, fill="x")
            else:
                card.pack_forget()

        self.ai_refresh_btn.configure(state="normal")

    def _refresh_ai_tips(self) -> None:
        if self.confirmed_emotion:
            self._request_ai_tips(self.confirmed_emotion)

    def _log_emotion(self, emotion: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.emotion_log.configure(state="normal")
        self.emotion_log.insert("end", f"{ts}  {emotion}\n")
        self.emotion_log.see("end")
        self.emotion_log.configure(state="disabled")

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
        self.is_loading        = False
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
        if self.is_loading:
            return
        self.is_loading = True

        def _task():
            query = f"{emotion} mood {self.current_genre} songs"
            self.ui(song=f"🔍 Searching: {query}", status="● Searching", progress=0.1)

            ydl_opts: dict = {
                "quiet":          True,
                "no_warnings":    True,
                "extract_flat":   True,
                "socket_timeout": SOCKET_TIMEOUT,
            }

            info = None
            for attempt in range(1, SEARCH_RETRIES + 1):
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore
                        info = ydl.extract_info(f"ytsearch1:{query}", download=False)
                    break
                except Exception as exc:
                    print(f"[Search attempt {attempt}/{SEARCH_RETRIES}] {exc}")
                    if attempt < SEARCH_RETRIES:
                        self.ui(status=f"● Retrying search ({attempt})…")
                        time.sleep(3 * attempt)

            if not info:
                self.ui(song="❌ Search failed after retries — check internet",
                        status="● No network", progress=0)
                self.confirmed_emotion = ""
                self.is_loading = False
                return

            entries = info.get("entries", [])
            if not entries:
                self.ui(song="❌ No results found", status="● Idle", progress=0)
                self.confirmed_emotion = ""
                self.is_loading = False
                return

            video = entries[0]
            url   = f"https://www.youtube.com/watch?v={video['id']}"
            title = video.get("title", "Unknown Title")
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

                ydl_opts: dict = {
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
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore
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
                self.is_loading = False

        threading.Thread(target=_task, daemon=True).start()

    # ──────────────────────────────────────────────
    # EMOTION STABILITY ENGINE
    # ──────────────────────────────────────────────
    def handle_emotion(self, emotion):
        if emotion == self.candidate_emotion:
            self.candidate_count += 1
        else:
            if self.candidate_emotion:
                self.ai.reset_emotion(self.candidate_emotion)
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

            # ── Fire AI suggestions for every new confirmed emotion ──────────
            self._request_ai_tips(emotion)
            self._log_emotion(emotion)

            self.search_and_play(emotion)

    # ──────────────────────────────────────────────
    # FACE CROP HELPER
    # ──────────────────────────────────────────────
    def crop_face(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None, False

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.20 * max(w, h))
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
            if not found or face_crop is None:
                self.ui(candidate="No face detected — skipping frame",
                        status="● Waiting for face…")
                return
            try:
                result = DeepFace.analyze(
                    face_crop, actions=["emotion"],
                    enforce_detection=True, silent=True
                )
                result_dict = result[0] if isinstance(result, list) else result
                emotions   = result_dict.get("emotion", {}) if isinstance(result_dict, dict) else {}
                dominant   = result_dict.get("dominant_emotion", "") if isinstance(result_dict, dict) else ""
                confidence = emotions.get(dominant, 0) if isinstance(emotions, dict) else 0

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
                self.video_imgtk = tk_img

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


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    NeuralVibeAI(root)
    root.mainloop()