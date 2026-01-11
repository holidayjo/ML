import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import pandas as pd
import os
from datetime import datetime, timedelta
import pathlib

# ================= CONFIGURATION =================
DEFAULT_START_TIME = "14:21:05" 
LOG_FOLDER = "video_logger"     
# =================================================

class VideoTaggerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Passenger Speed Tagger Pro")
        self.root.geometry("1000x750")
        
        # State Variables
        self.video_path = None
        self.cap = None
        self.fps = 0
        self.total_frames = 0
        self.is_paused = True
        self.start_frame = None
        self.current_frame_idx = 0
        self.is_fullscreen = False
        
        # Initial canvas size
        self.canvas_width = 800
        self.canvas_height = 500

        if not os.path.exists(LOG_FOLDER):
            os.makedirs(LOG_FOLDER)

        # --- GUI LAYOUT ---
        
        # 1. Top Bar
        top_frame = tk.Frame(root, pady=5, padx=5, bg="#f0f0f0")
        top_frame.pack(fill=tk.X, side=tk.TOP)
        
        tk.Button(top_frame, text="📂 Load Video", command=self.load_video_dialog, bg="#ddddff").pack(side=tk.LEFT, padx=5)
        
        tk.Label(top_frame, text="Start Time (HH:MM:SS):", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        self.entry_start_time = tk.Entry(top_frame, width=10)
        self.entry_start_time.insert(0, DEFAULT_START_TIME)
        self.entry_start_time.pack(side=tk.LEFT)
        
        self.lbl_file_name = tk.Label(top_frame, text="No video loaded", fg="gray", bg="#f0f0f0")
        self.lbl_file_name.pack(side=tk.LEFT, padx=20)

        # 2. Control Panel (Bottom)
        self.control_frame = tk.Frame(root, padx=10, pady=10)
        self.control_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Slider
        self.slider_var = tk.DoubleVar()
        self.slider = ttk.Scale(self.control_frame, from_=0, to=100, 
                                orient=tk.HORIZONTAL, variable=self.slider_var, command=self.on_slider_move)
        self.slider.pack(fill=tk.X, pady=5)

        # Info Labels
        info_frame = tk.Frame(self.control_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        self.lbl_frame_info = tk.Label(info_frame, text="Frame: 0 / 0", font=("Arial", 10, "bold"))
        self.lbl_frame_info.pack(side=tk.LEFT, padx=10)
        
        self.lbl_time_info = tk.Label(info_frame, text="Real Time: --:--:--", font=("Arial", 10, "bold"), fg="blue")
        self.lbl_time_info.pack(side=tk.LEFT, padx=10)
        
        self.lbl_status = tk.Label(info_frame, text="IDLE", fg="gray", font=("Arial", 10))
        self.lbl_status.pack(side=tk.RIGHT, padx=10)

        # Buttons
        btn_frame = tk.Frame(self.control_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        self.btn_play = tk.Button(btn_frame, text="▶ Play / Pause", command=self.toggle_pause, width=15, bg="#dddddd", state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="< -1 Frame", command=lambda: self.step_frame(-1)).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="+1 Frame >", command=lambda: self.step_frame(1)).pack(side=tk.LEFT)

        # Go to Frame Input
        tk.Label(btn_frame, text="  Go to Frame:").pack(side=tk.LEFT)
        self.entry_goto = tk.Entry(btn_frame, width=8)
        self.entry_goto.pack(side=tk.LEFT)
        # CRITICAL: Bind Enter Key (<Return>) to the function
        self.entry_goto.bind("<Return>", self.goto_frame_event)
        
        tk.Button(btn_frame, text="Go", command=self.goto_frame_event).pack(side=tk.LEFT)

        tk.Frame(btn_frame, width=20).pack(side=tk.LEFT) 
        
        self.btn_start = tk.Button(btn_frame, text="[S] Mark Start", bg="#ffcccc", command=self.mark_start, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_end = tk.Button(btn_frame, text="[E] Mark End & Save", bg="#ccffcc", command=self.mark_end, state=tk.DISABLED)
        self.btn_end.pack(side=tk.LEFT, padx=5)

        # 3. Video Display Area (Middle)
        self.canvas_frame = tk.Frame(root, bg='black')
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas_frame.bind("<Configure>", self.on_resize)

        self.video_label = tk.Label(self.canvas_frame, bg='black')
        self.video_label.pack(anchor=tk.CENTER, expand=True)

        # Key Bindings
        root.bind('<space>', lambda e: self.toggle_pause())
        root.bind('<Left>', lambda e: self.step_frame(-1))
        root.bind('<Right>', lambda e: self.step_frame(1))
        root.bind('s', lambda e: self.mark_start())
        root.bind('e', lambda e: self.mark_end())
        root.bind('<F11>', lambda e: self.toggle_fullscreen())
        root.bind('f', lambda e: self.toggle_fullscreen())
        root.bind('g', lambda e: self.focus_entry()) # 'g' to focus input box

        self.update_ui_loop()

    def focus_entry(self):
        """Focus the input box when 'g' is pressed"""
        self.entry_goto.focus_set()
        self.entry_goto.select_range(0, tk.END) # Select all text for easy replacement

    def load_video_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            self.load_video(file_path)

    def load_video(self, path):
        if self.cap:
            self.cap.release()
            
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video.")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0
        self.start_frame = None
        self.is_paused = True
        
        self.slider.config(to=self.total_frames-1)
        self.lbl_file_name.config(text=os.path.basename(path), fg="black")
        self.btn_play.config(state=tk.NORMAL)
        self.btn_start.config(state=tk.NORMAL)
        self.btn_end.config(state=tk.NORMAL)
        
        video_name = pathlib.Path(path).stem
        self.log_file_path = os.path.join(LOG_FOLDER, f"{video_name}.csv")
        
        self.seek_to(0)

    def on_resize(self, event):
        self.canvas_width = event.width
        self.canvas_height = event.height
        if self.cap and self.is_paused:
             self.display_frame_current()

    def display_frame_current(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)

    def toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)

    def get_real_time_str(self, frame_idx):
        start_time_str = self.entry_start_time.get()
        try:
            video_start_dt = datetime.strptime(start_time_str, "%H:%M:%S")
            elapsed_sec = frame_idx / self.fps
            curr_dt = video_start_dt + timedelta(seconds=elapsed_sec)
            return curr_dt.strftime("%H:%M:%S.%f")[:-4]
        except ValueError:
            return "Invalid Time"

    def update_ui_loop(self):
        if self.cap and not self.is_paused and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.display_frame(frame)
            else:
                self.is_paused = True 
        
        if self.cap:
            t_str = self.get_real_time_str(self.current_frame_idx)
            self.lbl_frame_info.config(text=f"Frame: {self.current_frame_idx} / {self.total_frames}")
            self.lbl_time_info.config(text=f"Time: {t_str}")
            
            if self.root.focus_get() != self.slider:
                self.slider_var.set(self.current_frame_idx)
            
            if self.start_frame is not None:
                self.lbl_status.config(text=f"RECORDING... (Start: {self.start_frame})", fg="red")
            else:
                txt = "PAUSED" if self.is_paused else "PLAYING"
                color = "orange" if self.is_paused else "green"
                self.lbl_status.config(text=txt, fg=color)

        self.root.after(33, self.update_ui_loop)

    def display_frame(self, frame):
        h, w = frame.shape[:2]
        target_w = max(1, self.canvas_width)
        target_h = max(1, self.canvas_height)
        ratio = min(target_w / w, target_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        if new_w < 1 or new_h < 1: return

        frame = cv2.resize(frame, (new_w, new_h))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.video_label.config(image=img_tk)
        self.video_label.image = img_tk

    def toggle_pause(self):
        if not self.cap: return
        self.is_paused = not self.is_paused

    def step_frame(self, step):
        if not self.cap: return
        self.is_paused = True
        target = self.current_frame_idx + step
        self.seek_to(target)

    # Added event=None to handle key bindings
    def goto_frame_event(self, event=None):
        if not self.cap: return
        try:
            target = int(self.entry_goto.get())
            self.is_paused = True
            self.seek_to(target)
            
            # CRITICAL: Move focus back to the main window
            # so 'Space' and 'S/E' keys work immediately after jumping
            self.root.focus() 
        except ValueError:
            pass

    def on_slider_move(self, val):
        if not self.cap: return
        target = int(float(val))
        self.is_paused = True
        self.seek_to(target)

    def seek_to(self, frame_idx):
        if not self.cap: return
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.current_frame_idx = frame_idx
        
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def mark_start(self):
        if not self.cap: return
        self.start_frame = self.current_frame_idx
        print(f"Start marked: {self.start_frame}")

    def mark_end(self):
        if not self.cap: return
        if self.start_frame is None:
            messagebox.showwarning("Warning", "Start point not marked! Press 'S' first.")
            return

        end_frame = self.current_frame_idx
        duration = (end_frame - self.start_frame) / self.fps
        
        if duration < 0:
            messagebox.showerror("Error", "End frame cannot be before Start frame.")
            return

        t_start = self.get_real_time_str(self.start_frame)
        t_end = self.get_real_time_str(end_frame)
        
        data = {
            'video_file': [os.path.basename(self.video_path)],
            'start_frame': [self.start_frame],
            'end_frame': [end_frame],
            'start_time': [t_start],
            'end_time': [t_end],
            'duration_sec': [round(duration, 3)]
        }
        df = pd.DataFrame(data)
        
        mode = 'a' if os.path.exists(self.log_file_path) else 'w'
        header = not os.path.exists(self.log_file_path)
        
        try:
            df.to_csv(self.log_file_path, mode=mode, header=header, index=False)
            print(f"Saved: {t_start} -> {t_end} ({duration:.2f}s)")
            self.start_frame = None
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoTaggerApp(root)
    root.mainloop()