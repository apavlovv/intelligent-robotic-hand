import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import serial
import serial.tools.list_ports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# ШЛЯХ ДО МОДЕЛІ MediaPipe
# ==========================================
MP_MODEL_PATH = r"C:\Models\hand_landmarker.task"

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def get_joint_angle(p1, p2, p3, clamp=True):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 180.0
    dot = np.dot(v1, v2)
    cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    if clamp:
        return np.clip(angle, 90.0, 180.0)
    return angle

class HandTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bionic Hand Control Center")
        self.root.geometry("1300x1000")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.current_result = None
        self.timestamp = 0
        self.previous_pts_3d = None
        self.previous_is_flipped = False
        self.SMOOTHING_FACTOR = 0.5 
        
        self.current_hand = None        
        self.target_hand = None
        self.thumb_transition = False   
        self.thumb_transition_start = 0 

        self.smoothed_angles = [90.0] * 16 

        self.ser = None
        self.is_connected = False
        self.last_log_time = time.time()
        self.last_serial_send_time = time.time()

        base_options = python.BaseOptions(model_asset_path=MP_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self.process_result
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        self.cap = cv2.VideoCapture(0)

        self.setup_ui()
        self.refresh_ports()
        
        self.update_frame()

    def process_result(self, result, output_image, timestamp):
        self.current_result = result

    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        uart_frame = tk.Frame(main_frame, bd=2, relief=tk.GROOVE)
        uart_frame.pack(fill=tk.X, pady=5, ipadx=5, ipady=5)
        
        tk.Label(uart_frame, text="UART Підключення:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)
        self.port_var = tk.StringVar()
        self.port_dropdown = ttk.Combobox(uart_frame, textvariable=self.port_var, state="readonly", width=15)
        self.port_dropdown.pack(side=tk.LEFT, padx=5)
        ttk.Button(uart_frame, text="Оновити", command=self.refresh_ports).pack(side=tk.LEFT, padx=5)
        
        self.btn_connect = tk.Button(uart_frame, text="Підключитися", bg="lightgreen", font=("Arial", 10, "bold"), command=self.toggle_connection)
        self.btn_connect.pack(side=tk.LEFT, padx=15)

        tk.Label(uart_frame, text="Затримка логів (мс):").pack(side=tk.LEFT, padx=5)
        self.delay_slider = tk.Scale(uart_frame, from_=100, to=2000, orient=tk.HORIZONTAL, length=150)
        self.delay_slider.set(500)
        self.delay_slider.pack(side=tk.LEFT, padx=5)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        self.tab_calib = tk.Frame(self.notebook)
        self.tab_group = tk.Frame(self.notebook)
        self.tab_track = tk.Frame(self.notebook)

        self.notebook.add(self.tab_calib, text="1. Калібрування та Таблиця")
        self.notebook.add(self.tab_group, text="2. Груповий тест (Кути)")
        self.notebook.add(self.tab_track, text="3. Трекінг (MediaPipe)")

        self.setup_tab_calib()
        self.setup_tab_group()
        self.setup_tab_track()

        bottom_frame = tk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=5)
        tk.Label(bottom_frame, text="Системна Консоль", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        scroll = tk.Scrollbar(bottom_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.console_text = tk.Text(bottom_frame, height=8, bg="black", fg="lime", font=("Consolas", 10), yscrollcommand=scroll.set)
        self.console_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scroll.config(command=self.console_text.yview)

        self.log("SYSTEM: Інтерфейс завантажено.")

    def setup_tab_calib(self):
        top_split = tk.PanedWindow(self.tab_calib, orient=tk.HORIZONTAL)
        top_split.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ctrl_frame = tk.Frame(top_split, relief=tk.RIDGE, bd=2)
        top_split.add(ctrl_frame, width=400)

        tk.Label(ctrl_frame, text="Управління каналом", font=("Arial", 12, "bold")).pack(pady=10)
        
        row1 = tk.Frame(ctrl_frame)
        row1.pack(pady=5)
        tk.Label(row1, text="Канал Серво (0-15):").pack(side=tk.LEFT, padx=5)
        self.calib_channel = ttk.Combobox(row1, values=list(range(16)), state="readonly", width=5)
        self.calib_channel.current(0)
        self.calib_channel.pack(side=tk.LEFT, padx=5)
        self.calib_channel.bind("<<ComboboxSelected>>", self.on_calib_channel_change)

        tk.Label(ctrl_frame, text="Тест ШІМ (рухати для перевірки):").pack(pady=(15, 0))
        self.calib_pwm_slider = tk.Scale(ctrl_frame, from_=100, to=500, orient=tk.HORIZONTAL, length=300, command=self.on_calib_slider)
        self.calib_pwm_slider.set(300)
        self.calib_pwm_slider.pack(pady=5)

        grid_frame = tk.Frame(ctrl_frame)
        grid_frame.pack(pady=15)
        tk.Label(grid_frame, text="Min (0°)").grid(row=0, column=0)
        tk.Label(grid_frame, text="Mid (90°)").grid(row=0, column=1)
        tk.Label(grid_frame, text="Max (180°)").grid(row=0, column=2)
        
        self.entry_min = tk.Entry(grid_frame, width=8, justify="center")
        self.entry_min.grid(row=1, column=0, padx=5)
        self.entry_mid = tk.Entry(grid_frame, width=8, justify="center")
        self.entry_mid.grid(row=1, column=1, padx=5)
        self.entry_max = tk.Entry(grid_frame, width=8, justify="center")
        self.entry_max.grid(row=1, column=2, padx=5)

        tk.Button(ctrl_frame, text="Відправити Ліміти в RAM", bg="lightblue", command=self.send_calibration).pack(pady=10, fill=tk.X, padx=20)
        tk.Button(ctrl_frame, text="Зберегти в NVS (Flash)", bg="orange", command=self.send_save).pack(pady=5, fill=tk.X, padx=20)

        table_frame = tk.Frame(top_split, relief=tk.RIDGE, bd=2)
        top_split.add(table_frame, width=500)

        header_frame = tk.Frame(table_frame)
        header_frame.pack(fill=tk.X, pady=5)
        tk.Label(header_frame, text="Поточна конфігурація в ESP32", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(header_frame, text="🔄 Оновити дані з ESP", bg="lightyellow", command=self.request_config).pack(side=tk.RIGHT, padx=10)

        columns = ("Ch", "Min ШІМ", "Mid ШІМ", "Max ШІМ")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=16)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.CENTER)
            
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        for i in range(16):
            self.tree.insert("", "end", iid=str(i), values=(f"Кан {i}", "-", "-", "-"))
            
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

    def on_tree_select(self, event):
        selected = self.tree.selection()
        if selected:
            item_id = selected[0]
            values = self.tree.item(item_id, "values")
            self.calib_channel.set(item_id)
            self.on_calib_slider(self.calib_pwm_slider.get())
            if values[1] != "-":
                self.entry_min.delete(0, tk.END)
                self.entry_min.insert(0, values[1])
                self.entry_mid.delete(0, tk.END)
                self.entry_mid.insert(0, values[2])
                self.entry_max.delete(0, tk.END)
                self.entry_max.insert(0, values[3])

    def setup_tab_group(self):
        tk.Label(self.tab_group, text="Синхронне керування (По кутах 0-180°)", font=("Arial", 14, "bold")).pack(pady=15)
        self.group_vars = []
        chk_frame = tk.Frame(self.tab_group)
        chk_frame.pack(pady=10)
        
        for i in range(16):
            var = tk.IntVar()
            self.group_vars.append(var)
            chk = tk.Checkbutton(chk_frame, text=f"Ch {i}", variable=var, command=self.update_group_servos)
            chk.grid(row=i//4, column=i%4, padx=15, pady=10, sticky="w")
            
        tk.Label(self.tab_group, text="Задати кут (0-180°) вибраним каналам:").pack(pady=10)
        self.group_pwm_slider = tk.Scale(self.tab_group, from_=0, to=180, orient=tk.HORIZONTAL, length=400, command=self.update_group_servos)
        self.group_pwm_slider.set(90)
        self.group_pwm_slider.pack(pady=10)

    def setup_tab_track(self):
        top_frame = tk.Frame(self.tab_track)
        top_frame.pack(fill=tk.X, expand=False, pady=5)
        
        self.lbl_video = tk.Label(top_frame, bg="black", width=400, height=400)
        self.lbl_video.pack(side=tk.LEFT, padx=30)
        
        self.lbl_canvas = tk.Label(top_frame, bg="white", width=400, height=400)
        self.lbl_canvas.pack(side=tk.RIGHT, padx=30)
        
        ctrl_frame = tk.Frame(self.tab_track, bd=2, relief=tk.RIDGE)
        ctrl_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=5)

        params_frame = tk.Frame(ctrl_frame)
        params_frame.pack(pady=10, fill=tk.X)

        mult_frame = tk.Frame(params_frame)
        mult_frame.pack(side=tk.LEFT, expand=True)
        tk.Label(mult_frame, text="Коефіцієнт великого пальця:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.thumb_mult_var = tk.DoubleVar(value=3.0)
        self.thumb_mult_slider = tk.Scale(mult_frame, variable=self.thumb_mult_var, from_=1.0, to=6.0, resolution=0.1, orient=tk.HORIZONTAL, length=150)
        self.thumb_mult_slider.pack(side=tk.LEFT)

        smooth_frame = tk.Frame(params_frame)
        smooth_frame.pack(side=tk.RIGHT, expand=True)
        tk.Label(smooth_frame, text="Плавність рухів (Швидкість):", font=("Arial", 10, "bold"), fg="blue").pack(side=tk.LEFT, padx=5)
        self.angle_smooth_var = tk.DoubleVar(value=0.15)
        self.angle_smooth_slider = tk.Scale(smooth_frame, variable=self.angle_smooth_var, from_=0.05, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=150)
        self.angle_smooth_slider.pack(side=tk.LEFT)

        tk.Label(ctrl_frame, text="Активні фаланги (зняти галочку = розслабити):", font=("Arial", 10, "bold")).pack(pady=5)
        
        self.track_vars = []
        chk_frame = tk.Frame(ctrl_frame)
        chk_frame.pack(pady=5)
        
        for i in range(16):
            var = tk.IntVar(value=1) 
            self.track_vars.append(var)
            chk = tk.Checkbutton(chk_frame, text=f"Ch {i}", variable=var)
            chk.grid(row=i//8, column=i%8, padx=10, pady=5, sticky="w")

        warn_lbl = tk.Label(self.tab_track, text="⚠️ Команди відправляються тільки коли ця вкладка активна!", fg="red", font=("Arial", 11, "bold"))
        warn_lbl.pack(pady=5)

    def send_command(self, cmd_str):
        if self.is_connected and self.ser and self.ser.is_open:
            try:
                self.ser.write((cmd_str + "\n").encode('utf-8'))
            except Exception as e:
                self.log(f"SYSTEM: Помилка відправки - {e}")
                self.toggle_connection()

    def read_serial_data(self):
        if self.is_connected and self.ser and self.ser.is_open:
            try:
                while self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self.process_incoming_data(line)
            except Exception as e:
                self.log(f"SYSTEM ERROR (UART read): {e}")

    def process_incoming_data(self, line):
        try:
            if line.startswith("CFG,"):
                parts = line.split(",")
                if len(parts) == 5:
                    _, ch, min_v, mid_v, max_v = parts
                    ch = str(ch).strip()
                    self.tree.item(ch, values=(f"Кан {ch}", min_v, mid_v, max_v))
            elif line.startswith("OK:") or line.startswith("INFO:"):
                self.log(f"ESP32: {line}")
        except Exception as e:
            self.log(f"SYSTEM ERROR (Parse): Помилка оновлення таблиці - {e}")

    def request_config(self):
        self.send_command("R")
        self.log("SENT: Запит таблиці конфігурації...")

    def on_calib_channel_change(self, event):
        self.on_calib_slider(self.calib_pwm_slider.get())

    def on_calib_slider(self, val):
        if self.notebook.index("current") == 0:
            ch = int(self.calib_channel.get())
            mask = 1 << ch 
            self.send_command(f"W,{mask},{val}")

    def update_group_servos(self, *_):
        if self.notebook.index("current") == 1:
            val = self.group_pwm_slider.get() 
            mask = 0
            for i in range(16):
                if self.group_vars[i].get() == 1:
                    mask |= (1 << i)
            self.send_command(f"G,{mask},{val}")

    def send_calibration(self):
        ch = self.calib_channel.get()
        min_v = self.entry_min.get().strip()
        mid_v = self.entry_mid.get().strip()
        max_v = self.entry_max.get().strip()
        
        if min_v.isdigit() and mid_v.isdigit() and max_v.isdigit():
            self.send_command(f"C,{ch},{min_v},{mid_v},{max_v}")
            self.log(f"SENT: Ліміти каналу {ch} оновлено (RAM).")
            self.root.after(200, self.request_config)
        else:
            self.log("ERROR: Введіть коректні числа (без букв і пробілів) для лімітів.")

    def send_save(self):
        self.send_command("S")

    def refresh_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        if not ports: ports = ["Немає портів"]
        self.port_dropdown['values'] = ports
        self.port_dropdown.current(0)

    def toggle_connection(self):
        if not self.is_connected:
            port = self.port_var.get()
            if port == "Немає портів" or port == "": return
            try:
                self.ser = serial.Serial(port, 115200, timeout=0.05) 
                self.is_connected = True
                self.btn_connect.config(text="Відключитися", bg="salmon")
                self.log(f"SYSTEM: Підключено до {port}")
                self.root.after(1500, self.request_config)
            except Exception as e:
                self.log(f"SYSTEM: Помилка підключення: {e}")
        else:
            if self.ser: self.ser.close()
            self.is_connected = False
            self.btn_connect.config(text="Підключитися", bg="lightgreen")
            self.log("SYSTEM: Відключено.")

    def log(self, message):
        self.console_text.insert(tk.END, message + "\n")
        self.console_text.see(tk.END)

    def calculate_esp32s3_angle(self, mp_angles, is_right_hand, channel, thumb_mult):
        if channel == 6: 
            return 0 if is_right_hand else 180
            
        if channel in [7, 8, 9]:
            base_angle = mp_angles[9] 
            
            OPEN_THRESHOLD = 165.0
            bend = OPEN_THRESHOLD - base_angle
            
            if bend < 0: 
                bend = 0.0
                
            scaled_bend = np.clip(bend * thumb_mult, 0.0, 90.0)
            
            if channel == 7:
                return int(90.0 - scaled_bend)
            else:
                if is_right_hand: 
                    return int(90.0 - scaled_bend)
                else: 
                    return int(90.0 + scaled_bend)

        mp_angle = mp_angles[channel]
        
        bend = 175.0 - mp_angle
        if bend < 0:
            bend = 0.0
            
        bend = np.clip(bend, 0.0, 90.0) 
        
        if is_right_hand: 
            return int(90.0 - bend)
        else: 
            return int(90.0 + bend)

    def update_frame(self):
        self.read_serial_data()

        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            canvas = np.ones((400, 400, 3), dtype=np.uint8) * 255

            self.landmarker.detect_async(mp_image, self.timestamp)
            self.timestamp += 1
            res = self.current_result

            delay_ms = self.delay_slider.get()
            current_time = time.time()
            is_tracking_active = (self.notebook.index("current") == 2)

            if res and res.hand_landmarks:
                hand_landmarks = res.hand_landmarks[0]
                handedness_label = res.handedness[0][0].category_name if res.handedness else "Unknown"
                handedness_label = "Right" if handedness_label == "Left" else ("Left" if handedness_label == "Right" else "Unknown")
                is_right = (handedness_label == "Right")

                if handedness_label != "Unknown":
                    if self.current_hand is None: self.current_hand = handedness_label
                    if handedness_label != self.current_hand and not self.thumb_transition:
                        self.thumb_transition = True
                        self.thumb_transition_start = current_time
                        self.target_hand = handedness_label
                        self.log(f"Зміна руки: {self.current_hand} -> {self.target_hand}")

                pts_2d_raw = np.array([(lm.x, lm.y) for lm in hand_landmarks])
                h, w, _ = frame.shape
                for connection in CONNECTIONS:
                    s, e = pts_2d_raw[connection[0]], pts_2d_raw[connection[1]]
                    cv2.line(frame, (int(s[0]*w), int(s[1]*h)), (int(e[0]*w), int(e[1]*h)), (0, 255, 0), 2)
                
                max_y = max([int(pt[1]*h) for pt in pts_2d_raw])
                center_x = int(np.mean([int(pt[0]*w) for pt in pts_2d_raw]))
                cv2.putText(frame, f"{handedness_label} Hand", (max(10, min(center_x-50, w-150)), min(max_y+30, h-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if is_right else (255, 0, 0), 2)

                pts_3d = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks])
                is_flipped = pts_3d[2, 0] > pts_3d[17, 0]
                if is_flipped:
                    center_x_3d = (np.min(pts_3d[:,0]) + np.max(pts_3d[:,0])) / 2.0
                    pts_3d[:,0] = 2 * center_x_3d - pts_3d[:,0]

                if self.previous_pts_3d is None or is_flipped != self.previous_is_flipped:
                    self.previous_pts_3d = pts_3d.copy()
                else:
                    pts_3d = (pts_3d * self.SMOOTHING_FACTOR) + (self.previous_pts_3d * (1.0 - self.SMOOTHING_FACTOR))
                    self.previous_pts_3d = pts_3d.copy()
                self.previous_is_flipped = is_flipped

                canvas_pts = pts_3d[:, :2]
                min_c, max_c = np.min(canvas_pts, axis=0), np.max(canvas_pts, axis=0)
                
                center = (min_c + max_c) / 2
                size = np.max(max_c - min_c) or 0.01
                scale = (400 * 0.7) / size
                offset = 200 - (center * scale)
                
                for conn in CONNECTIONS:
                    s, e = canvas_pts[conn[0]], canvas_pts[conn[1]]
                    cv2.line(canvas, (int(s[0]*scale+offset[0]), int(s[1]*scale+offset[1])), 
                             (int(e[0]*scale+offset[0]), int(e[1]*scale+offset[1])), (0, 255, 0), 3)

                if is_tracking_active and (current_time - self.last_serial_send_time >= 0.02):
                    mp_angles = [
                        get_joint_angle(pts_3d[20], pts_3d[19], pts_3d[18]), get_joint_angle(pts_3d[19], pts_3d[18], pts_3d[17]), get_joint_angle(pts_3d[18], pts_3d[17], pts_3d[0]),
                        get_joint_angle(pts_3d[16], pts_3d[15], pts_3d[14]), get_joint_angle(pts_3d[15], pts_3d[14], pts_3d[13]), get_joint_angle(pts_3d[14], pts_3d[13], pts_3d[0]),
                        0,
                        get_joint_angle(pts_3d[2], pts_3d[1], pts_3d[0]), get_joint_angle(pts_3d[3], pts_3d[2], pts_3d[1]), get_joint_angle(pts_3d[4], pts_3d[3], pts_3d[2]),
                        get_joint_angle(pts_3d[10], pts_3d[9], pts_3d[0]), get_joint_angle(pts_3d[12], pts_3d[11], pts_3d[10]), get_joint_angle(pts_3d[11], pts_3d[10], pts_3d[9]),
                        get_joint_angle(pts_3d[8], pts_3d[7], pts_3d[6]), get_joint_angle(pts_3d[7], pts_3d[6], pts_3d[5]), get_joint_angle(pts_3d[6], pts_3d[5], pts_3d[0])
                    ]
                    
                    track_mask = 0
                    for i in range(16):
                        if self.track_vars[i].get() == 1:
                            track_mask |= (1 << i)
                    cmds = f"X,{track_mask}\n" 
                    
                    current_thumb_mult = self.thumb_mult_var.get()
                    smooth_alpha = self.angle_smooth_var.get()

                    for ch in range(16):
                        target_esp32s3_angle = self.calculate_esp32s3_angle(mp_angles, is_right, ch, current_thumb_mult)
                        
                        if self.thumb_transition:
                            el = current_time - self.thumb_transition_start
                            
                            f_ang_current = 0 if self.current_hand == "Right" else 180
                            f_ang_target = 0 if self.target_hand == "Right" else 180
                            
                            if el < 2.0: 
                                if ch in [8, 9]: target_esp32s3_angle = f_ang_current 
                                elif ch == 7: target_esp32s3_angle = 0  
                                elif ch == 6: target_esp32s3_angle = 0 if self.current_hand=="Right" else 180 
                                else: target_esp32s3_angle = f_ang_current
                            elif el < 3.5: 
                                if ch in [8, 9]: target_esp32s3_angle = f_ang_target 
                                elif ch == 7: target_esp32s3_angle = 0 
                                elif ch == 6: target_esp32s3_angle = 0 if self.target_hand=="Right" else 180 
                                else: target_esp32s3_angle = f_ang_target
                            else: 
                                self.thumb_transition, self.current_hand = False, self.target_hand
                        else:
                            if ch == 6: target_esp32s3_angle = 0 if self.current_hand == "Right" else 180
                        
                        self.smoothed_angles[ch] = (smooth_alpha * target_esp32s3_angle) + ((1.0 - smooth_alpha) * self.smoothed_angles[ch])
                        
                        if self.track_vars[ch].get() == 1: 
                            cmds += f"A,{ch},{int(self.smoothed_angles[ch])}\n"

                    self.send_command(cmds.strip())
                    self.last_serial_send_time = current_time

                if is_tracking_active and (current_time - self.last_log_time)*1000 >= delay_ms:
                    self.log(f"SENT: Hand={handedness_label} | Tracking")
                    self.last_log_time = current_time
            else:
                self.previous_pts_3d = None

            frame_resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (400, 400))
            self.lbl_video.imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_resized), master=self.root)
            self.lbl_video.configure(image=self.lbl_video.imgtk)
            
            self.lbl_canvas.imgtk = ImageTk.PhotoImage(image=Image.fromarray(canvas), master=self.root)
            self.lbl_canvas.configure(image=self.lbl_canvas.imgtk)

        self.root.after(10, self.update_frame)

    def on_closing(self):
        if self.is_connected and self.ser: self.ser.close()
        self.cap.release()
        self.landmarker.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandTrackerApp(root)
    root.mainloop()