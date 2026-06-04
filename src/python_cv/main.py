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
        self.root.title("Ultra-Fast Hand Tracking + UART")
        self.root.geometry("1250x850")
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
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top_frame = tk.Frame(main_frame)
        top_frame.pack(fill=tk.X)

        self.lbl_video = tk.Label(top_frame, bg="black", width=400, height=400)
        self.lbl_video.pack(side=tk.LEFT, padx=5)

        self.lbl_canvas = tk.Label(top_frame, bg="white", width=400, height=400)
        self.lbl_canvas.pack(side=tk.LEFT, padx=5)

        ctrl_frame = tk.Frame(top_frame, width=350)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20)

        tk.Label(ctrl_frame, text="Налаштування UART", font=("Arial", 14, "bold")).pack(pady=10)

        port_frame = tk.Frame(ctrl_frame)
        port_frame.pack(fill=tk.X, pady=5)
        self.port_var = tk.StringVar()
        self.port_dropdown = ttk.Combobox(port_frame, textvariable=self.port_var, state="readonly")
        self.port_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(port_frame, text="Оновити", command=self.refresh_ports).pack(side=tk.LEFT, padx=5)

        self.btn_connect = tk.Button(ctrl_frame, text="Підключитися", bg="lightgreen", font=("Arial", 12, "bold"), command=self.toggle_connection)
        self.btn_connect.pack(fill=tk.X, pady=15)

        tk.Label(ctrl_frame, text="Затримка логів (мс):").pack(anchor=tk.W, pady=(10, 0))
        self.delay_slider = tk.Scale(ctrl_frame, from_=100, to=2000, orient=tk.HORIZONTAL)
        self.delay_slider.set(500)
        self.delay_slider.pack(fill=tk.X)

        bottom_frame = tk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        tk.Label(bottom_frame, text="Системна Консоль", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        scroll = tk.Scrollbar(bottom_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.console_text = tk.Text(bottom_frame, height=15, bg="black", fg="lime", font=("Consolas", 10), yscrollcommand=scroll.set)
        self.console_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.console_text.yview)

        self.log("[INFO] Інтерфейс завантажено.")

    def refresh_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        if not ports:
            ports = ["Немає портів"]
        self.port_dropdown['values'] = ports
        self.port_dropdown.current(0)

    def toggle_connection(self):
        if not self.is_connected:
            port = self.port_var.get()
            if port == "Немає портів" or port == "":
                self.log("[WARNING] Порт не вибрано.")
                return
            try:
                self.ser = serial.Serial(port, 115200, timeout=1)
                self.is_connected = True
                self.btn_connect.config(text="Відключитися", bg="salmon")
                self.log(f"[INFO] Підключено до порту: {port}")
            except Exception as e:
                self.log(f"[ERROR] Помилка підключення: {e}")
        else:
            if self.ser:
                self.ser.close()
            self.is_connected = False
            self.btn_connect.config(text="Підключитися", bg="lightgreen")
            self.log("[INFO] Відключено від порту.")

    def log(self, message):
        self.console_text.insert(tk.END, message + "\n")
        self.console_text.see(tk.END)

    def calculate_arduino_angle(self, mp_angle, is_right_hand, channel):
        if channel == 6:
            return 0 if is_right_hand else 180

        if channel == 7:
            val = (mp_angle - 90) * 3
            return int(np.clip(val, 0, 180))

        if is_right_hand:
            val = mp_angle - 90
        else:
            val = 270 - mp_angle

        return int(np.clip(val, 0, 180))

    def update_frame(self):
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

            if res and res.hand_landmarks:
                hand_landmarks = res.hand_landmarks[0]
                
                if res.handedness and len(res.handedness[0]) > 0:
                    mp_label = res.handedness[0][0].category_name
                    handedness_label = "Right" if mp_label == "Left" else "Left"
                else:
                    handedness_label = "Unknown"
                
                is_right = (handedness_label == "Right")

                if handedness_label != "Unknown":
                    if self.current_hand is None:
                        self.current_hand = handedness_label
                    
                    if handedness_label != self.current_hand and not self.thumb_transition:
                        self.thumb_transition = True
                        self.thumb_transition_start = current_time
                        self.target_hand = handedness_label
                        self.log(f"[INFO] Зміна руки: {self.current_hand} -> {self.target_hand}")

                pts_2d_raw = np.array([(lm.x, lm.y) for lm in hand_landmarks])
                h, w, _ = frame.shape
                
                for connection in CONNECTIONS:
                    s, e = pts_2d_raw[connection[0]], pts_2d_raw[connection[1]]
                    cv2.line(frame, (int(s[0] * w), int(s[1] * h)), (int(e[0] * w), int(e[1] * h)), (0, 255, 0), 2)
                cv2.putText(frame, handedness_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0) if not is_right else (0,0,255), 2)

                pts_3d = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks])
                is_flipped = pts_3d[2, 0] > pts_3d[17, 0]

                if is_flipped:
                    min_x, max_x = np.min(pts_3d[:, 0]), np.max(pts_3d[:, 0])
                    center_x = (min_x + max_x) / 2.0
                    pts_3d[:, 0] = 2 * center_x - pts_3d[:, 0]

                if self.previous_pts_3d is None or is_flipped != self.previous_is_flipped:
                    self.previous_pts_3d = pts_3d.copy()
                else:
                    pts_3d = (pts_3d * self.SMOOTHING_FACTOR) + (self.previous_pts_3d * (1.0 - self.SMOOTHING_FACTOR))
                    self.previous_pts_3d = pts_3d.copy()
                self.previous_is_flipped = is_flipped

                p_tip = get_joint_angle(pts_3d[20], pts_3d[19], pts_3d[18])
                p_mid = get_joint_angle(pts_3d[19], pts_3d[18], pts_3d[17])
                p_base = get_joint_angle(pts_3d[18], pts_3d[17], pts_3d[0])
                r_tip = get_joint_angle(pts_3d[16], pts_3d[15], pts_3d[14])
                r_mid = get_joint_angle(pts_3d[15], pts_3d[14], pts_3d[13])
                r_base = get_joint_angle(pts_3d[14], pts_3d[13], pts_3d[0])
                t_tip = get_joint_angle(pts_3d[4], pts_3d[3], pts_3d[2])
                t_mid = get_joint_angle(pts_3d[3], pts_3d[2], pts_3d[1])
                t_mid90 = get_joint_angle(pts_3d[2], pts_3d[1], pts_3d[0])
                m_tip = get_joint_angle(pts_3d[12], pts_3d[11], pts_3d[10])
                m_mid = get_joint_angle(pts_3d[11], pts_3d[10], pts_3d[9])
                m_base = get_joint_angle(pts_3d[10], pts_3d[9], pts_3d[0])
                i_tip = get_joint_angle(pts_3d[8], pts_3d[7], pts_3d[6])
                i_mid = get_joint_angle(pts_3d[7], pts_3d[6], pts_3d[5])
                i_base = get_joint_angle(pts_3d[6], pts_3d[5], pts_3d[0])

                mp_angles_map = {
                    0: p_tip, 1: p_mid, 2: p_base,
                    3: r_tip, 4: r_mid, 5: r_base,
                    6: 0,
                    7: t_mid90, 8: t_mid, 9: t_tip,
                    10: m_base, 11: m_tip, 12: m_mid,
                    13: i_tip, 14: i_mid, 15: i_base
                }

                canvas_pts = pts_3d[:, :2]
                min_c, max_c = np.min(canvas_pts, axis=0), np.max(canvas_pts, axis=0)
                center = (min_c + max_c) / 2
                size = np.max(max_c - min_c) or 0.01
                scale = (400 * 0.7) / size
                offset = 200 - (center * scale)
                
                for connection in CONNECTIONS:
                    s, e = canvas_pts[connection[0]], canvas_pts[connection[1]]
                    cv2.line(canvas, (int(s[0]*scale+offset[0]), int(s[1]*scale+offset[1])), (int(e[0]*scale+offset[0]), int(e[1]*scale+offset[1])), (0, 255, 0), 3)

                if current_time - self.last_serial_send_time >= 0.02:
                    arduino_commands = ""
                    for ch in range(16):
                        ard_angle = self.calculate_arduino_angle(mp_angles_map[ch], is_right, ch)

                        if self.thumb_transition:
                            elapsed = current_time - self.thumb_transition_start
                            
                            fold_angle = 0 if self.target_hand == "Right" else 180
                            old_base = 0 if self.current_hand == "Right" else 180
                            new_base = 0 if self.target_hand == "Right" else 180
                            
                            if elapsed < 2.0:
                                if ch in [7, 8, 9]:
                                    ard_angle = fold_angle
                                elif ch == 6:
                                    ard_angle = old_base
                                    
                            elif elapsed < 3.5:
                                if ch in [7, 8, 9]:
                                    ard_angle = fold_angle
                                elif ch == 6:
                                    ard_angle = new_base
                                    
                            else:
                                self.thumb_transition = False
                                self.current_hand = self.target_hand
                                self.log("[INFO] Зміна сторони роботи великого пальця завершено.")
                        else:
                            if ch == 6:
                                ard_angle = 0 if self.current_hand == "Right" else 180

                        arduino_commands += f"A:{ch}:{ard_angle}\n"

                    if self.is_connected and self.ser and self.ser.is_open:
                        try:
                            self.ser.write(arduino_commands.encode('utf-8'))
                            self.last_serial_send_time = current_time
                        except Exception as e:
                            self.log(f"[ERROR] Помилка відправки UART: {e}")
                            self.toggle_connection()

                if (current_time - self.last_log_time) * 1000 >= delay_ms:
                    self.log(f"[INFO] Трекінг: {handedness_label} рука")
                    self.last_log_time = current_time

            else:
                self.previous_pts_3d = None
                
                if (current_time - self.last_log_time) * 1000 >= delay_ms:
                    self.log("[INFO] Руку не виявлено. Положення зафіксовано.")
                    self.last_log_time = current_time

            frame_resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (400, 400))
            self.lbl_video.imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
            self.lbl_video.configure(image=self.lbl_video.imgtk)

            self.lbl_canvas.imgtk = ImageTk.PhotoImage(image=Image.fromarray(canvas))
            self.lbl_canvas.configure(image=self.lbl_canvas.imgtk)

        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.log("[INFO] Завершення роботи...")
        if self.is_connected and self.ser:
            self.ser.close()
        self.cap.release()
        self.landmarker.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandTrackerApp(root)
    root.mainloop()