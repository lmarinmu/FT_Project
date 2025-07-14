import tkinter as tk
import threading
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
import soundfile as sf
import datetime
import os
import csv

# Cargar modelo YAMNet
model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = model.class_map_path().numpy().decode()

# Leer clases
class_names = []
with tf.io.gfile.GFile(class_map_path) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_names.append(row['display_name'])

# Lista de sonidos peligrosos
agrupaciones = {
    "Pitido": [
        "Honk", "Beep, bleep", "Vehicle horn, car horn, honking", "Air horn, truck horn", "Reverse beeper"
    ],
    "Sirena": [
        "Siren", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)", "Emergency vehicle"
    ],
    "Motor": [
        "Engine", "Idling", "Revving", "Motor vehicle (road)", "Engine knocking", "Mechanical fan"
    ],
    "Choque": [
        "Crash", "Skidding", "Tire squeal"
    ],
    "Explosión": [
        "Boom", "Explosion", "Burst, pop"
    ],
    "Disparo": [
        "Gunshot, gunfire"
    ],
    "Alarma": [
        "Car alarm", "Alarm"
    ],
    "Vehículo": [
        "Vehicle", "Car", "Truck", "Bus", "Motorcycle", "Car passing by", "Road, roadway noise", "Traffic noise", "Rail transport", "Train", "Subway, metro, underground", "Light engine"
    ],
    "Grito": [
        "Shout", "Scream"
    ]
}

def traducir_sonido(detected):
    for categoria, clases in agrupaciones.items():
        if detected in clases:
            return categoria
    return "Otro sonido"

# Configuración
SAMPLE_RATE = 16000
DURATION = 2.0  # segundos

# Carpeta local segura para guardar los audios
AUDIO_DIR = "C:/audios_guardados"
os.makedirs(AUDIO_DIR, exist_ok=True)

class App:
    def __init__(self, root):
        self.root = root
        self.running = False
        self.root.geometry("450x500")
        self.root.title("Detector de Sonidos")

        # Título
        self.title_label = tk.Label(root, text="DETECTOR DE SONIDOS DE RIESGO", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=10)

        # Canvas y círculo centrado
        self.canvas = tk.Canvas(root, width=400, height=300)
        self.canvas.pack()
        cx, cy, r = 200, 150, 80
        self.circle = self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill='grey', width=3)

        # Texto del sonido detectado
        self.sound_label = tk.Label(root, text="", font=("Arial", 14))
        self.sound_label.pack(pady=10)

        # Botones
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        self.btn_start = tk.Button(btn_frame, text='Escuchar', width=15, command=self.start)
        self.btn_stop = tk.Button(btn_frame, text='Detener', width=15, command=self.stop)
        self.btn_start.pack(side='left', padx=10)
        self.btn_stop.pack(side='left', padx=10)

    def start(self):
        if not self.running:
            self.running = True
            self.canvas.itemconfig(self.circle, fill='green')  # Verde al iniciar
            self.sound_label.config(text="Escuchando...")
            threading.Thread(target=self.listen_loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.canvas.itemconfig(self.circle, fill='grey')
        self.sound_label.config(text="")

    def listen_loop(self):
        while self.running:
            audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                           channels=1, dtype='float32')
            sd.wait()
            waveform = np.squeeze(audio)

            # Validar audio
            if (
                waveform.size == 0 or
                np.any(np.isnan(waveform)) or
                np.any(np.isinf(waveform)) or
                np.max(np.abs(waveform)) < 1e-5
            ):
                print("⚠️ Fragmento inválido o silencioso, no se guarda.")
                continue

            # Inferencia YAMNet
            scores, _, _ = model(waveform)
            mean_scores = tf.reduce_mean(scores, axis=0).numpy()
            top_idx = np.argmax(mean_scores)
            detected = class_names[top_idx]
            categoria = traducir_sonido(detected)

            # Interfaz gráfica
            if categoria != "Otro sonido":
                self.canvas.itemconfig(self.circle, fill='red')
            else:
                self.canvas.itemconfig(self.circle, fill='green')

            self.sound_label.config(text=f"Sonido detectado: {categoria}")

            # Guardar audio
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            categoria_safe = categoria.replace(" ", "_")
            filename = f"{AUDIO_DIR}/audio_{timestamp}_{categoria_safe}.wav"
            try:
                sf.write(filename, waveform, SAMPLE_RATE)
            except Exception as e:
                print(f"❌ Error al guardar: {filename} → {e}")

            self.root.update()


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
