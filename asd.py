import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.fft import rfft, rfftfreq

# Carpeta con los audios
AUDIO_DIR = "audios_test"
SAMPLE_RATE = 16000  # Asegúrate de que tus audios estén en 16kHz o ajusta aquí

# Crear carpeta para guardar gráficas
OUTPUT_DIR = "graficas_espectro"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Procesar cada archivo de audio
for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(".wav"):
        path = os.path.join(AUDIO_DIR, filename)
        waveform, sr = sf.read(path)
        
        # Asegurar formato adecuado
        if waveform.ndim > 1:
            waveform = waveform[:, 0]  # Mono

        # FFT
        N = len(waveform)
        yf = np.abs(rfft(waveform))
        xf = rfftfreq(N, 1 / sr)

        # Graficar
        plt.figure(figsize=(10, 4))
        plt.plot(xf, yf)
        plt.title(f"Espectro de: {filename}")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Magnitud")
        plt.grid(True)
        plt.tight_layout()
        
        # Guardar imagen
        output_path = os.path.join(OUTPUT_DIR, f"{filename[:-4]}_espectro.png")
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Guardado: {output_path}")
