"""
Passo 3 - Reconhecimento em Tempo Real de Sinais de Libras.

Uso: python play.py

Controles:
  q = sair
  s = salvar captura atual como exemplo (.png)
"""
import os
import json
import time
from collections import Counter, deque
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "gesture_model.keras"
CLASSES_PATH = "class_names.json"
IMG_SIZE = 224

if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
    raise SystemExit(
        f"Arquivo '{MODEL_PATH}' ou '{CLASSES_PATH}' nao encontrado.\n"
        "Execute primeiro: python train.py para treinar o modelo."
    )

print("Carregando modelo treinado...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)
print(f"Classes carregadas ({len(class_names)}): {class_names}")

# Descricoes amigaveis de cada sinal para exibicao na tela
SIGN_DESCRIPTIONS = {
    "A": "A - Punho fechado, polegar ao lado",
    "B": "B - Quatro dedos estendidos juntos",
    "C": "C - Mao curvada em formato 'C'",
    "D": "D - Indicador para cima, argola nos outros",
    "I": "I - Apenas mindinho levantado",
    "L": "L - Indicador e polegar a 90 graus",
    "O": "O - Todos os dedos tocam polegar (circulo)",
    "V": "V - Indicador e medio abertos em 'V'",
    "W": "W - Tres dedos levantados em 'W'",
    "Y": "Y - Polegar e mindinho abertos (hang loose)",
    "nada": "Nenhum sinal detectado (repouso)",
}

def predict(roi):
    img = cv2.cvtColor(cv2.resize(roi, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    probs = model.predict(img[None].astype("float32"), verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Nao consegui abrir a webcam. Verifique conexao e permissoes.")

recent_probs = deque(maxlen=6)  # suavizacao temporal pela media das probabilidades
prev_time = time.time()
fps = 0

WIN = "Reconhecimento de Libras - Deep Learning"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

print("Reconhecimento iniciado. Pressione 'q' na janela para sair.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Falha ao capturar imagem da webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Calcula FPS
    curr_time = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
    prev_time = curr_time

    # Regiao de Interesse (ROI central quadrada)
    side = min(h, w) // 2 + 60
    x1, y1 = w // 2 - side // 2, h // 2 - side // 2
    roi = frame[y1:y1 + side, x1:x1 + side]

    # Inferencia do modelo com suavizacao temporal das probabilidades
    _, _, raw_probs = predict(roi)
    recent_probs.append(raw_probs)
    probs = np.mean(recent_probs, axis=0)
    best_idx = int(np.argmax(probs))
    stable_label = class_names[best_idx]
    conf = float(probs[best_idx])

    # Cor e estilo do retangulo conforme a deteccao
    if stable_label != "nada" and conf >= 0.70:
        box_color = (0, 255, 0)      # Verde: sinal detectado com boa confianca
        box_thick = 3
    elif stable_label != "nada":
        box_color = (0, 200, 255)    # Amarelo/Laranja: confianca moderada
        box_thick = 2
    else:
        box_color = (180, 180, 180)  # Cinza: classe 'nada' / repouso
        box_thick = 2

    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), box_color, box_thick)

    # Painel superior com o resultado do sinal
    cv2.rectangle(frame, (0, 0), (w, 62), (25, 25, 25), -1)

    if stable_label != "nada":
        title_text = f"SINAL: {stable_label}  ({conf:.1%})"
        title_color = (0, 255, 128) if conf >= 0.7 else (0, 220, 255)
    else:
        title_text = f"SINAL: nada (repouso - {conf:.1%})"
        title_color = (180, 180, 180)

    cv2.putText(frame, title_text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, title_color, 2)

    desc = SIGN_DESCRIPTIONS.get(stable_label, stable_label)
    cv2.putText(frame, desc, (15, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

    # Indicador de FPS no canto superior direito
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Painel lateral esquerdo: Barras de probabilidade de todas as classes
    panel_w = 175
    panel_h = 75 + len(class_names) * 22
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 68), (panel_w, panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "Probabilidades:", (12, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    for i, c in enumerate(class_names):
        p = probs[i]
        bar_len = int(p * 75)
        y_pos = 108 + i * 22

        # Destaque se for a classe vencedora
        is_winner = (c == stable_label)
        bar_color = (0, 255, 0) if is_winner else (0, 165, 255)
        text_color = (0, 255, 255) if is_winner else (220, 220, 220)

        # Fundo da barra
        cv2.rectangle(frame, (12, y_pos - 10), (12 + 75, y_pos + 2), (50, 50, 50), -1)
        # Barra de probabilidade preenchida
        if bar_len > 0:
            cv2.rectangle(frame, (12, y_pos - 10), (12 + bar_len, y_pos + 2), bar_color, -1)

        # Nome da classe e porcentagem
        cv2.putText(frame, f"{c[:6]:<6} {p*100:4.0f}%", (95, y_pos - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1)

    # Rodapé com instruções
    cv2.rectangle(frame, (0, h - 28), (w, h), (25, 25, 25), -1)
    cv2.putText(frame, "Posicione a mao dentro do quadrado | 's': salvar foto | 'q': sair",
                (15, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)

    cv2.imshow(WIN, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        snap_name = f"captura_{stable_label}_{int(time.time())}.png"
        cv2.imwrite(snap_name, frame)
        print(f"Captura salva como: {snap_name}")

cap.release()
cv2.destroyAllWindows()
