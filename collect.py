"""
Passo 1 - Coleta de dados pela webcam para Sinais de Libras.

Uso:  python collect.py

Fluxo de Coleta Inteligente (300 fotos por classe):
  1. Ao apertar a tecla do sinal, comeca a FASE 1: FUNDO CLARO (luz acesa / parede clara).
  2. Ao atingir 150 fotos, o script PAUSA AUTOMATICAMENTE e avisa para mudar para o FUNDO ESCURO.
  3. Voce apaga a luz do quarto ou usa um fundo escuro (ex: cadeira, jaqueta preta).
  4. Aperta a mesma tecla do sinal para iniciar a FASE 2: FUNDO ESCURO (150 a 300 fotos).
  5. Ao atingir 300 fotos, o sinal e finalizado!

Teclas de atalho para os 10 sinais de Libras + classe "nada":
  1 ou a = A     2 ou b = B     3 ou c = C     4 ou d = D     5 ou i = I
  6 ou l = L     7 ou o = O     8 ou v = V     9 ou w = W     0 ou y = Y
  n ou ESPACO = nada (sem sinal / repouso)
  q = sair
"""
import os
import time
import cv2

# Mapa de teclas para os 10 sinais selecionados + classe nada
KEYS = {
    ord("1"): "A", ord("a"): "A",
    ord("2"): "B", ord("b"): "B",
    ord("3"): "C", ord("c"): "C",
    ord("4"): "D", ord("d"): "D",
    ord("5"): "I", ord("i"): "I",
    ord("6"): "L", ord("l"): "L",
    ord("7"): "O", ord("o"): "O",
    ord("8"): "V", ord("v"): "V",
    ord("9"): "W", ord("w"): "W",
    ord("0"): "Y", ord("y"): "Y",
    ord("n"): "nada", ord(" "): "nada",
}

CLASSES = ["A", "B", "C", "D", "I", "L", "O", "V", "W", "Y", "nada"]
DATA_DIR = "data"
IMG_SIZE = 224
SAVE_EVERY_N_FRAMES = 3
MAX_PER_CLASS = 300
HALF_PER_CLASS = MAX_PER_CLASS // 2  # 150 fotos para Fundo Claro e 150 para Fundo Escuro
COUNTDOWN = 3.0
DEBOUNCE = 0.5   # ignora repeticao de tecla por 0.5s

# Criar pastas para cada classe
for c in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, c), exist_ok=True)

def count(cls):
    p = os.path.join(DATA_DIR, cls)
    if not os.path.exists(p):
        return 0
    return len([f for f in os.listdir(p) if f.lower().endswith(".jpg")])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Nao consegui abrir a webcam. Feche outros apps que usam a camera "
                     "(Zoom, Meet, Teams...) ou tente cv2.VideoCapture(1).")

recording = None            # classe sendo gravada no momento
countdown_start = None
last_key_time = 0
frame_count = 0
waiting_for_dark_bg = False # flag de pausa automatica aos 150 frames
WIN = "Coleta - Sinais de Libras (Assistente Claro / Escuro)"

print("=" * 70)
print("Coleta de Dados - 10 Sinais de Libras + Nada (Assistente Fisico)")
print("Objetivo: 150 fotos no FUNDO CLARO + 150 fotos no FUNDO ESCURO = 300 fotos")
print("Teclas: 1-9, 0 ou letras (A,B,C,D,I,L,O,V,W,Y) | 'n'/'espaco' para nada | 'q' para sair")
print("=" * 70)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Falha ao ler a webcam.")
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    side = min(h, w) // 2 + 60
    x1, y1 = w // 2 - side // 2, h // 2 - side // 2
    roi = frame[y1:y1 + side, x1:x1 + side]

    curr_count = count(recording) if recording else 0

    # Logica de Estado e Mensagens na Tela
    if waiting_for_dark_bg:
        status = f"PAUSA: 150 fotos (Claro) concluidas! Mude para FUNDO ESCURO e aperte tecla"
        color = (0, 165, 255)  # Laranja de aviso
    elif recording and countdown_start is not None:
        left = COUNTDOWN - (time.time() - countdown_start)
        fase_nome = "FUNDO CLARO" if curr_count < HALF_PER_CLASS else "FUNDO ESCURO"
        if left > 0:
            status = f"{recording} [{fase_nome}]: Prepare a mao... {int(left) + 1}s"
            color = (0, 255, 255)
        else:
            countdown_start = None
    elif recording and countdown_start is None:
        if curr_count < HALF_PER_CLASS:
            status = f"GRAVANDO: [{recording}] - FASE 1: FUNDO CLARO ({curr_count}/{HALF_PER_CLASS})"
            color = (0, 255, 128)  # Verde claro
        else:
            status = f"GRAVANDO: [{recording}] - FASE 2: FUNDO ESCURO ({curr_count}/{MAX_PER_CLASS})"
            color = (0, 100, 255)  # Laranja / Avermelhado

        frame_count += 1
        if frame_count % SAVE_EVERY_N_FRAMES == 0:
            if curr_count < MAX_PER_CLASS:
                img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                timestamp = int(time.time() * 1000)
                cv2.imwrite(os.path.join(DATA_DIR, recording, f"{timestamp}.jpg"), img)
                curr_count += 1

                # Pausa automatica ao completar exatamente a metade (Fase 1: Claro)
                if curr_count == HALF_PER_CLASS:
                    waiting_for_dark_bg = True
                    saved_cls = recording
                    recording = None
                    print(f"\n>>> [PAUSA AUTOMATICA] 150 fotos (Fundo Claro) concluidas para '{saved_cls}'!")
                    print(">>> Agora ajuste o ambiente para FUNDO ESCURO (apague a luz ou use fundo escuro).")
                    print(f">>> Quando estiver pronto, aperte a tecla de '{saved_cls}' novamente para continuar a Fase 2!\n")
            else:
                print(f"\n>>> [CONCLUIDO] '{recording}': 300 fotos coletadas (150 Claro + 150 Escuro)!")
                recording = None
    else:
        status = "PARADO - aperte tecla (1-9, 0, n) para iniciar"
        color = (0, 255, 0)

    # Desenha ROI
    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), color, 3)

    # Barra superior de status
    cv2.rectangle(frame, (0, 0), (w, 42), (25, 25, 25), -1)
    cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

    # Alerta visual central caso esteja aguardando a troca para fundo escuro
    if waiting_for_dark_bg:
        overlay_alert = frame.copy()
        cv2.rectangle(overlay_alert, (x1 - 20, y1 + side // 2 - 45),
                      (x1 + side + 20, y1 + side // 2 + 45), (10, 10, 10), -1)
        cv2.addWeighted(overlay_alert, 0.8, frame, 0.2, 0, frame)
        cv2.putText(frame, "FASE 1 (CLARO) CONCLUIDA!", (x1 - 10, y1 + side // 2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, "Mude para Fundo Escuro", (x1 - 10, y1 + side // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "e aperte a tecla para continuar", (x1 - 10, y1 + side // 2 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # Painel lateral semi-transparente para contagem e fase de cada classe
    overlay = frame.copy()
    panel_h = 48 + len(CLASSES) * 26 + 10
    cv2.rectangle(overlay, (5, 48), (175, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, c in enumerate(CLASSES):
        key_hint = "0/y" if c == "Y" else ("n/esp" if c == "nada" else f"{i + 1}/{c.lower()}")
        qtd = count(c)
        is_current = (recording == c)

        if qtd >= MAX_PER_CLASS:
            phase_tag = "OK"
            txt_color = (0, 255, 0)
        elif qtd >= HALF_PER_CLASS:
            phase_tag = "Escuro"
            txt_color = (0, 165, 255) if is_current else (200, 200, 200)
        else:
            phase_tag = "Claro"
            txt_color = (0, 255, 255) if is_current else (220, 220, 220)

        thick = 2 if is_current else 1
        cv2.putText(frame, f"[{key_hint}] {c}: {qtd}/{MAX_PER_CLASS} ({phase_tag})",
                    (10, 70 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, txt_color, thick)

    # Rodapé com dicas de instrucao
    cv2.rectangle(frame, (0, h - 25), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "150 Claro + 150 Escuro por sinal | Varie angulo/distancia/mao | 'q': sair",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    cv2.imshow(WIN, frame)

    key = cv2.waitKey(10) & 0xFF
    if key == 255:
        continue
    if key == ord("q"):
        break

    if key in KEYS and (time.time() - last_key_time > DEBOUNCE):
        last_key_time = time.time()
        cls = KEYS[key]
        waiting_for_dark_bg = False  # limpa alerta caso retome a gravacao

        if recording == cls:
            recording = None
            print(f"Pausado: {cls} ({count(cls)} fotos)")
        else:
            qtd = count(cls)
            if qtd >= MAX_PER_CLASS:
                print(f"A classe '{cls}' ja atingiu o maximo de {MAX_PER_CLASS} fotos!")
            else:
                recording = cls
                countdown_start = time.time()
                proxima_fase = "Fundo Claro" if qtd < HALF_PER_CLASS else "Fundo Escuro"
                print(f"Iniciando contagem de 3s para '{cls}' [{proxima_fase}] (ja tem {qtd} fotos)...")
    elif key not in KEYS and key != 255:
        print(f"(tecla nao reconhecida: codigo {key})")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("Resumo final da coleta de imagens:")
for c in CLASSES:
    qtd = count(c)
    fase = "COMPLETO (150 claro + 150 escuro)" if qtd >= MAX_PER_CLASS else f"{qtd}/{MAX_PER_CLASS}"
    print(f" - Classe {c:>5}: {fase}")
print("=" * 50)
