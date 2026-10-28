"""
Passo 2 - Treino com Transfer Learning (MobileNetV2) para Sinais de Libras.

Uso: python train.py

Ideia: a base (MobileNetV2, pre-treinada na ImageNet) ja sabe "enxergar"
bordas, texturas e formas. Congelamos ela e treinamos a camada final
que classifica as imagens entre as classes encontradas na pasta data.

Split Balanceado:
  Como a coleta foi feita em 2 fases (1ª metade: Fundo Claro, 2ª metade: Fundo Escuro),
  o split separa 80% treino e 20% validacao DE CADA UMA DAS METADES.
  Assim, tanto o treino quanto a validacao contem fotos claras e escuras de forma equilibrada!

Ao final, gera automaticamente o arquivo 'dataset_imagens.pdf' exigido para a entrega.
"""
import os
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = "data"
IMG_SIZE = 224
BATCH = 16
EPOCHS = 8
VAL_FRACTION = 0.2
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

if not os.path.exists(DATA_DIR):
    raise SystemExit(f"A pasta '{DATA_DIR}' nao existe. Execute o collect.py primeiro para coletar os dados!")

class_names = sorted(d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d)))

if not class_names:
    raise SystemExit(f"Nenhuma subpasta de classe encontrada em '{DATA_DIR}'. Colete dados antes de treinar.")

print("=" * 65)
print(f"Classes encontradas ({len(class_names)}): {class_names}")
print("=" * 65)

# Split balanceado entre as duas iluminacoes (Fundo Claro e Fundo Escuro)
train_paths, train_y, val_paths, val_y = [], [], [], []
for idx, cls in enumerate(class_names):
    pasta_classe = os.path.join(DATA_DIR, cls)
    files = sorted(os.listdir(pasta_classe))
    files = [os.path.join(pasta_classe, f) for f in files if f.lower().endswith(".jpg")]
    if len(files) == 0:
        print(f" [AVISO] A classe '{cls}' nao possui fotos .jpg!")
        continue

    n = len(files)
    half = n // 2

    # Metade 1 (Fundo Claro) e Metade 2 (Fundo Escuro)
    fase_claro = files[:half]
    fase_escuro = files[half:]

    cut_c = int(len(fase_claro) * (1 - VAL_FRACTION))
    cut_e = int(len(fase_escuro) * (1 - VAL_FRACTION))

    cls_train = fase_claro[:cut_c] + fase_escuro[:cut_e]
    cls_val = fase_claro[cut_c:] + fase_escuro[cut_e:]

    train_paths += cls_train
    train_y += [idx] * len(cls_train)

    val_paths += cls_val
    val_y += [idx] * len(cls_val)

    val_c = len(fase_claro) - cut_c
    val_e = len(fase_escuro) - cut_e
    print(f" {cls:>10}: {len(cls_train)} treino ({cut_c} claro + {cut_e} escuro) / "
          f"{len(cls_val)} validacao ({val_c} claro + {val_e} escuro) [total {n}]")

if not train_paths or not val_paths:
    raise SystemExit("Dados insuficientes para treino e validacao. Colete fotos com collect.py.")

def load(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, label

def make_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH)
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(train_paths, train_y, True)
val_ds = make_ds(val_paths, val_y, False)

# Data augmentation calibrado para evitar que o modelo decore posicao/distancia fixa
# O zoom negativo (-0.30) aproxima a mao (como no play.py) e corta a borda inferior (pulso/manga)
augment = tf.keras.Sequential([
    tf.keras.layers.RandomTranslation(0.15, 0.15, fill_mode="nearest"),
    tf.keras.layers.RandomZoom((-0.30, 0.15), fill_mode="nearest"),
    tf.keras.layers.RandomRotation(0.10, fill_mode="nearest"),
    tf.keras.layers.RandomBrightness(0.20),
    tf.keras.layers.RandomContrast(0.20),
])

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False  # Etapa 1: base 100% congelada para nao distorcer os pesos pre-treinados

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)

# Combina Average Pooling e Max Pooling com LayerNormalization (estavel em treino e validacao)
avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)
x = tf.keras.layers.Concatenate()([avg_pool, max_pool])
x = tf.keras.layers.LayerNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# --- ETAPA 1: Treinar apenas a cabeca classificadora (estabiliza os pesos novos) ---
print("\n[Etapa 1/2] Treinando camada classificadora (base congelada)...")
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

hist1 = model.fit(train_ds, validation_data=val_ds, epochs=6)

# --- ETAPA 2: Fine-Tuning suave nas ultimas 20 camadas para focar nos dedos ---
print("\n[Etapa 2/2] Ajuste fino (Fine-Tuning) nas camadas finais da MobileNetV2...")
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    )
]

hist2 = model.fit(train_ds, validation_data=val_ds, epochs=6, callbacks=callbacks)

# Combina o historico das duas etapas para o grafico curvas.png
history_combined = {
    k: hist1.history[k] + hist2.history[k] for k in hist1.history
}
model.save("gesture_model.keras")

with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

# Curvas de treino vs validacao
plt.figure(figsize=(10, 4))
for i, (m, t) in enumerate([("accuracy", "Acuracia"), ("loss", "Loss")]):
    plt.subplot(1, 2, i + 1)
    plt.plot(history_combined[m], label="treino")
    plt.plot(history_combined["val_" + m], label="validacao")
    plt.title(t); plt.xlabel("epoca"); plt.legend()
plt.tight_layout(); plt.savefig("curvas.png")
print("\nGrafico de curvas de aprendizado salvo em: curvas.png")

# Matriz de confusao
y_true = np.array(val_y)
y_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
cm = np.zeros((len(class_names), len(class_names)), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

print("\nMatriz de confusao (linhas = real, colunas = previsto):")
header = " " * 11 + " ".join(f"{c[:6]:>6}" for c in class_names)
print(header)
for i, c in enumerate(class_names):
    row = f"{c[:10]:>10} " + " ".join(f"{v:>6}" for v in cm[i])
    print(row)

print("\nModelo treinado salvo em gesture_model.keras e classes em class_names.json.")

# ---- DICA: gera um PDF com as imagens do dataset, separadas por classe ----
def gerar_pdf_dataset(data_dir="data", saida="dataset_imagens.pdf", cols=10, qualidade=70):
    print("\nGerando PDF com as imagens do dataset...")
    pag_w, pag_h, margem, topo = 1240, 1754, 40, 110  # pagina A4 a 150 dpi
    cell = (pag_w - 2 * margem) // cols               # tamanho de cada miniatura
    rows = (pag_h - topo - margem) // cell
    por_pagina = cols * rows

    try:
        fonte = ImageFont.load_default(size=30)
    except TypeError:  # Pillow mais antigo
        fonte = ImageFont.load_default()

    paginas = []
    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)))

    for classe in classes:  # uma secao por classe
        pasta = os.path.join(data_dir, classe)
        fotos = sorted(f for f in os.listdir(pasta) if f.lower().endswith(".jpg"))
        if not fotos:
            continue
        n_pag = max(1, -(-len(fotos) // por_pagina))  # divisao arredondada para cima
        for p in range(n_pag):
            pagina = Image.new("RGB", (pag_w, pag_h), "white")
            draw = ImageDraw.Draw(pagina)
            draw.text((margem, 40),
                      f"Classe: {classe} | {len(fotos)} imagens | pagina {p + 1}/{n_pag}",
                      fill="black", font=fonte)
            for i, nome in enumerate(fotos[p * por_pagina:(p + 1) * por_pagina]):
                try:
                    img = Image.open(os.path.join(pasta, nome)).convert("RGB")
                    img.thumbnail((cell - 6, cell - 6))
                    x = margem + (i % cols) * cell + 3
                    y = topo + (i // cols) * cell + 3
                    pagina.paste(img, (x, y))
                except Exception as e:
                    print(f"Erro ao processar imagem {nome}: {e}")
            paginas.append(pagina)

    if paginas:
        paginas[0].save(saida, save_all=True, append_images=paginas[1:],
                        quality=qualidade, resolution=150)
        print(f"PDF salvo com sucesso: {saida} ({len(paginas)} paginas)")
    else:
        print("Nenhuma pagina gerada (verifique se existem fotos na pasta data).")

gerar_pdf_dataset(DATA_DIR)

print("\nConcluido! Para testar o modelo em tempo real, execute: python play.py")
