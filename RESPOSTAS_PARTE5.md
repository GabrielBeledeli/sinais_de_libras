# Atividade Prática: Reconhecimento de Sinais de Libras
## Parte 5: Perguntas e Respostas Conceituais

* **Instituição:** Centro Universitário Campo Real  
* **Curso:** Engenharia de Software  
* **Disciplina:** 2026/2 TA - Machine Learning e Deep Learning  
* **Docente:** Profa. Thaís Amanda Santos  
* **Autores:** Gabriel Beledeli Hul; Nicolas Miguel; Vinicius Buskievicz; Alisson Eraldo  

---

### 1. Esse modelo é um deep learning ou um machine learning? Justifique sua resposta com base na arquitetura usada no projeto.

**Resposta:**
O modelo desenvolvido é estritamente de **Deep Learning** (que é um subcampo especializado do Machine Learning). 

**Justificativa com base na arquitetura:**
1. **Extração Automática de Características (*Feature Learning*):** No Machine Learning tradicional (como SVM, Random Forest ou Regressão Logística), o desenvolvedor precisa extrair características manualmente (*feature engineering*), calculando descritores matemáticos artesanais (como contornos, histogramas de gradientes HOG ou momentos de Hu). No nosso projeto, o modelo recebe diretamente a **imagem RGB em pixels brutos (224x224x3)** e aprende de forma autônoma a hierarquia de representações visuais.
2. **Rede Convolucional Profunda (CNN):** A espinha dorsal do projeto é a **MobileNetV2**, uma rede neural convolucional profunda composta por 53 camadas com blocos residuais invertidos (*inverted residuals*) e convoluções separáveis em profundidade (*depthwise separable convolutions*), totalizando mais de 3 milhões de parâmetros.
3. **Transfer Learning e Camadas Finais:** Utilizamos a técnica de aprendizado por transferência congelando a base pré-treinada da MobileNetV2 (que já compreende bordas, padrões e texturas da ImageNet) e construímos uma "cabeça" neural profunda contendo:
   - Uma camada de redução espacial **GlobalAveragePooling2D**;
   - Uma camada de regularização contra sobreajuste **Dropout(0.3)**;
   - Uma camada densa final **Dense(11, activation='softmax')** com pesos totalmente conexos que geram as 11 probabilidades de saída.

Portanto, por se basear em múltiplas camadas convolucionais ocultas e representações neurais hierárquicas ponta a ponta sem extração manual de atributos, classifica-se categoricamente como **Deep Learning**.

---

### 2. Que características da imagem o modelo usa para distinguir os gestos? Descreva quais características da mão você acredita que o modelo usa e proponha um experimento para testar a sua hipótese. O que você observou?

**Resposta:**
Com base no comportamento das convoluções e nos resultados da nossa matriz de confusão, o modelo utiliza as seguintes características visuais:
1. **Projeções e quantidade de dedos estendidos:** O modelo identifica a presença de descontinuidades verticais e diagonais na silhueta. Por exemplo:
   - Na letra **B**: uma parede contínua de 4 dedos juntos;
   - Na letra **V**: duas linhas separadas em ângulo aberto;
   - Na letra **W**: três projeções distintas;
   - Na letra **I**: uma única haste estreita no canto externo (o mindinho);
   - Na letra **L**: o ângulo ortogonal perpendicular de 90° entre o polegar e o indicador.
2. **Espaços negativos e vazios:** Nas letras **C** e **O**, o modelo aprende o padrão de curvatura e o espaço escuro/oco central formado pelo fechamento ou curvatura dos dedos com o polegar.
3. **Massa compacta (punho fechado):** Em sinais como a letra **A**, o modelo reconhece um bloco maciço e fechado com apenas uma saliência lateral (o polegar ereto ao lado da palma).

#### Proposta de Experimento para Testar a Hipótese:
* **Hipótese:** O modelo não compreende a mão anatomicamente, mas sim o contorno geométrico das pontas dos dedos e a presença de bloqueios na silhueta. Se oclusões parciais alterarem a projeção dos dedos, o modelo deve trocar a predição para sinais com geometrias correlatas.
* **Procedimento:**
  1. Executar o `play.py` e posicionar a mão fazendo o sinal da letra **V** (confiança acima de 95%).
  2. Com a outra mão ou com um pedaço de fita/cartão escuro, tampar exclusivamente o dedo indicador, deixando apenas o dedo médio exposto.
  3. No sinal da letra **I**, dobrar o dedo mindinho para dentro da palma simulando a letra **A**.
* **O que foi observado na prática:**
  - Ao cobrir um dos dedos do **V**, a predição muda instantaneamente para **D** ou **I** (pois a rede identifica apenas uma única haste vertical).
  - Mais evidente ainda foi o resultado observado empiricamente na nossa **Matriz de Confusão do `train.py`**: de 60 fotos da letra **I** no conjunto de validação, **7 fotos foram preditas como A**. Isso ocorreu exatamente porque ambos os sinais compartilham o mesmo bloco maciço de punho fechado; em ângulos em que o dedo mindinho ficou ligeiramente inclinado ou mesclado com a borda, a rede interpretou a imagem como um punho fechado puro, ativando a classe **A**. Isso comprova que a projeção isolada dos dedos é a característica crítica usada pelo modelo.

---

### 3. Que decisões na coleta de dados ajudam ou prejudicam o modelo? Descreva duas decisões que você tomou ao coletar os dados e como cada uma pode influenciar o resultado do modelo.

**Resposta:**

#### Decisão 1 (Que ajudou muito): Dividir a coleta em 2 fases equilibradas (150 fotos Claro + 150 fotos Escuro) com Pausa Automática e Split Balanceado no Treino
* **Como influenciou positivamente o modelo:**
  Redes neurais profundas são propensas a aprender "atalhos espúrios" (*shortcut learning*). Se todas as fotos fossem coletadas na mesma sala com iluminação e parede idênticas, o modelo correria o risco de associar a cor e as sombras da parede aos gestos. Ao coletar metade com luz do quarto acesa (fundo claro) e a outra metade com luz apagada e apenas o monitor iluminando a mão (fundo escuro com alto contraste), forçamos a rede a ignorar as cores do fundo e a aprender padrões de forma invariantes à iluminação. 
  Além disso, adaptar o `train.py` para pegar 80% de treino e 20% de validação **de cada uma das metades** garantiu que o conjunto de validação refletisse ambas as condições lumínicas de forma justa (50/50), o que resultou na nossa acurácia de **98.64%**.

#### Decisão 2 (Que poderia prejudicar caso não cuidada): Variação de distância e ângulo durante a gravação
* **Como influenciou o resultado:**
  Durante a contagem e gravação de cada sinal, nós realizamos pequenas inclinações de ângulo e aproximações/afastamentos da mão em relação à câmera (mantendo a mesma mão para consistência).
  * **Se não tivéssemos variado o ângulo e a distância (prejuízo):** Se a mão ficasse estritamente imóvel na mesma posição durante todas as 300 fotos, os dados seriam praticamente cópias idênticas. A rede sofreria de severo *overfitting* de pose: qualquer leve inclinação de 10 a 15 graus ou aproximação natural ao testar no `play.py` faria o modelo falhar na predição.
  * **Cuidado tomado com a borda (evitando prejuízo):** Tomamos o cuidado de manter toda a mão rigorosamente contida dentro do quadrado verde (ROI). Se a ponta dos dedos cortasse a borda superior do quadrado, a rede aprenderia a classificar uma letra **B** truncada parecendo uma letra **A**, inserindo ruído destrutivo nos gradientes do treinamento.

---

### 4. Teste: como o modelo se comporta se você treinar com dados que têm pessoas no fundo? Faça o experimento e descreva o que aconteceu com as previsões. Explique por que o modelo teve esse comportamento e proponha duas estratégias para melhorá-lo.

**Resposta:**

#### 1. O Experimento na Prática (Com Pessoa no Fundo vs. Fundo Liso):
Nós realizamos exatamente os dois cenários na prática durante o desenvolvimento do trabalho:

* **1ª Coleta (Com pessoa/rosto aparecendo no fundo):**
  Na primeira tentativa, gravamos os sinais no `collect.py` sentados normalmente em frente à webcam, de modo que o nosso próprio rosto, pescoço e tronco apareciam no fundo do quadrado de captura atrás da mão.
  * **O que aconteceu com as previsões:** Deu bastante problema na hora de testar em tempo real no `play.py`! As previsões ficaram instáveis, a confiança oscilava muito e o modelo errava vários sinais ou apontava falsos positivos mesmo quando mudávamos levemente a posição da cabeça ou do corpo ao fundo.
* **2ª Coleta (Sem pessoas no fundo — Apenas fundo claro liso e fundo escuro liso):**
  Para resolver o problema, refizemos todo o dataset no `collect.py` garantindo que nenhuma pessoa ou rosto aparecesse no enquadramento — utilizando apenas uma parede clara lisa (Fase 1) e um fundo preto liso (Fase 2).
  * **Resultado:** Só conseguimos um resultado realmente estável e preciso no `play.py` após essa segunda coleta com fundos lisos, onde o modelo passou a focar exclusivamente no formato dos dedos!

#### 2. Por que o modelo teve esse comportamento ruim com pessoas no fundo?
1. **Confusão de tons de pele e contornos:** O rosto, pescoço e braços da pessoa no fundo possuem a mesma cor de pele, textura e gradientes de borda que a mão em primeiro plano. A rede convolucional acaba misturando os traços do rosto com os dedos da mão.
2. **Ativação global dos filtros (Pooling Espacial):** As camadas de *Global Pooling* (`GlobalAveragePooling2D` / `GlobalMaxPooling2D`) agregam as ativações de toda a área de `224x224` pixels. A CNN não sabe semanticamente que "deve ignorar o rosto lá atrás e olhar só para a mão da frente"; para ela, tudo são matrizes de pixels que ativam fortemente os filtros da MobileNetV2 (que foi pré-treinada na ImageNet justamente para detectar formas humanas e objetos).
3. **Correlação espúria (*Overfitting* no fundo):** Quando há uma pessoa no fundo durante o treino, a rede "trapaceia" decorando a posição da cabeça, da gola da camiseta ou das sombras do rosto que estavam presentes naquele momento específico, falhando assim que a pessoa se mexe no teste ao vivo.

#### 3. Duas estratégias para melhorar esse comportamento:

1. **Segmentação ou Rastreamento de Mão em Primeiro Plano (MediaPipe Hands / Máscara de Pele):**
   * *Como funciona:* Em vez de passar o recorte bruto da webcam (com fundo e pessoas) direto para a MobileNetV2, utiliza-se um detector prévio como o *MediaPipe Hands*. Ele localiza exclusivamente os 21 pontos articulares (*landmarks*) da mão em primeiro plano e recorta/mascara todo o fundo em preto puro antes de enviar para a rede classificadora. Assim, qualquer pessoa ou rosto ao fundo é 100% removido da imagem.
2. **Treinamento em 2 Etapas com Fine-Tuning + Data Augmentation Pesado de Fundo:**
   * *Como funciona:* Utilizar técnicas de *Data Augmentation* espacial (`RandomTranslation`, `RandomZoom`) combinadas com *Background Replacement* (substituir artificialmente o fundo das fotos de treino por imagens variadas e ruidosas) e realizar o *Fine-Tuning* das camadas finais da MobileNetV2. Isso força a rede neural a penalizar neurônios que olham para o fundo e a ajustar seus filtros convolucionais exclusivamente para a geometria das falanges e pontas dos dedos.
