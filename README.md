# Reconhecimento de Sinais de Libras com Deep Learning (MobileNetV2)

### 🎓 Informações
* **Instituição:** Centro Universitário Campo Real
* **Curso:** Engenharia de Software
* **Disciplina:** 2026/2 TA - MACHINE LEARNING E DEEP LEARNING
* **Autores:** Gabriel Beledeli Hul; Nicolas Miguel; Vinicius Buskievicz; Alisson Eraldo

---

Este projeto implementa um pipeline completo de Visão Computacional e Deep Learning para coletar, treinar e reconhecer em tempo real **10 sinais estáticos de Libras** (Alfabeto Dactilológico: `A, B, C, D, I, L, O, V, W, Y`) mais a classe `nada`, utilizando a técnica de **Transfer Learning** sobre a arquitetura **MobileNetV2**.

---

## 🚀 Guia Rápido de Execução

### 1. Configurar o Ambiente Virtual (Python 3.11)
```powershell
# Criar o ambiente virtual utilizando Python 3.11
py -3.11 -m venv venv

# Ativar o ambiente virtual
.\venv\Scripts\Activate.ps1

# Instalar dependências
pip install -r requirements.txt
```

### 2. Fluxo dos Scripts
1. **`python collect.py`**: Coleta as fotos pela webcam (150 no fundo claro + 150 no fundo escuro para cada sinal).
2. **`python train.py`**: Treina o modelo, gera os gráficos de desempenho (`curvas.png`) e compila automaticamente o PDF de entrega (`dataset_imagens.pdf`).
3. **`python play.py`**: Executa a inferência contínua na webcam em tempo real.

---

## 🔄 Relatório de Mudanças: Original (`pedra_papel_tesoura`) vs (`sinais_de_libras`)

O projeto original (`pedra_papel_tesoura`) foi desenvolvido para o jogo de "Pedra, Papel e Tesoura" (4 classes simples). Para atender aos requisitos acadêmicos da atividade de Libras, foram implementadas as seguintes alterações estruturais, metodológicas e visuais em cada arquivo:

---

### 1. Modificações no `collect.py`

| Aspecto | Original (`pedra_papel_tesoura`) | Novo (`sinais_de_libras`) |
| :--- | :--- | :--- |
| **Classes** | 4 classes: `pedra`, `papel`, `tesoura`, `nada`. | **11 classes**: 10 letras de Libras (`A, B, C, D, I, L, O, V, W, Y`) + `nada`. |
| **Mapeamento de Teclas** | Teclas `1, 2, 3, 4` (ou `p, a, t, n`). | Mapeamento expandido para teclas numéricas **`1` a `9`, `0`**, letras mnemônicas e **`n`/`espaço`** para `nada`. |
| **Fluxo de Coleta** | Coleta contínua e sem distinção de iluminação ou fundo. | **Assistente Físico em 2 Fases (150 Claro + 150 Escuro)**: meta de 300 fotos dividida em duas etapas de 150. |
| **Pausa Automática Inteligente** | Não existia (gravava direto até o limite). | Ao atingir exatamente a foto **150** (Fundo Claro), a gravação **pausa sozinha** e emite um alerta na tela orientando o usuário a apagar a luz/mudar para Fundo Escuro antes de retomar. |
| **Interface Visual (HUD)** | Texto simples no canto superior esquerdo que podia sobrepor o quadrado da mão (ROI). | Painel lateral com **fundo semi-transparente** indicando a fase atual de cada classe (`Claro`, `Escuro`, `OK`) sem obstruir a visão da webcam. |

---

### 2. Modificações no `train.py`

| Aspecto | Original (`pedra_papel_tesoura`) | Novo (`sinais_de_libras`) |
| :--- | :--- | :--- |
| **Estratégia de Split (Treino vs Validação)** | **Split ingênuo no final do tempo**:<br>`files[:cut]` (primeiros 80%) para treino e `files[cut:]` (últimos 20%) para validação. | **Split Balanceado Estratificado no Tempo**:<br>Divide as fotos de cada classe em duas metades temporais (Metade 1 = Claro, Metade 2 = Escuro) e extrai **80% de treino e 20% de validação de cada metade separadamente**. |
| **Resolução do Viés de Validação** | Se o aluno gravasse claro primeiro e escuro depois, a validação teria **100% de fotos escuras e 0% de fotos claras**, gerando um modelo com avaliação distorcida. | Com a nova divisão, tanto o treino quanto a validação possuem **exatamente 50% de fotos claras e 50% de fotos escuras**. |
| **Arquitetura e Treino em 2 Etapas (Fine-Tuning)** | Base MobileNetV2 100% congelada com apenas `GlobalAveragePooling2D` e `Dense`. | Combinação de `GlobalAveragePooling2D` + `GlobalMaxPooling2D` + `LayerNormalization`, `RandomTranslation`, e **Treino em 2 Etapas** (Etapa 1 com base congelada + Etapa 2 de *Fine-Tuning* nas últimas 20 camadas com `EarlyStopping(restore_best_weights=True)`). |
| **Geração Automática do PDF de Entrega** | Não existia. | Adicionada ao final a função `gerar_pdf_dataset()`, que compila todas as fotos da pasta `data/` em um documento A4 com miniaturas organizadas por classe (`dataset_imagens.pdf`), atendendo à regra de entrega da atividade. |
| **Tratamento de Exceções** | Não verificava se a pasta `data/` ou as classes estavam vazias antes de rodar o treino. | Adicionadas validações que avisam no terminal de forma clara caso falte coletar dados. |

---

### 3. Modificações no `play.py`

| Aspecto | Original (`pedra_papel_tesoura`) | Novo (`sinais_de_libras`) |
| :--- | :--- | :--- |
| **Propósito do Script** | Jogo interativo contra a máquina (com placar, contagem regressiva para a jogada, regras de quem ganha de quem (`BEATS`) e escolha aleatória do computador). | **Reconhecedor contínuo de sinais de Libras em tempo real**, focado na avaliação e demonstração do modelo. |
| **Feedback Visual do Sinal** | Mostrava apenas o resultado da rodada do jogo e barras das 4 classes. | Banner destacado com: **nome do sinal**, **descrição anatômica do gesto**, **percentual de certeza**, e **caixa adaptativa** (muda para verde em alta confiança). |
| **Distribuição de Probabilidades** | Barras horizontais simples para 4 classes. | Painel lateral estilizado exibindo a probabilidade em tempo real de todas as **11 classes**, destacando a classe vencedora. |
| **Recursos Extras** | Apenas tecla espaço para jogar e `q` para sair. | Adicionado **contador de FPS** e tecla **`s`** para salvar capturas instantâneas das predições para relatórios/apresentações. |

---

## 📊 Estrutura de Arquivos Mantida no Git

Graças ao arquivo `.gitignore`, o repositório mantém apenas o código-fonte essencial e documentação:
* `collect.py` - Coleta assistida de imagens via webcam.
* `train.py` - Treinamento com MobileNetV2 e gerador de PDF.
* `play.py` - Demonstração em tempo real.
* `requirements.txt` - Dependências do Python.
* `README.md` - Documentação completa e relatório de mudanças.
* `RESPOSTAS_PARTE5.md` - Respostas conceituais e teóricas da Parte 5 da atividade.
* `.gitignore` - Ignora as pastas `data/`, `venv/`, caches e arquivos binários gerados (`.keras`, `.pdf`, `.png`).

---