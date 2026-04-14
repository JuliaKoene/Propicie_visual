# Propicie: Avaliação Automatizada da Aptidão Física

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-v0.8+-green.svg)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Um sistema inovador para a avaliação automatizada de testes de aptidão física da Bateria de Fullerton, projetado para apoiar o envelhecimento ativo. Este projeto utiliza visão computacional com um sensor Kinect V2 e a biblioteca MediaPipe Holistic do Google para fornecer medições precisas e em tempo real.

## 📋 Sobre o Projeto

Este projeto, desenvolvido como parte da colaboração **PROPICIE - IPBEJA & IFSC** e contribuindo para a iniciativa **CAPACITA**, visa automatizar a avaliação da aptidão física de idosos. Ao automatizar os testes da Bateria de Testes Funcionais de Fullerton, podemos coletar dados objetivos sobre flexibilidade e força, o que é crucial para monitorar o declínio físico e promover programas de envelhecimento ativo personalizados.

O sistema foca em duas avaliações principais:
* **Teste de Sentar e Alcançar** (`Sentado e alcançar os pés com as mãos`): Mede a flexibilidade dos membros inferiores.
* **Teste de "Coçar as Costas"** (`Alcançar as mãos atrás das costas`): Mede a flexibilidade dos membros superiores (ombros).

O núcleo do projeto é uma aplicação em Python que utiliza um sensor Kinect V2 para capturar os movimentos do usuário e o framework MediaPipe Holistic para realizar a detecção de marcos corporais em tempo real. Essa abordagem permite o cálculo preciso dos ângulos corporais para validação da postura e das distâncias-chave para a pontuação dos testes.

### 📊 Principais Descobertas
A pesquisa conduzida por Artem Bukhantsev e aprofundada por mim concluiu que:
* A implementação com **MediaPipe** demonstrou uma precisão superior para o teste de Sentar e Alcançar, com um Erro Médio Absoluto (MAE) de aproximadamente **2.25 cm**.
* Esta abordagem foi significativamente mais precisa do que uma implementação nativa com PyKinect, que apresentou um MAE de 8.65 cm, devido a desafios como a instabilidade do esqueleto virtual ("jittering").
* O teste de "Coçar as Costas" provou ser um desafio para a visão computacional devido à oclusão de membros e à orientação do usuário de costas para a câmera.

## ✨ Funcionalidades

* **Avaliação em Tempo Real**: Análise automatizada dos exercícios "Sentar e Alcançar" e "Coçar as Costas".
* **Rastreamento de Alta Precisão**: Utiliza o MediaPipe Holistic para um rastreamento robusto e em tempo real de 33 marcos de pose, além de marcos detalhados das mãos.
* **Validação de Postura**: Calcula ângulos articulares (joelho, quadril, cotovelo) para garantir que o usuário esteja executando o exercício corretamente antes de realizar a medição.
* **Cadastro de Usuário**: Uma interface simples para registrar os dados do participante (idade, altura, peso, gênero) antes de iniciar os testes.
* **Registro de Dados**: Salva automaticamente os resultados dos testes, incluindo a distância calculada, a distância real (para validação) e o erro de medição, em arquivos Excel (`.xlsx`) para análise posterior.
* **Análise Estatística**: Inclui scripts em Python para analisar os dados coletados e calcular estatísticas-chave sobre o erro de medição.
* **Feedback em Tempo Real**: Fornece visualizações na tela do esqueleto, métricas-chave e instruções para guiar o usuário.

## 🛠️ Como Funciona

O sistema segue um fluxo de trabalho claro para cada avaliação:
1.  **Cadastro do Usuário**: O usuário insere seus dados demográficos.
2.  **Captura de Vídeo**: Um Kinect V2 captura o feed de vídeo do usuário.
3.  **Detecção de Marcos**: O vídeo é processado quadro a quadro. O MediaPipe Holistic detecta os marcos do corpo, mãos e face do usuário.
4.  **Calibração e Verificação da Postura**:
    * Para o teste de **Sentar e Alcançar**, o sistema valida a postura verificando se os ângulos do joelho, quadril e cotovelo estão dentro de limites pré-definidos (por exemplo, o joelho deve estar estendido). Uma vez que o usuário mantém uma pose de calibração válida, a posição do pé é fixada como referência.
    * Para o teste de **Coçar as Costas**, o sistema aguarda o usuário manter uma pose estável com as mãos atrás das costas.
5.  **Medição da Distância**: A distância euclidiana entre os marcos-chave (por exemplo, pontas dos dedos até a posição calibrada do pé, ou pontas dos dedos de uma mão para a outra) é calculada em pixels e convertida para centímetros. Um fator de correção de erro, derivado de testes empíricos, é aplicado para aumentar a precisão.
6.  **Exibição e Registro dos Resultados**: A distância final calculada é exibida na tela, e os resultados completos são salvos em um arquivo de log e em uma planilha Excel para o grupo de usuários.

## 🚀 Tecnologias Utilizadas

* **Linguagem**: **Python 3.8+**
* **Visão Computacional**: **OpenCV**, **MediaPipe Holistic**
* **Hardware**: **Microsoft Kinect for Windows v2**
* **Wrapper do SDK do Kinect**: **PyKinect2**
* **Manipulação e Análise de Dados**: **Pandas**, **NumPy**
* **Orquestração**: Os scripts podem ser executados diretamente com Python (`runner.py`) ou através de um **Runner em C# .NET** (`CsRunner/`).

## ⚙️ Configuração e Instalação

Para executar este projeto, siga os passos abaixo.

### Pré-requisitos
* Um computador com **Windows 10/11** (necessário para o SDK do Kinect).
* Um sensor **Microsoft Kinect v2** com seu respectivo adaptador de energia e cabo USB 3.0.
* Uma porta **USB 3.0** livre.
* **Python 3.8** (a distribuição Anaconda é recomendada).

### Passos de Instalação

1.  **Instale o SDK do Kinect para Windows 2.0**:
    * Baixe e instale o SDK do site oficial da Microsoft: [Kinect for Windows SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561).
    * Conecte seu sensor Kinect ao PC via USB 3.0 e a uma fonte de energia. Verifique se ele é reconhecido no Gerenciador de Dispositivos.

2.  **Configure o Ambiente Python**:
    * É altamente recomendável usar um ambiente virtual. Com o Anaconda, você pode criar um com:
        ```bash
        conda create -n propicie_env python=3.8
        conda activate propicie_env
        ```

3.  **Instale as Bibliotecas Necessárias**:
    * Instale as dependências principais usando pip:
        ```bash
        pip install opencv-python mediapipe pandas numpy openpyxl
        ```

4.  **Instale o PyKinect2**:
    * `PyKinect2` requer uma instalação manual. Clone o repositório oficial e execute o script de setup.
        ```bash
        git clone [https://github.com/Kinect/PyKinect2.git](https://github.com/Kinect/PyKinect2.git)
        cd PyKinect2
        python setup.py install
        ```
    * Se encontrar problemas, pode ser necessário instalar o `comtypes`.

## ▶️ Uso

Após a conclusão da configuração, você pode executar as avaliações.

### Executando a Suíte de Testes Completa
Você pode executar os testes de Sentar e Alcançar e de Coçar as Costas sequencialmente usando o script de execução fornecido.

```bash
python runner.py
```

### Executando Testes Individuais
Você também pode executar cada script de teste individualmente:

* **Para o Teste de Sentar e Alcançar**:
    ```bash
    python ./Sit-and-Reach/sit_and_reach_holistic_2.py
    ```
* **Para o Teste de Coçar as Costas**:
    ```bash
    python ./Back-Scratch/back_scratch.py
    ```

### O Processo
1.  Quando um script é iniciado, uma janela aparecerá solicitando as informações do usuário (Idade, Altura, Peso, Gênero). Preencha os campos e pressione `Enter`.
2.  Em seguida, uma janela solicitará a distância real medida. Isso é usado para validação e cálculo de erro. Insira o valor e pressione `Enter`.
3.  A janela principal da aplicação será aberta, mostrando o feed da câmera do Kinect com a sobreposição do esqueleto do MediaPipe.
4.  Siga as instruções na tela para se posicionar corretamente.
5.  O sistema detectará automaticamente quando você estiver na postura correta, manterá a pose e, em seguida, calculará o resultado.
6.  O resultado será exibido, e você será solicitado a continuar (`c`) ou sair (`q`).

## 📁 Estrutura do Projeto

```
.
├── /analises/              # Scripts e resultados para análise estatística dos dados.
├── /Back-Scratch/          # Contém o script Python para o teste de Coçar as Costas.
├── /CsRunner/              # Um projeto em C# .NET para executar os scripts Python.
├── /relatorios/            # Relatórios de progresso e finais detalhados.
├── /Sit-and-Reach/         # Contém scripts Python para o teste de Sentar e Alcançar.
├── /tabelas_testes/        # Planilhas de dados de teste.
├── /tabelas_utentes/       # Planilhas com dados coletados dos testes com usuários.
├── .gitignore              # Especifica arquivos a serem ignorados pelo Git.
├── runner.py               # Um script Python simples para executar todos os testes.
└── README.md               # Este arquivo.
```


## 🙏 Agradecimentos

* Este trabalho faz parte de uma colaboração de pesquisa entre o **Instituto Politécnico de Beja (IPBeja)** e o **Instituto Federal de Santa Catarina (IFSC)**.
* Este projeto contribui para o projeto mais amplo **CAPACITA**, que visa desenvolver ferramentas digitais para avaliar e melhorar as capacidades físicas da população idosa.
