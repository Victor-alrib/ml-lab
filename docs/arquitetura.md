# Arquitetura

## Visão geral

```
┌──────────────────────────────┐        ┌──────────────────────────────────────┐
│ Frontend (React + TS)        │  JSON  │ Backend (FastAPI)                     │
│  pages/  um arquivo por      │ ─────► │  api/routes.py                        │
│          módulo              │        │  services/                            │
│  components/Plot.tsx         │ ◄───── │    profiling · eda · preprocessing    │
│   (Plotly, cores do tema)    │        │    models · training · compare        │
│  lib/education.ts            │        │  core/storage.py (disco)              │
└──────────────────────────────┘        └──────────────────────────────────────┘
```

## Fluxo de um experimento

```
CSV ─► prepare()
        ├─ remove linhas sem alvo / (opcional) com ausentes
        ├─ codifica o alvo (classe positiva = 1)
        ├─ separa treino/teste  (aleatória | estratificada | temporal)
        └─ detect_leakage() — só no treino
                │
                ▼
Pipeline = [ColumnTransformer(num: clip → imputer → scaler | cat: imputer → encoder)
            → seleção de atributos → modelo]
                │
     ┌──────────┼─────────────────────────────┐
     ▼          ▼                             ▼
cross_validate  fit(treino) → predict(teste)  baseline (Dummy) nas mesmas condições
(pipeline       métricas de treino e teste
 reajustado     curvas ROC/PR, resíduos
 por fold)      importância, coeficientes
     │
     └─► probabilidades out-of-fold → explorador de limiar
```

O resultado é gravado como JSON (`storage/experiments/<id>.json`) e o pipeline ajustado como `storage/models/<id>.joblib`.

## Decisões

**Persistência em arquivos.** Um laboratório local não precisa de banco de dados; JSON é legível, versionável e fácil de inspecionar. Trocar por SQLite/PostgreSQL afeta apenas `core/storage.py`.

**Agregação no servidor.** Histogramas, quartis e correlações são calculados com Pandas/NumPy sobre todas as linhas; o navegador recebe só os agregados. Gráficos de dispersão e de previsões são amostrados para no máximo alguns milhares de pontos, mas as estatísticas exibidas (r, p-valor, reta) usam todos os pontos — a interface informa quando há amostragem.

**Hiperparâmetros declarados no backend.** `services/models.py` descreve tipo, faixa, padrão e ajuda de cada hiperparâmetro; o formulário do frontend é gerado a partir disso. Adicionar um modelo = registrar um `ModelSpec`.

**Explorador de limiar no cliente.** O backend envia pares (rótulo, probabilidade) reais; o navegador recalcula precision/recall/F1 para cada limiar. Nenhum valor é interpolado ou estimado.

**Compatibilidade com scikit-learn.** A partir da 1.8 o parâmetro `penalty` da `LogisticRegression` foi substituído por `l1_ratio`; `models.py` detecta a versão instalada e usa a API correspondente.

## Endpoints principais

| Método | Rota | Função |
|---|---|---|
| GET | `/api/datasets` | Lista datasets |
| POST | `/api/datasets/upload` | Envia CSV |
| POST | `/api/samples/{key}` | Carrega dataset de exemplo |
| GET | `/api/datasets/{id}/profile` | Tipos, estatísticas, ausentes, outliers |
| POST | `/api/datasets/{id}/rows` | Linhas paginadas, com filtro/busca/ordenação |
| POST | `/api/datasets/{id}/eda` | `histogram`, `boxplot`, `correlation`, `scatter`, `distribution` |
| POST | `/api/preprocess/preview` | Ajusta o pré-processamento no treino e mostra o que aprendeu |
| GET | `/api/models` | Catálogo de modelos e hiperparâmetros |
| POST | `/api/train` | Treina, avalia e salva um experimento |
| GET/PATCH/DELETE | `/api/experiments/{id}` | Consulta, renomeia, exclui |
| GET | `/api/experiments/{id}/model` | Baixa o pipeline `.joblib` |
| POST | `/api/experiments/compare` | Comparação e testes pareados |

A documentação OpenAPI completa fica em `/docs` com a API rodando.
