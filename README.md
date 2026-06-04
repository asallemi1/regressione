# Vehicle MPG Prediction API

API REST sviluppata in **Python** con **Flask** per l'analisi esplorativa di dataset, la pulizia dei dati e la previsione del consumo di carburante (MPG) di un veicolo tramite modelli di regressione e riduzione della dimensionalità mediante PCA.

## Funzionalità

Il progetto offre:

* Analisi dei valori mancanti (NA)
* Identificazione e rimozione degli outlier tramite metodo IQR
* Visualizzazione delle distribuzioni tramite istogrammi
* Visualizzazione degli outlier tramite boxplot
* Analisi delle correlazioni tra variabili numeriche
* PCA (Principal Component Analysis)
* Test di normalità di Jarque-Bera
* Addestramento di modelli di regressione:

  * Linear Regression
  * Ridge Regression
  * Lasso Regression
  * Elastic Net
* Predizione del valore MPG di un nuovo veicolo

---

## Dataset

Il dataset viene recuperato automaticamente dalla repository UCI Machine Learning tramite la libreria `ucimlrepo`.

Dataset utilizzato:

* **Auto MPG Dataset** (UCI Repository)
* ID Dataset: `9`

Variabili utilizzate:

| Variabile    | Descrizione                    |
| ------------ | ------------------------------ |
| displacement | Cilindrata                     |
| cylinders    | Numero cilindri                |
| horsepower   | Potenza                        |
| weight       | Peso del veicolo               |
| acceleration | Accelerazione                  |
| model_year   | Anno modello                   |
| origin       | Area geografica di provenienza |

Target:

* `mpg` (Miles Per Gallon)

---

## Struttura del Progetto

```text
exploration_regression/
│
├── main.py
├── Dockerfile
├── requirements.txt
│
└── it/
    └── akron/
        ├── data_explorer.py
        └── regressione.py
```

### Descrizione dei moduli

#### main.py

Espone le API REST e coordina le operazioni di:

* pulizia del dataset
* analisi esplorativa
* addestramento dei modelli
* generazione delle predizioni

#### data_explorer.py

Contiene la classe `PuliziaDataset` responsabile di:

* gestione dei valori mancanti
* eliminazione dei record incompleti
* individuazione degli outlier
* grafico PCA 
* generazione di grafici
* matrice di correlazione
* test di normalità

#### regressione.py

Contiene la classe `Regressione` che implementa:

* train/test split
* standardizzazione dei dati
* PCA a 3 componenti
* addestramento dei modelli
* valutazione delle performance
* nuove predizioni

---

## Installazione

### Installare le dipendenze

```bash
pip install -r requirements.txt
```

---

## Avvio Applicazione

```bash
python main.py
```

L'applicazione sarà disponibile all'indirizzo:

```text
http://localhost:5000
```

---

## API Endpoints

### 1. Predizione nuovo veicolo

**GET/POST**

```http
/new_veicle?metodo=elasticnet
```

Metodi disponibili:

* elasticnet
* lineare
* ridge
* lasso

Body JSON:

```json
{
  "displacement": 140,
  "cylinders": 4,
  "horsepower": 90,
  "weight": 2400,
  "acceleration": 15.5,
  "model_year": 80,
  "origin": 1
}
```

Risposta:

```json
{
  "prediction": 31.45
}
```

---

### 2. Metriche del modello

```http
/error?metodo=ridge
```

Risposta:

```json
{
  "mse": 8.32,
  "mae": 2.11,
  "rmse": 2.88,
  "r2": 0.84
}
```

---

### 3. Analisi valori mancanti

```http
/na
```

Restituisce:

* numero di valori mancanti
* percentuale di missing values

---

### 4. PCA

```http
/pca
```

Restituisce un'immagine PNG contenente la varianza cumulativa spiegata dalle componenti principali.

---

### 5. Istogrammi

```http
/hist
```

Restituisce un'immagine PNG con la distribuzione delle variabili numeriche.

---

### 6. Boxplot

```http
/boxplot
```

Restituisce un'immagine PNG con i boxplot delle variabili numeriche.

---

### 7. Matrice di correlazione

```http
/corr
```

Risposta:

```json
{
  "horsepower": {
    ...
  }
}
```

---

### 8. Test di Jarque-Bera

```http
/testJB
```

Risposta:

```json
{
  "horsepower": {
    "stat": 15.32,
    "p_value": 0.001,
    "normale": false
  }
}
```

---

## Pipeline di Machine Learning

1. Recupero dataset UCI
2. Sostituzione valori mancanti anomali
3. Eliminazione record incompleti
4. Rimozione outlier (IQR)
5. Train/Test Split
6. Standardizzazione delle feature
7. PCA (3 componenti)
8. Addestramento modello di regressione
9. Valutazione tramite:

   * MSE
   * MAE
   * RMSE
   * R²
10. Predizione su nuovi dati

---

## Tecnologie Utilizzate

* Python
* Flask
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* SciPy
* UCI ML Repository
* Docker

---

## Docker

Build immagine:

```bash
docker build -t api_regressione .
```

Avvio container:

```bash
docker run -p 5000:5000 api_regressione
```
