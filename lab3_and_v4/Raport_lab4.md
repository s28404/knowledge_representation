### Porownanie modeli klasycznych z ztuningowanym

Po treningu bazowych modeli v1 i v2, zrobilismy tuning hiperparametrow dla modelu v2 przy uzyciu Keras Tuner z algorytmem RandomSearch. Testowalismy 20 roznych kombinacji hiperparametrow, maksymalizujac dokladnosc walidacyjna (val_accuracy).

**Hiperparametry testowane:**
- Liczba neuronow w pierwszej warstwie ukrytej (units1): 16, 32, 64
- Funkcja aktywacji: relu, tanh, elu
- Wspolczynnik dropout: 0.1 do 0.5 (krok 0.1)
- Liczba neuronow w drugiej warstwie ukrytej (units2): 16, 32, 64
- Wspolczynnik uczenia (learning_rate): od 0.0001 do 0.01 (skala logarytmiczna)

**Najlepsze hiperparametry znalezione przez tuner:**
- units1: 64
- activation: tanh
- dropout: 0.2
- units2: 16
- learning_rate: ~0.0013

**Porownanie wynikow:**
- Model v1 (bazowy): Dokladnosc testowa 94.44%, trening byl stabilny, ale moze byc overfitting bo dalismy za duzo parametrow (bez dropout).
- Model v2 (bazowy): Dokladnosc testowa 97.22%, trening mniej stabilny z powodu dropout, ale lepsza generalizacja.
- Model ztuningowany (tuned v2): Dokladnosc testowa 97.22%, podobne do bazowego v2, ale z optymalnymi hiperparametrami, co moze poprawic stabilnosc i efektywnosc.

Tuning potwierdzil, ze model v2 z odpowiednio dobranymi parametrami osiaga wysoka dokladnosc, przewyzszajac bazowy v1. Ztuningowany model ma mniej neuronow w drugiej warstwie (16 vs 32), co zmniejsza zlozonosc, oraz optymalny learning_rate, co przyspiesza konwergencje. Dropout 0.2 okazal sie wystarczajacy do zapobiegania overfittingowi bez nadmiernego oslabiania modelu.

**Macierz pomylek dla ztuningowanego modelu:**
[[15  0  0]
 [ 0 13  1]
 [ 0  0  7]]

Model prawie doskonale klasyfikuje klasy 0 i 2, z jedna pomylka w klasie 1.

**Podsumowanie modelu ztuningowanego:**
Model sekwencyjny z warstwami: Normalization, Dense(64, tanh), Dropout(0.2), Dense(16, tanh), Dropout(0.2), Dense(3, softmax). Laczenie 5,990 parametrow.