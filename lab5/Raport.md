Jak widzimy dla warst w pełni połączonych (całkiem bez warst splotowych) keras tuner znalazł najlepsze hiperparametry:
Best hyperparameters:
  num_layers: 3
  units_0: 480
  learning_rate: 0.001
  units_1: 512
  tuner/epochs: 4
  tuner/initial_epoch: 0
  tuner/bracket: 0
  tuner/round: 0
  units_2: 32

a tam gdzie użyliśmy warstw splotowych:

Jak widizmy dla 
Best hyperparameters:
  num_conv_layers: 1
  filters_0: 128
  dense_units: 256
  learning_rate: 0.0001
  filters_1: 160
  tuner/epochs: 4
  tuner/initial_epoch: 2
  tuner/bracket: 1
  tuner/round: 1
  tuner/trial_id: 0001
  filters_2: 64
.

### Ocena learning rate z tunera
Widzmy, że siec w pełni połączone nie potrzebują tak niskich kroków gradientu (wyższy learning rate) jak warstwy splotowe, jest to prawdopodobnie spowodowany tym, że warstwa w pełni połączona ma bardziej wrażliwe parametry jak filtry gdzie mała zmiana wag wpływa na wiele pikseli na raz, a w warswie w pełni połączonej nie mamy weight-sharing (np filtra 3x3 z 9 wagami i jedym biasem) tylko osobna wage dla kazdego neuronu i osobny bias, co jest znacznie mniej wrażliwe na zmiany.

### Liczenie Parametrow
Porownanie liczby parametrów
wiemy, ze mnist-fashion po zmienieniu na czarno biale obrazy ((28,28,10) -> (28,28,1)) ma 28*28*1 = 784 piksele.

Z uzyciem warstwy w pelni połaczonej mamy
y = xW + b = wejscia * wagi + biasy

- input_layer = 784 (dense) * 480 (dense1) + 480 = 376,320
- hidden_layer1 (dense) = 480 * 512 (dense2) + 512 = 245,632
- hidden_layer2 (dense) = 512 * 32 (dense3) + 32 = 16,416
- output_layer (dense) = 32 * 10 (output) + 10 = 330
- suma = input_layer + hidden_layer1 + hidden_layer2 + output_layer 649,818 parametrow

Z uzyciem z uzyciem warstwy splotowej

- input_layer (conv net) = 128 (filters) * (3 * 3 (kernel) * 1 (in_channels=1 bo mamy grayscale a nie rgb gdzize bysmy uzyli 3)) + 128 (bias) = 1,280

Mapa cech po konwolucji: 28 * 28 * 128 (dim_x * dim_y * filters) = 100,352
y = xW + b = wejscie * wagi + bias

- hidden_layer (dense) 100,352 * 256 (dense units) + 256 = 25,690,624
- output_layer (dense) 256 * 10 + 10 = 2,570

suma = input_layer + hidden_layer + output_layer = 25,694,218 parametrow

### Ocena Liczby parametrow razem z wynikami loss i accuracy
Jak widzimy z użyciem warswy splotowej mamy znacznie więcej parametrów niż z samymi warstwami w pełni polączonymi (co jest bardzo częste) więc traning był znacznie wolniejszy. Z użyciem warswy w pełni połączonej mamy mniej stabilny validation loss a training loss dochodzi do konwergencji nieco wolniej niż z użyciem warst splotowych. Uzycie warst splotowych jak widizmy poprawilo stabilnosc uczenia ale accuracy było tylko marginalnie wyższe co może nie być równe warstości potrzebnej mocy obliczeniowej jej uczenia w tym konkretnym przypadku z tym konkretnie zbiorem danych jakim jest mnist fashion.

# Ocena Macierzy pomyłek
Dla warstwy w pełni połączonej model najlepie przewidywał klasę 4 (najciemniejsza na diagonalnej) a najgorzej przewidywał klasę 3 (najaśniejsza na diagonalnej).
Ale model na ogół wyjątkowo często przewidywał, że coś jest klasą 4 (praktycznie każdą klasę) co podowoduje, że model jest lekko mówiąc bezużyteczny.

Z użyciem warswy konwolucyjnej model
najlepiej przewidywał klasę 0 (najczemniejsze na diagonalnej) i najgorzej klasy 4 i 6 (najaśniejsze na diagonalnej). Choć model często gdy widział klasę 4 mówił, że to klasa 0.