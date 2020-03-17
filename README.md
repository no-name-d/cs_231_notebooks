# Задание для курса "Свёрточные нейроныне сети для визуального распознавания".

## Настройка
Для запуска проектов можно использовать виртуальную среду.

**1. Anaconda:**

Для настройки среды выполните команду:

`conda create -n cs231n python=3.7 anaconda`

Затем активируйте среду и подключитесь к ней:

`source activate cs231n`

Чтобы выйти, просто закройте окно или введите:

`source deactivate cs231n`

**2. Python virtualenv:**

Второй вариант - использовать среду virtualenv. Для настройки используйте следующие команды:

`cd assignment1`
`sudo pip install virtualenv`    
`virtualenv -p python3 .env`       
`source .env/bin/activate`         
`pip install -r requirements.txt`

Чтобы закрыть среду, введите `deactivate`.

## Загрузка данных
Запустите следующую команду из директории `assignment1`, чтобы загрузить датасет CIFAR-10, необходимый для выполнения заданий:

`cd cs231n/datasets
./get_datasets.sh`

## Запуск Jupyter Notebook
Вы можете использовать Jupyter Notebook и Jupyter Lab для запуска .ipynb-файлов.

https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html

## Выполнение заданий
**1: Классификатор ближайшего соседа (kNN)**

Откройте файл **knn.ipynb** и выполните указанные в нём действия для реализации knn-классификатора.

**2: Метод опорных векторов (SVM)**

Задание находится в файле **svm.ipynb**.

**3: Классификатор Softmax**

Задание находится в файле **softmax.ipynb**.

**4: Двухслойная нейросеть**

Задание находится в файле **two_layer_net.ipynb**.

**5: Обучение признакам изображения**

Задание находится в файле **features.ipynb**.