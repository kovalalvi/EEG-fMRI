# EEG-fMRI
**SOME NEW CHANGED**


There is a project of prediction subcortical BOLD signal from concurrent EEG using machine learning

Inspired by https://github.com/laughinghuman/EEG_fMRI

Previous 
- Make_prediction.ipynb - It is the main file. You download data, generate dataset, train Ridge loss and make prediction on test data


**ALL_AGGREGATE.ipynb** I update the main file 

**SOME NEW CHANGED**

- Реализовал предварительную обработку данных
    - Убрал данные вначале и в конце эксперимента. В них было много шума и они мешали нормализации.
    - Интерполяция RoI fMRI signals
    - Нормализация
    - В процессе. Удаление шумов от сети 50, 100 Гц  → in progress

- Реализовал скрипт для генерации датасета.
    - Можно работать как с сырыми EEG данными так и с spectrogram.
    - Генерация Mel spectrogram происходит "на лету" в процессе обращения к к классу по индексу.
- Написал скрипт для обучения моделей pytorch.
    - Отслеживать прогресс обучения можно в tensorboard
        - Train loss
        - Val loss
        - Visualize prediction on validation data.
    - Веса модели сохраняются в процессе обучения
    - Добавил возможность использовать различные метрики для отслеживания прогресса( использую L2 difference and Pearson correlation)
- Реализация модели.
    - Реализовал простую сверточную сеть с полно связным слоем для тестирования.
