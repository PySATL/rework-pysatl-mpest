# Core

Предоставляет класс смеси распределений, а так же дескриптор, позволяющий реализовывать свои распределения. (Решил его вынести сюда а не в Distributions)


```{image} ../_static/core.png
:alt: Select Parameters
:width: 800px
:align: center
```

## Классы

### Parameter

Класс, реализующий протокол дескриптора Python для управления атрибутами-параметрами в классах распределений.

Этот класс решает две ключевые задачи:

1. **Валидация:** Гарантирует, что параметр всегда имеет корректное значение (например, `scale` для нормального распределения должен быть больше нуля).

2. **Фиксация:** Позволяет "замораживать" параметры, чтобы они не менялись в процессе оптимизации.



- **Атрибуты**

  - **\- invariant: Callable[[float], bool]**

    Функция-предикат, которая проверяет корректность присваиваемого значения. Должна возвращать `True`, если значение допустимо, и `False` в противном случае. По умолчанию принимает любое значение (`lambda x: True`).

  - **\- error_message: str**

    Текст ошибки, которая будет вызвана, если значение не прошло проверку инварианта.

  - **\- public_name: str**

    Публичное имя атрибута, как оно задано в классе-владельце (например, `loc` или `scale`). Это имя устанавливается автоматически при создании класса через метод `__set_name__`.

  - **\- private_name: str**

    Приватное имя, используемое для хранения значения внутри экземпляра класса-владельца (например, _loc или _scale). Использование отдельного приватного имени необходимо, чтобы избежать бесконечной рекурсии при вызове getattr и setattr внутри дескриптора.

- **Методы**

  - **\- \_\_set_name\_\_(self, owner, name)**

    Магический метод, который автоматически вызывается интерпретатором Python при  создании класса-владельца. Он "сообщает" дескриптору имя, под которым он был присвоен атрибуту класса (например, `name` будет равно `"scale"`).

  - **\- \_\_get\_\_(self, instance, owner) -> float**

    Метод для получения значения атрибута. Когда мы обращаемся к `distribution.scale`, вызывается этот метод, который возвращает значение из `distribution._scale`.

  - **\- \_\_set\_\_self, instance, value)**

    Метод для установки нового значения. Это ключевой метод, где происходит вся логика:

    1. Проверяет, не является ли этот параметр "зафиксированным" (не входит ли его `public_name` в множество `instance._fixed_params`).
    2. Если не зафиксирован, проверяет новое `value` с помощью функции `self.invariant`.
    3. Если обе проверки пройдены, сохраняет `value` в `instance` под приватным именем (`self.private_name`).

- **Альтернативы**

    В прошлой версии архитектуры параметры хранились в объекте распределения в виде списка и без имён. Такой подход приводил ко многим проблемам:

    - Внешний пользователь работал с параметрами как со списком, поэтому не мог знать о том, какой параметр закодирован в массиве под индексом 0 или под индексом 1, об этом знал только разработчик.

    - Логика преобразования параметров к внутреннему и внешнему виду лежала в самом объекте распределения, и проводилась с помощью специальных методов. Дескриптор решает эту проблему, инкапсулируя эту логику внутри себя.

    - Если хранить параметры в таком виде, то неудобно поддерживать заморозку и разморозку параметров, т.к. это происходит по индексам, а не по именам, и для конечного пользователя все так же является непонятным.



### MixtureModel

Класс смеси распределений.

- **Атрибуты**

  - **\+ n_components: int**

    Количество компонент в смеси.

  - **\+ components: list[Distribution]**

    Компоненты смеси.

  - **\+ weights: ndarray**

    Веса компонент.

  - **\- logits: ndarray**

    Логиты. Необходимы для численной стабильности и соблюдения инварианта для весов. Можно оптимизировать их напрямую.

- **Методы**

  - **\+ add_component(component: int, weight: float)**

    Добавляет новую компоненту и назначает ей выбранный вес. Остальные веса пересчитываются согласно их пропорциям.

  - **\+ remove_component(component_idx: int)**

    Удаляет компоненту из смеси по ее индексу. Остальные веса пересчитываются согласно их пропорциям так, чтобы соблюдался инвариант.

  - **+ pdf(X: ArrayLike): ndarray**

    Функция плотности.

  - **\+ lpdf(X: ArrayLike): ndarray**

    Логарифм функции плотности.

  - **\+ loglikelihood(X: ArrayLike): ndarray**

    Логарифм правдоподобия.

  - **\+ generate(size: int): ndarray**

    Сэмплирование выборки размера `size`.



## Различные диаграммы

### Пример взаимодействия ContinuousDistribution и дескриптора Parameter

```{mermaid}
sequenceDiagram
    participant User
    participant Distribution as "экземпляр<br>Exponential"
    participant Parameter as "дескриптор<br>Parameter ('rate')"
    participant Invariant as "lambda s: s > 0"

    User->>Distribution: dist.rate = 2.0
    activate Distribution

    Distribution->>Parameter: __set__(dist, 2.0)
    activate Parameter

    Parameter-->>Distribution: является ли 'rate' в dist._fixed_params?
    Distribution-->>Parameter: Нет

    Parameter->>Invariant: invariant(2.0)
    activate Invariant
    Invariant-->>Parameter: True
    deactivate Invariant

    Parameter->>Distribution: setattr(dist, "_rate", 2.0)
    note right of Parameter: Значение корректно и<br>параметр не зафиксирован.<br>Обновляем приватное поле.

    deactivate Parameter
    deactivate Distribution

    User->>Distribution: dist.rate = -1.0
    activate Distribution

    Distribution->>Parameter: __set__(dist, -1.0)
    activate Parameter

    Parameter-->>Distribution: является ли 'rate' в dist._fixed_params?
    Distribution-->>Parameter: Нет

    Parameter->>Invariant: invariant(-1.0)
    activate Invariant
    Invariant-->>Parameter: False
    deactivate Invariant

    Parameter-->>User: raise ValueError
    deactivate Parameter
    deactivate Distribution
```
