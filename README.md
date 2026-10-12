# Hyperspectral Object Tracking

Параметрично-ефективний трекінг об'єктів у гіперспектральному відео для HOT2026. Система отримує гіперспектральний куб і BBox цілі на першому кадрі, а повертає BBox для наступних кадрів.

Повний [пропоузал проєкту](proposal_Kolodchak_Nuchuporyk_Moroz.pdf).

## Підхід

Поточний план: використати заморожені ViPT backbone і prediction head, адаптувавши гіперспектральний вхід легкими модулями, що навчаються.

```mermaid
flowchart LR
	A[Гіперспектральні кадри VIS / NIR / RedNIR] --> B[X2Cube та Dataset]
	B --> C[Спектральний проєктор: C каналів до 3]
	B --> D[CNN Prompt-Adapter]
	C --> E[Заморожений ViPT Backbone]
	D --> E
	E --> F[Заморожена BBox Head]
	F --> G[BBox, AUC, DP@20, FPS]
```

Мета пропоузалу: trainable-модулі мають займати менш ніж 3% параметрів. Оцінювання планується окремо для VIS (16 каналів), NIR (25) і RedNIR (15), з порівнянням зі статичною рамкою, false-color SAMURAI та доступними HSI-трекерами.

## Дані та ваги

Використовується HOT2026. Датасет завантажується за правилами змагання; сирі дані, проміжні куби й ваги моделей не додаються до Git. Інструкції та структура каталогів описані в [data/README.md](data/README.md) і [models/README.md](models/README.md).

## Налаштування

Створіть середовище й активуйте його. Потім установіть у нього збірку PyTorch, сумісну з вашою системою та CUDA, за інструкціями на [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
make setup
source .venv/bin/activate
python -m pip install torch  # choose the matching command from pytorch.org for CUDA support
pytest
```

Конфігурація за замовчуванням: [config/default.yaml](config/default.yaml). Навчання, конвертація HOT2026 та повний evaluation pipeline ще не реалізовані; поточний репозиторій готує відтворювану структуру для цих компонентів.

## Структура репозиторію

```text
config/                  Конфігурація даних, моделі, навчання й оцінювання
data/                    Локальні raw/interim/processed дані, не для Git
docs/                    Пропоузал, постер, слайди та командні правила
models/                  Локальні checkpoint-и, не для Git
notebooks/               Дослідження даних та аналіз помилок
results/                 Метрики, передбачення й візуалізації
scripts/                 Команди запуску дослідів
src/hsi_tracking/        Python-пакет: дані, модель, тренування, оцінювання
tests/                   Автоматизовані тести
```

## Команда

| Учасник | Напрямок за пропоузалом |
|---|---|
| Ничипорук Дарина | X2Cube, Dataset/DataLoader, інтеграція BBox prediction head |
| Колодчак Богдан | Спектральний проєктор, ViPT, CNN Prompt-Adapter, бейзлайни |
| Мороз Тетяна | Метрики AUC/DP@20, аналіз, CSV та відеовізуалізація |

## Розробка

```bash
ruff check .
ruff format --check .
pytest
```

Поточний стан і план експериментів ведуться в [results/experiments.md](results/experiments.md). Правила внеску: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).
