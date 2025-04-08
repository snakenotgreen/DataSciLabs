# 📊 **DataScience Lab 2: Аналіз індексу VHI по регіонах України**

## 🎯 **Мета роботи**
> Ознайомитись із основними етапами обробки сирих даних: від збору та очищення до аналізу і візуалізації. Дослідити індекс **VHI** для регіонів України за даними **NOAA**.

---

## 📁 **Проєкт 1 — Python-скрипт для підготовки та аналізу даних**

### 🔧 **Функціонал**
- 🔽 **Автоматичне завантаження** `.csv` файлів із сайту NOAA для кожної області
- 🗂 Збереження файлів із **унікальними назвами** (за датою й часом)
- 🧹 **Очищення та об’єднання** даних у єдиний `DataFrame`
- 🔢 **Перетворення індексів регіонів** за українською абеткою
- 📊 **Аналітика**:
  - Отримання ряду **VHI** по регіону і року
  - Пошук **min / max / mean / median** значень
  - Вивід даних по кількох регіонах і роках
  - Виявлення **екстремальної посухи (VHI < 15)** в понад _N_% регіонів

### 🧠 **Основні функції**
- `csv_down(index)` – завантаження даних по ID області
- `data_in_frame(path)` – зчитування даних у DataFrame
- `convert_region_indices(df)` – трансформація індексів
- `vhi_year()`, `min_max()`, `vhi_row_year_region()` – аналітика
- `drought_stats()` – виявлення посух

### 📂 **Приклад структури CSV**
```
Year, Week, SMN, SMT, VCI, TCI, VHI, Region
1982, 1, ..., ..., ..., ..., ..., ...
```

---

## 🌐 **Проєкт 2 — Веб-додаток Streamlit: Інтерактивна аналітика**

### 🖥 **Інтерфейс**
- 🔘 Вибір показника: `VCI`, `TCI`, `VHI`
- 📍 Вибір області
- 📅 Слайдери **діапазону тижнів і років**
- 🔁 Кнопка **скидання фільтрів** до стандартних
- 📈 Вкладки:
  1. **Таблиця** — відфільтровані дані
  2. **Графік** — індекс по обраній області
  3. **Порівняння** — індекси для всіх областей з виділенням обраної

### ⚙️ **Технології**
- `streamlit`
- `pandas`
- `matplotlib`, `seaborn`

### ▶️ **Запуск додатку**
```bash
streamlit run lab3.py
```


