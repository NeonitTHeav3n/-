# -*- coding: utf-8 -*-
"""
Лабораторная работа 1. Работа с Polars.

Задание:
  - Скачать данные по ссылке:
    https://docs.google.com/spreadsheets/d/1bqhgCMkZyRzZAT7WO1SqCXc05ihQIRz-/edit?usp=sharing
  - Создать таблицу в Polars со столбцами:
      T1 (Заголовок), AU (Авторы - список), PY (Год),
      KW (Ключевые слова - список), DO (DOI), AB (Аннотация)
  - Авторов и ключевые слова записать в ячейку каждой статьи в виде списка.
  - Число строк, занимаемое каждым журналом, подсчитать с помощью пропущенных значений.
  - Удалить лишний столбец VL.
  - Сохранить результат в Excel-файл.

Примечание 1 (Polars):
  Исходный файл имеет формат Scopus-export:
    - TY = 'JOUR'  -> начало новой статьи
    - TY = null    -> continuation (дополнительные авторы / ключевые слова)
    - JO (журнал) заполнен только в JOUR-строке, в continuation - пропущен
  Поэтому для подсчёта строк по журналам используется forward-fill пропущенных JO.

Примечание 2 (Pandas):
  При выполнении с Pandas максимум = 8 баллов (см. исходный butebk).
"""

import polars as pl
import os

# ============================================================
# 1. Считывание исходных данных
# ============================================================
INPUT_FILE = "source_data.xlsx"
OUTPUT_FILE = "lab1_result.xlsx"

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА 1: РАБОТА С POLARS")
print("=" * 60)
print()

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Файл {INPUT_FILE} не найден. Скачайте данные со страницы задания."
    )

df = pl.read_excel(INPUT_FILE)
print(f"[1] Прочитан файл: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"    Столбцы: {df.columns}")
print()

# Посмотрим на уникальные значения столбца TY
ty_counts = df.group_by("TY").agg(pl.len())
print("[1a] Распределение по TY:")
for row in ty_counts.iter_rows(named=True):
    print(f"     TY={row['TY']!r}: {row['len']} строк")
print()

# ============================================================
# 2. Очистка данных
#    - удалить столбец VL (по заданию)
#    - удалить служебные строки TY='TY'
# ============================================================
df = df.drop("VL")
df = df.filter((pl.col("TY") == "JOUR") | pl.col("TY").is_null())

print(f"[2] После очистки: {df.shape[0]} строк")
print(f"    JOUR (статьи): {(df['TY'] == 'JOUR').sum()}")
print(f"    null (продолжения): {df['TY'].is_null().sum()}")
print()

# ============================================================
# 3. Подсчёт строк по журналам через пропущенные значения
#    (Примечание 1 к заданию)
# ============================================================
print("[3] Подсчёт строк по журналам (через пропущенные JO):")
print("    Журнал указан только в JOUR-строке (JO=null в continuation)")
print()

df_with_journal = df.with_columns(
    pl.col("JO").fill_null(strategy="forward").alias("JO_filled")
)

journal_stats = df_with_journal.group_by("JO_filled").agg(
    pl.len().alias("total_rows"),
    pl.col("TY").filter(pl.col("TY") == "JOUR").count().alias("articles"),
).sort("total_rows", descending=True)

print(f"    Всего журналов: {journal_stats.shape[0]}")
print()
print("    Топ-10 журналов:")
print(f"    {'Журнал':42} {'Строк':>7} {'Статей':>7}")
print("    " + "-" * 60)
for row in journal_stats.head(10).iter_rows(named=True):
    print(f"    {row['JO_filled']:40} {row['total_rows']:>7} {row['articles']:>7}")
print()

# ============================================================
# 4. Создание итоговой таблицы
#    Для каждой статьи собираем:
#      T1  - заголовок (из JOUR-строки)
#      AU  - список всех авторов (implode continuation)
#      PY  - год (из JOUR-строки)
#      KW  - список всех ключевых слов (implode continuation)
#      DO  - DOI (из JOUR-строки)
#      AB  - аннотация (из JOUR-строки)
# ============================================================
print("[4] Создание итоговой таблицы...")

# article_id = cumsum по индикатору "новая статья" (TY=='JOUR')
df = df.with_columns(
    pl.when(pl.col("TY") == "JOUR").then(1).otherwise(0).cum_sum()
      .alias("article_id")
)

result = df.group_by("article_id").agg(
    pl.col("T1").filter(pl.col("T1").is_not_null()).first().alias("T1"),
    pl.col("AU").filter(pl.col("AU").is_not_null()).implode().alias("AU"),
    pl.col("PY").filter(pl.col("PY").is_not_null()).first().alias("PY"),
    pl.col("KW").filter(pl.col("KW").is_not_null()).implode().alias("KW"),
    pl.col("DO").filter(pl.col("DO").is_not_null()).first().alias("DO"),
    pl.col("AB").filter(pl.col("AB").is_not_null()).first().alias("AB"),
).sort("PY").drop("article_id")

print(f"    Результат: {result.shape[0]} статей, {result.shape[1]} столбцов")
print()
print("    Столбцы и типы:")
for col in result.columns:
    print(f"      {col:6} -> {result[col].dtype}")
print()

# ============================================================
# 5. Преобразование списков (List[String]) в строки для Excel
#    В Excel значения в ячейке записываются через "; ":
#      "Author1; Author2; Author3"
# ============================================================
print("[5] Преобразование списков в строки (для Excel):")

def list_to_string(s: pl.Series) -> list:
    """Конвертирует polars List[String] в Python-список строк,
    где элементы соединены '; '."""
    out = []
    for val in s.to_list():
        if val is None:
            out.append("")
        elif isinstance(val, list):
            out.append("; ".join(str(x) for x in val))
        else:
            out.append(str(val))
    return out

result_excel = result.with_columns([
    pl.Series("AU_excel", list_to_string(result["AU"]), dtype=pl.String),
    pl.Series("KW_excel", list_to_string(result["KW"]), dtype=pl.String),
])

print(f"    Пример (статья №1):")
print(f"      AU: {result_excel.row(0, named=True)['AU_excel'][:80]}...")
print(f"      KW: {result_excel.row(0, named=True)['KW_excel'][:80]}...")
print()

# ============================================================
# 6. Сохранение результата
# ============================================================
print("[6] Сохранение результата в Excel...")

final = result_excel.select([
    "T1",
    pl.col("AU_excel").alias("AU"),
    "PY",
    pl.col("KW_excel").alias("KW"),
    "DO",
    "AB",
])

final.write_excel(OUTPUT_FILE, worksheet="Sheet1")
print(f"    Сохранено: {OUTPUT_FILE}")
print()

# Проверка
check = pl.read_excel(OUTPUT_FILE)
print("[6a] Проверка сохраненного файла:")
print(f"    Размер: {check.shape[0]} строк x {check.shape[1]} столбцов")
print(f"    Столбцы: {check.columns}")
print()

print("=" * 60)
print("РЕЗУЛЬТАТЫ ЛАБОРАТОРНОЙ РАБОТЫ 1")
print("=" * 60)
print()
print(f"  Выходной файл: {OUTPUT_FILE}")
print(f"  Количество статей: {result.shape[0]}")
print()
print("  Столбцы:")
print("    T1 - Заголовок статьи")
print("    AU - Авторы (спиsk, значения разделены '; ' в ячейке Excel)")
print("    PY - Год публикации")
print("    KW - Ключевые слова (список, значения разделены '; ' в ячейке Excel)")
print("    DO - DOI")
print("    AB - Аннотация")
print()
print("  Статистика:")
print(f"    Среднее авторов на статью: {result['AU'].list.len().mean():.2f}")
print(f"    Среднее ключевых слов на статью: {result['KW'].list.len().mean():.2f}")
print(f"    Годы: {result['PY'].min()} - {result['PY'].max()}")
print(f"    Всего журналов в выборке: {journal_stats.shape[0]}")
print()
print("  Интересные факты:")
max_auth = result['AU'].list.len().max()
max_kw = result['KW'].list.len().max()
r_auth = result.filter(pl.col('AU').list.len() == max_auth).row(0, named=True)
r_kw = result.filter(pl.col('KW').list.len() == max_kw).row(0, named=True)
print(f"    Максимум авторов ({max_auth}): {r_auth['T1'][:55]}...")
print(f"    Максимум ключевых слов ({max_kw}): {r_kw['T1'][:55]}...")
print()
print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА 1 ВЫПОЛНЕНА УСПЕШНО")
print("=" * 60)
