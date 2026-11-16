
import polars as pl

# 1. Чтение исходного Excel-файла 
df = pl.read_excel("Merged file.xlsx")  

# Добавляем колонку с индексом строки
df = df.with_row_index("row_idx")

start_indices = df.filter(pl.col("TY") == "JOUR")["row_idx"].to_list()
results = []

for i, start in enumerate(start_indices):
    end = start_indices[i + 1] if i + 1 < len(start_indices) else len(df)
    article_df = df[start:end]
    
    first_row = article_df[0]
    title = first_row["T1"]
    year = first_row["PY"]
    doi = first_row["DO"]
    abstract = first_row["AB"]
    
    # Собираем авторов и ключевые слова из всех строк статьи 
    authors = []
    keywords = []
    for row in article_df.rows(named=True):
        au = row.get("AU")
        if au and str(au).strip():
            authors.append(str(au).strip())
        kw = row.get("KW")
        if kw and str(kw).strip():
            for part in str(kw).split(','):
                part = part.strip()
                if part:
                    keywords.append(part)
    
    keywords = list(dict.fromkeys(keywords))
    authors = list(dict.fromkeys(authors))
    
    results.append({
        "Title": title,
        "Author": "; ".join(authors),
        "Year": year,
        "Key Words": "; ".join(keywords),
        "DOI": doi,
        "Abstract": abstract
    })
result_df = pl.DataFrame(results)
result_df.write_excel("exit.xlsx")
