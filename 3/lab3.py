import time
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import optuna
import shap
import duckdb
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error


def main():    

    lazy_df = pl.scan_csv('OnlineNewsPopularity.csv').with_columns([
        pl.col(' shares').str.strip_chars().cast(pl.Float64).log1p().alias('log_shares'),
        pl.col('url').str.extract(r'https?://(?:www\.)?([^/]+)').alias('domain')
    ]).filter(pl.col(' shares').str.strip_chars().cast(pl.Float64) > 0)

    df = lazy_df.collect()
    print(f"Данные загружены. Размерность: {df.shape}")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df['log_shares'].to_numpy(), bins=50, color='steelblue', edgecolor='white')
    plt.title('Гистограмма log_shares')
    plt.subplot(1, 2, 2)
    plt.boxplot(df['log_shares'].to_numpy())
    plt.title('Boxplot log_shares')
    plt.savefig('1_eda_distributions.png', bbox_inches='tight')
    plt.close()
    print("Сохранен график: 1_eda_distributions.png")

    # 2. TF-IDF
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['url'].to_list())
    svd = TruncatedSVD(n_components=20, random_state=42)
    tfidf_features = svd.fit_transform(tfidf_matrix)

    for i in range(20):
        df = df.with_columns(pl.Series(f'tfidf_{i}', tfidf_features[:, i]))

    # 3. VIF и Отбор признаков
    numerical_cols = [c for c in df.columns if c not in ['url', 'domain', ' shares', 'log_shares']]
    X_num = df.select(numerical_cols).to_numpy()

    vif_values = []
    for i in range(X_num.shape[1]):
        X_other = np.delete(X_num, i, axis=1)
        y_other = X_num[:, i]
        lr = LinearRegression().fit(X_other, y_other)
        r2 = lr.score(X_other, y_other)
        vif = 1 / (1 - r2) if r2 < 1 else float('inf')
        vif_values.append(vif)

    vif_df = pd.DataFrame({'Feature': numerical_cols, 'VIF': vif_values})
    features_to_drop = vif_df[vif_df['VIF'] > 10]['Feature'].tolist()
    
    df = df.drop(features_to_drop)
    clean_num_cols = [c for c in numerical_cols if c not in features_to_drop]

    rf_rfe = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf_rfe, n_features_to_select=20)
    
    X_clean = df.select(clean_num_cols).to_numpy()
    y = df['log_shares'].to_numpy()
    rfe.fit(X_clean, y)
    
    selected_features = [c for c, s in zip(clean_num_cols, rfe.support_) if s]

    # 4. Кодирование категорий
    df_oh = df.to_dummies(columns=['domain'])
    
    alpha = 10
    global_mean = df['log_shares'].mean()
    cat_stats = df.group_by('domain').agg([
        pl.col('log_shares').mean().alias('cat_mean'),
        pl.len().alias('cat_count')
    ])
    cat_stats = cat_stats.with_columns(
        ((pl.col('cat_mean') * pl.col('cat_count') + alpha * global_mean) / (pl.col('cat_count') + alpha)).alias('domain_target_enc')
    )
    df = df.join(cat_stats.select(['domain', 'domain_target_enc']), on='domain', how='left')

    freq = df.group_by('domain').agg((pl.len() / df.height).alias('domain_freq_enc'))
    df = df.join(freq, on='domain', how='left')

    final_feature_names = selected_features + ['domain_target_enc']
    X_final = df.select(final_feature_names).to_numpy().astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X_final, y, test_size=0.2, random_state=42)

    # 5. Обучение моделей и Optuna
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0)
        }
        model = xgb.XGBRegressor(**params, random_state=42)
        model.fit(X_tr, y_tr)
        return np.sqrt(mean_squared_error(y_te, model.predict(X_te)))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=100)

    best_xgb = xgb.XGBRegressor(**study.best_params, random_state=42).fit(X_tr, y_tr)
    xgb_pred = best_xgb.predict(X_te)
    xgb_rmse = np.sqrt(mean_squared_error(y_te, xgb_pred))
    
    lr_model = LinearRegression().fit(X_tr, y_tr)
    lr_rmse = np.sqrt(mean_squared_error(y_te, lr_model.predict(X_te)))
    
    en_model = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_tr, y_tr)
    en_rmse = np.sqrt(mean_squared_error(y_te, en_model.predict(X_te)))

    # --- ГРАФИК 2: Bar chart сравнения моделей ---
    plt.figure(figsize=(8, 4))
    models_names = ['Linear Regression', 'ElasticNet', 'XGBoost (Optuna)']
    rmses = [lr_rmse, en_rmse, xgb_rmse]
    plt.barh(models_names, rmses, color=['coral', 'gold', 'seagreen'])
    plt.xlabel('RMSE (меньше - лучше)')
    plt.title('Сравнение моделей')
    plt.savefig('2_models_comparison.png', bbox_inches='tight')
    plt.close()
    print("Сохранен график: 2_models_comparison.png")

    # --- ГРАФИК 3: Предсказания vs Фактические ---
    plt.figure(figsize=(8, 6))
    plt.scatter(y_te, xgb_pred, alpha=0.3, s=10, color='steelblue')
    plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--', lw=2)
    plt.xlabel('Фактические значения (log_shares)')
    plt.ylabel('Предсказания XGBoost')
    plt.title('Предсказания vs Фактические значения')
    plt.savefig('3_pred_vs_actual.png', bbox_inches='tight')
    plt.close()
    print("Сохранен график: 3_pred_vs_actual.png")

    # --- ГРАФИК 4: QQ-plot остатков ---
    residuals = y_te - lr_model.predict(X_te)
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('QQ-plot остатков линейной регрессии')
    plt.savefig('4_qq_plot.png', bbox_inches='tight')
    plt.close()
    print("Сохранен график: 4_qq_plot.png")

    # --- ГРАФИК 5: Learning curves ---
    print("Построение Learning Curves...")
    train_sizes, train_scores, val_scores = learning_curve(
        xgb.XGBRegressor(**study.best_params, random_state=42), 
        X_final, y, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_rmse_lc = -train_scores.mean(axis=1)
    val_rmse_lc = -val_scores.mean(axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_rmse_lc, 'o-', color='steelblue', label='Train RMSE')
    plt.plot(train_sizes, val_rmse_lc, 'o-', color='coral', label='Validation RMSE')
    plt.xlabel('Размер обучающей выборки')
    plt.ylabel('RMSE')
    plt.title('Кривые обучения (Learning Curves)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('5_learning_curves.png', bbox_inches='tight')
    plt.close()
    print("Сохранен график: 5_learning_curves.png")

    # --- ГРАФИКИ 6, 7, 8, 9: SHAP ---
    print("Генерация SHAP-графиков...")
    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_te[:1000]) 
    
    shap.summary_plot(shap_values, X_te[:1000], feature_names=final_feature_names, show=False)
    plt.tight_layout()
    plt.savefig('6_shap_summary.png', bbox_inches='tight')
    plt.close()
    print("Сохранен график: 6_shap_summary.png")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top3_idx = np.argsort(mean_abs_shap)[-3:][::-1]
    
    for rank, idx in enumerate(top3_idx, start=1):
        feature_name = final_feature_names[idx]
        shap.dependence_plot(idx, shap_values, X_te[:1000], feature_names=final_feature_names, show=False)
        plt.tight_layout()
        plt.savefig(f'7_shap_dependence_top{rank}_{feature_name}.png', bbox_inches='tight')
        plt.close()
    print("Сохранены 3 графика SHAP dependence_plot.")

    print("\nПайплайн завершен! Все графики сохранены в папке с проектом.")

if __name__ == "__main__":
    main()