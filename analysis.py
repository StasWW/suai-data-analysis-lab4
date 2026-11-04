from pathlib import Path
import json
import hashlib
import os
import tempfile

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'suai-lab4-matplotlib'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
OUT = BASE / 'output'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})


def save_figure(name):
    plt.savefig(OUT / name, dpi=300, bbox_inches='tight')
    plt.close()


def make_data(n, seed):
    np.random.seed(seed)
    shift_matrix = np.array([[3, 3], [6, 9], [9, 3]])
    data = np.random.randn(3, 2, n) + shift_matrix.reshape((3, 2, 1))
    data = np.swapaxes(data, 1, 2)
    data = data.reshape((-1, 2))
    data *= np.array([[20, 0.5]])
    df = pd.DataFrame({'x': data[:, 0], 'y': data[:, 1]}, columns=['x', 'y'])
    return df.sample(frac=1.0)


def metrics(x, labels, target, distances):
    return {
        'silhouette': silhouette_score(distances, labels, metric='precomputed'),
        'calinski_harabasz': calinski_harabasz_score(x, labels),
        'davies_bouldin': davies_bouldin_score(x, labels),
        'ARI': adjusted_rand_score(target, labels),
        'AMI': adjusted_mutual_info_score(target, labels),
        'FMI': fowlkes_mallows_score(target, labels),
        'homogeneity': homogeneity_score(target, labels),
        'completeness': completeness_score(target, labels),
        'V_measure': v_measure_score(target, labels),
    }


def run():
    synthetic = make_data(100, 8)
    scaler = StandardScaler()
    x = scaler.fit_transform(synthetic)
    assert np.allclose(x.mean(axis=0), 0, atol=1e-12)
    assert np.allclose(x.std(axis=0), 1)
    synthetic_rows = []
    synthetic_labels = synthetic.copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, k in zip(axes, [2, 3, 4]):
        model = KMeans(n_clusters=k, n_init=20, random_state=8)
        labels = model.fit_predict(x)
        synthetic_labels[f'cluster_k{k}'] = labels
        centers = scaler.inverse_transform(model.cluster_centers_)
        ax.scatter(synthetic.x, synthetic.y, c=labels, cmap='tab10', s=10)
        ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=90, label='Центры')
        ax.set(xlabel='x', ylabel='y', title=f'Число кластеров {k}')
        ax.legend(fontsize=8)
        synthetic_rows.append({'k': k, 'silhouette': silhouette_score(x, labels), 'inertia': model.inertia_})
    fig.tight_layout()
    save_figure('synthetic_clusters.png')
    synthetic_labels.to_csv(OUT / 'synthetic_labels.csv', index_label='source_row')
    pd.DataFrame(synthetic_rows).to_csv(OUT / 'synthetic_metrics.csv', index=False)
    synthetic_elbow = []
    for k in range(1, 11):
        model = KMeans(n_clusters=k, n_init=20, random_state=8).fit(x)
        synthetic_elbow.append({'k': k, 'inertia': model.inertia_})
    pd.DataFrame(synthetic_elbow).to_csv(OUT / 'synthetic_elbow.csv', index=False)
    plt.figure(figsize=(7, 4))
    plt.plot([r['k'] for r in synthetic_elbow], [r['inertia'] for r in synthetic_elbow], 'o-')
    plt.xticks(range(1, 11))
    plt.xlabel('Число кластеров')
    plt.ylabel('Сумма квадратов расстояний')
    plt.grid(alpha=.3)
    save_figure('synthetic_elbow.png')

    source = BASE / 'input' / 'ecommerce_fraud.csv'
    raw = pd.read_csv(source)
    numeric = ['transaction_amount', 'num_items', 'time_on_site_sec', 'hour_of_day', 'account_age_days']
    categorical = ['device_type', 'browser', 'payment_method', 'country', 'is_first_purchase', 'referral']
    audit = {
        'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'raw_rows': len(raw), 'columns': len(raw.columns),
        'missing': raw.isna().sum().to_dict(), 'duplicates': int(raw.duplicated().sum()),
        'numeric_features': numeric, 'categorical_features': categorical,
        'target_counts': raw.Fraud.value_counts().sort_index().to_dict(),
    }
    invalid = (raw.transaction_amount < 0) | (raw.time_on_site_sec < 0) | (raw.account_age_days < 0) | (~raw.num_items.between(1, 7)) | (~raw.hour_of_day.between(0, 23))
    raw.loc[invalid].to_csv(OUT / 'rejected_rows.csv', index_label='source_row')
    audit['invalid_rows'] = int(invalid.sum())
    audit['negative_amount_rows'] = int((raw.transaction_amount < 0).sum())
    audit['negative_time_rows'] = int((raw.time_on_site_sec < 0).sum())
    audit['negative_amount_and_time_rows'] = int(((raw.transaction_amount < 0) & (raw.time_on_site_sec < 0)).sum())
    audit['whitespace_categories'] = {column: int(raw[column].ne(raw[column].str.strip()).sum()) for column in categorical}
    data = raw.loc[~invalid].drop_duplicates().copy()
    audit['clean_target_counts'] = data.Fraud.value_counts().sort_index().to_dict()
    assert data[numeric + categorical + ['Fraud']].notna().all().all()
    assert set(data.Fraud.unique()) == {0, 1}
    assert (data[['transaction_amount', 'time_on_site_sec', 'account_age_days']] >= 0).all().all()
    assert data.num_items.between(1, 7).all()
    assert data.hour_of_day.between(0, 23).all()
    data[numeric].describe().to_csv(OUT / 'descriptive_statistics.csv')
    data.to_csv(OUT / 'clean_data.csv', index=False)
    audit['clean_rows'] = len(data)
    labels_ru = ['Сумма, USD', 'Число товаров', 'Время, с', 'Час суток', 'Возраст аккаунта, дней']
    fig, axes = plt.subplots(5, 5, figsize=(13, 13))
    colors = ['#2374ab', '#e4572e']
    for i, col_y in enumerate(numeric):
        for j, col_x in enumerate(numeric):
            ax = axes[i, j]
            for target, color in enumerate(colors):
                subset = data[data.Fraud == target]
                if i == j:
                    ax.hist(subset[col_x], bins=24, color=color, alpha=.5, density=True)
                else:
                    ax.scatter(subset[col_x], subset[col_y], color=color, alpha=.25, s=3, rasterized=True)
            ax.tick_params(labelsize=7)
            if i == 4:
                ax.set_xlabel(labels_ru[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(labels_ru[i], fontsize=9)
    fig.suptitle('Синий: обычная транзакция; оранжевый: мошенническая', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97))
    save_figure('scatter_matrix.png')
    scaler = StandardScaler()
    x = scaler.fit_transform(data[numeric])
    assert 'Fraud' not in numeric
    assert np.allclose(x.mean(axis=0), 0, atol=1e-12)
    assert np.allclose(x.std(axis=0), 1)
    pd.DataFrame({'feature': numeric, 'mean': scaler.mean_, 'scale': scaler.scale_}).to_csv(OUT / 'scaler.csv', index=False)
    distances = pairwise_distances(x)
    np.fill_diagonal(distances, 0)
    models = {}
    rows = []
    for method in ['KMeans', 'Ward']:
        for k in range(2, 11):
            model = (KMeans(n_clusters=k, n_init=20, random_state=8) if method == 'KMeans'
                     else AgglomerativeClustering(n_clusters=k, linkage='ward'))
            labels = model.fit_predict(x)
            models[(method, k)] = labels
            row = {'method': method, 'k': k, **metrics(x, labels, data.Fraud, distances)}
            row['inertia'] = model.inertia_ if method == 'KMeans' else np.nan
            rows.append(row)
    all_metrics = pd.DataFrame(rows)
    all_metrics.to_csv(OUT / 'model_selection.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    km = all_metrics[all_metrics.method == 'KMeans']
    axes[0].plot(km.k, km.inertia, 'o-')
    axes[0].set(xlabel='Число кластеров', ylabel='Сумма квадратов расстояний', title='Метод локтя для KMeans')
    for method in ['KMeans', 'Ward']:
        sub = all_metrics[all_metrics.method == method]
        axes[1].plot(sub.k, sub.silhouette, 'o-', label=method)
    axes[1].set(xlabel='Число кластеров', ylabel='Коэффициент силуэта', title='Сравнение разбиений')
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(range(2, 11))
        ax.grid(alpha=.3)
    fig.tight_layout()
    save_figure('model_selection.png')
    linked = linkage(x, method='ward')
    np.save(OUT / 'ward_linkage.npy', linked)
    pd.DataFrame(linked[-12:], columns=['left', 'right', 'distance', 'size']).to_csv(OUT / 'ward_last_merges.csv', index=False)
    plt.figure(figsize=(10, 5))
    dendrogram(linked, truncate_mode='lastp', p=20, show_leaf_counts=True, leaf_rotation=90, leaf_font_size=9, color_threshold=50)
    plt.axhline(50, color='black', linestyle='--', linewidth=1, label='Срез для трёх кластеров')
    plt.legend(fontsize=9)
    plt.xlabel('Объединённые группы, в скобках число объектов')
    plt.ylabel('Расстояние Уорда')
    plt.title('Дендрограмма всех транзакций, последние 20 групп')
    save_figure('dendrogram.png')
    selected = {}
    combined_labels = data.copy()
    for method in ['KMeans', 'Ward']:
        sub = all_metrics[all_metrics.method == method]
        best = sub.loc[sub.silhouette.idxmax()]
        k = int(best.k)
        selected[method] = {'k': k, **{key: float(best[key]) for key in metrics(x, models[(method, k)], data.Fraud, distances)}}
        labels = models[(method, k)]
        combined_labels[f'cluster_{method}'] = labels
        profile = data.assign(cluster=labels).groupby('cluster')[numeric + ['Fraud']].mean()
        profile.insert(0, 'size', pd.Series(labels).value_counts().sort_index())
        profile.to_csv(OUT / f'{method.lower()}_profiles.csv')
        pd.crosstab(labels, data.Fraud).to_csv(OUT / f'{method.lower()}_target_counts.csv')
        shares = []
        for feature in categorical:
            for cluster in range(k):
                for category, share in data.loc[labels == cluster, feature].value_counts(normalize=True).sort_index().items():
                    shares.append({'cluster': cluster, 'feature': feature, 'category': category, 'share': share})
        pd.DataFrame(shares).to_csv(OUT / f'{method.lower()}_category_shares.csv', index=False)
    combined_labels.to_csv(OUT / 'cluster_assignments.csv', index_label='source_row')
    audit['selected'] = selected
    audit['synthetic'] = {'n_per_group': 100, 'seed': 8, 'rows': len(synthetic), 'metrics': synthetic_rows}
    audit['numeric_ranges'] = {key: {'min': float(data[key].min()), 'max': float(data[key].max())} for key in numeric}
    (OUT / 'results.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = pd.DataFrame(selected).T.reset_index(names='Метод')
    prep = pd.DataFrame({'Проверка': ['Исходные строки', 'Пропуски', 'Дубликаты', 'Отрицательная сумма', 'Отрицательное время', 'Пересечение причин', 'После очистки', 'Fraud после очистки'], 'Количество': [len(raw), int(raw.isna().sum().sum()), int(raw.duplicated().sum()), audit['negative_amount_rows'], audit['negative_time_rows'], audit['negative_amount_and_time_rows'], len(data), int(data.Fraud.sum())]})
    html = '<!doctype html><html lang="ru"><meta charset="utf-8"><title>ЛР4. Результаты расчётов</title><style>body{font:18px Arial;margin:24px;color:#202124}section{margin-bottom:50px}table{border-collapse:collapse;font-size:16px}th,td{padding:8px 12px;border:1px solid #aaa;text-align:right}th{background:#eee}h1{font-size:26px}h2{font-size:23px}</style><h1>ЛР4. Результаты выполненных расчётов</h1>'
    html += '<section id="preprocessing"><h2>Проверка данных</h2>' + prep.to_html(index=False) + '</section>'
    html += '<section id="synthetic"><h2>Синтетические данные, n=100, seed=8</h2>' + pd.DataFrame(synthetic_rows).round(6).to_html(index=False) + '</section>'
    html += '<section id="comparison"><h2>Выбранные разбиения транзакций</h2>' + summary[['Метод', 'k', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'ARI', 'AMI']].round(6).to_html(index=False) + '</section>'
    for method in ['kmeans', 'ward']:
        profile = pd.read_csv(OUT / f'{method}_profiles.csv').rename(columns={'cluster':'Кластер','size':'Число','transaction_amount':'Сумма USD','num_items':'Товары','time_on_site_sec':'Время с','hour_of_day':'Час','account_age_days':'Возраст дней','Fraud':'Доля Fraud'})
        html += f'<section id="{method}"><h2>Профили {method}</h2>' + profile.round(4).to_html(index=False) + '</section>'
    html += '</html>'
    (OUT / 'results.html').write_text(html, encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(all_metrics.to_string(index=False))
    for method in ['kmeans', 'ward']:
        print(method)
        print(pd.read_csv(OUT / f'{method}_profiles.csv').to_string(index=False))
    assert len(combined_labels) == len(data)
    assert len(rows) == 18
    assert len(linked) == len(data) - 1
    assert all((OUT / name).exists() for name in ['synthetic_clusters.png', 'synthetic_elbow.png', 'scatter_matrix.png', 'model_selection.png', 'dendrogram.png'])


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
