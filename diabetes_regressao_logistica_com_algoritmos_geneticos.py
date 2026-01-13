import pandas as pd
import numpy as np
import pygad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score

# --- 1. PREPARAÇÃO DOS DADOS ---
df = pd.read_csv('datasets/diabetes.csv')
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)
for col in cols_to_replace:
    df[col] = df[col].fillna(df[col].median())

X_raw = df.drop('Outcome', axis=1)
y_raw = df['Outcome']

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

ranking_data = []

# --- 2. FUNÇÃO DE FITNESS ---
def fitness_func(ga_instance, solution, solution_idx):
    c_val, solver_idx, _, iqr_f = solution[0:4]
    feature_mask = solution[4:].astype(bool)
    selected_solver = ['lbfgs', 'liblinear', 'saga'][int(solver_idx)]

    df_temp = pd.concat([X_train_full, y_train], axis=1).copy()
    for col in X_train_full.columns:
        Q1, Q3 = df_temp[col].quantile(0.25), df_temp[col].quantile(0.75)
        df_temp = df_temp[(df_temp[col] >= Q1 - iqr_f*(Q3-Q1)) & (df_temp[col] <= Q3 + iqr_f*(Q3-Q1))]
    
    if len(df_temp) < 150 or not any(feature_mask): return -99999 

    X_train_ga = df_temp.drop('Outcome', axis=1).iloc[:, feature_mask]
    y_train_ga = df_temp['Outcome']
    X_test_ga = X_test_full.iloc[:, feature_mask]

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_train_ga)
    X_ts_sc = sc.transform(X_test_ga)

    model = LogisticRegression(C=c_val, solver=selected_solver, class_weight='balanced', max_iter=3000, random_state=42)
    model.fit(X_tr_sc, y_train_ga)
    
    preds = model.predict(X_ts_sc)
    cm = confusion_matrix(y_test, preds)
    if len(cm) < 2: return -99999
    
    recall = recall_score(y_test, preds, zero_division=0)
    fp, fn, acc = cm[0][1], cm[1][0], accuracy_score(y_test, preds)
    
    fitness = recall * 1000
    if recall < 0.80: fitness -= (0.80 - recall) * 5000
    elif recall > 0.95: fitness -= (recall - 0.95) * 2000
    if fp > 60: fitness -= (fp - 60) * 100
    
    ranking_data.append({
        'fitness': fitness, 'c': c_val, 'solver': selected_solver, 
        'iqr': iqr_f, 'recall': recall, 'fp': fp, 'fn': fn, 'acc': acc,
        'features': int(sum(feature_mask))
    })
    return fitness

# --- 3. EXECUÇÃO DO AG ---
gene_space = [{'low': 0.001, 'high': 50.0}, [0, 1, 2], [0], {'low': 1.5, 'high': 5.0}] + [[0, 1]] * 8
ga_instance = pygad.GA(num_generations=100, num_parents_mating=10, fitness_func=fitness_func, 
                       sol_per_pop=50, num_genes=len(gene_space), gene_space=gene_space, mutation_percent_genes=15)
print("Otimizando modelo...")
ga_instance.run()

# --- 4. RECONSTRUÇÃO PARA GRÁFICOS ---
best_sol, _, _ = ga_instance.best_solution()
c_b, s_idx, _, iqr_b = best_sol[0:4]
mask_b = best_sol[4:].astype(bool)
solver_b = ['lbfgs', 'liblinear', 'saga'][int(s_idx)]

df_f = pd.concat([X_train_full, y_train], axis=1)
for col in X_train_full.columns:
    Q1, Q3 = df_f[col].quantile(0.25), df_f[col].quantile(0.75)
    df_f = df_f[(df_f[col] >= Q1 - iqr_b*(Q3-Q1)) & (df_f[col] <= Q3 + iqr_b*(Q3-Q1))]

sc_ag = StandardScaler()
X_tr_sc_ag = sc_ag.fit_transform(df_f.drop('Outcome', axis=1).iloc[:, mask_b])
X_ts_sc_ag = sc_ag.transform(X_test_full.iloc[:, mask_b])
model_ag = LogisticRegression(C=c_b, solver=solver_b, class_weight='balanced', max_iter=3000).fit(X_tr_sc_ag, df_f['Outcome'])
y_pred_ag = model_ag.predict(X_ts_sc_ag)
cm_ag = confusion_matrix(y_test, y_pred_ag)

sc_orig = StandardScaler()
mod_orig = LogisticRegression(solver='liblinear').fit(sc_orig.fit_transform(X_train_full), y_train)
y_pred_orig = mod_orig.predict(sc_orig.transform(X_test_full))
cm_orig = confusion_matrix(y_test, y_pred_orig)

# --- 5. GRAVAÇÃO DOS GRÁFICOS COM LEGENDAS E RÓTULOS ---

# A. Evolução Linear
plt.figure(figsize=(10, 5))
plt.plot(ga_instance.best_solutions_fitness, label="Melhor Fitness da Geração", color='darkgreen', linewidth=2)
plt.title("Evolução da Função de Aptidão (Fitness) ao Longo das Gerações")
plt.xlabel("Geração")
plt.ylabel("Aptidão (Fitness)")
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('grafico_evolucao_linear.png')
plt.close()

# B. Matrizes de Confusão
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Reds', ax=ax[0])
ax[0].set_title(f'Original\nRecall: {recall_score(y_test, y_pred_orig):.2f}')
ax[0].set_xlabel('Previsão'); ax[0].set_ylabel('Real')

sns.heatmap(cm_ag, annot=True, fmt='d', cmap='Greens', ax=ax[1])
ax[1].set_title(f'AG Otimizado\nRecall: {recall_score(y_test, y_pred_ag):.2f}')
ax[1].set_xlabel('Previsão'); ax[1].set_ylabel('Real')
plt.savefig('matrizes_confusao_antes_depois.png')
plt.close()

# C. Gráfico de Barras com Rótulos
labels = ['Original', 'AG Balanceado']
recall_vals = [recall_score(y_test, y_pred_orig), recall_score(y_test, y_pred_ag)]
fp_vals = [cm_orig[0][1], cm_ag[0][1]]
fn_vals = [cm_orig[1][0], cm_ag[1][0]]

x = np.arange(len(labels))
width = 0.25
fig, ax1 = plt.subplots(figsize=(12, 7))

b1 = ax1.bar(x - width, recall_vals, width, label='Recall (Taxa)', color='#3498db')
ax1.set_ylabel('Taxa de Recall', color='#3498db', fontsize=12)
ax1.set_ylim(0, 1.1)

ax2 = ax1.twinx()
b2 = ax2.bar(x, fp_vals, width, label='Falsos Positivos (Qtd)', color='#e67e22')
b3 = ax2.bar(x + width, fn_vals, width, label='Falsos Negativos (Qtd)', color='#e74c3c')
ax2.set_ylabel('Quantidade de Casos', color='black', fontsize=12)

plt.title('Comparativo de Performance: Antes vs Depois do AG')
ax1.set_xticks(x); ax1.set_xticklabels(labels)

# Adicionando Rótulos (Labels) nas barras
def add_labels(rects, ax, is_percent=False):
    for rect in rects:
        h = rect.get_height()
        label = f'{h:.2f}' if is_percent else f'{int(h)}'
        ax.text(rect.get_x() + rect.get_width()/2., h, label, ha='center', va='bottom', fontweight='bold')

add_labels(b1, ax1, True)
add_labels(b2, ax2); add_labels(b3, ax2)

# Unificando legendas de eixos diferentes
lines, labs = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labs + labs2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

plt.tight_layout()
plt.savefig('grafico_barras_comparativo.png')
plt.close()

# --- 6. TABELA RANKING ---
df_ranking = pd.DataFrame(ranking_data).sort_values(by='fitness', ascending=False).drop_duplicates(subset=['c', 'iqr']).head(5)
print("\n" + "="*110)
print(f"{'RANK':<4} | {'FITNESS':<8} | {'RECALL':<7} | {'FP':<4} | {'FN':<4} | {'ACC':<6} | {'C':<7} | {'SOLVER':<10} | {'IQR':<4} | {'FEAT'}")
print("-" * 110)
for i, (idx, row) in enumerate(df_ranking.iterrows(), 1):
    print(f"{i:<4} | {row['fitness']:<8.1f} | {row['recall']:<7.2%} | {int(row['fp']):<4} | {int(row['fn']):<4} | {row['acc']:<6.2%} | {row['c']:<7.3f} | {row['solver']:<10} | {row['iqr']:<4.2f} | {int(row['features'])}")
print("="*110)