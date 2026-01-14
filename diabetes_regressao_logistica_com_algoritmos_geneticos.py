import pandas as pd
import numpy as np
import pygad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score
import base64
import os
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from datetime import datetime
import re

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

# --- 7. INTEGRAÇÃO COM LLM PARA ANÁLISE E GERAÇÃO DE RELATÓRIO PDF ---

def encode_image(image_path):
    """Codifica uma imagem em base64 para envio ao LLM."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analisar_resultados_com_llm(imagem1_path, imagem2_path, imagem3_path):
    """
    Envia as três imagens para o LLM e solicita uma análise detalhada dos resultados.
    Retorna o texto da análise gerada pelo LLM.
    """
    # Verifica se a API key está configurada
    #api_key = os.getenv('OPENAI_API_KEY')
    #api_key = "{OPEN_AI_CHAT_GPT_CONTA_BRUNOJOSE1977_API_KEY}"
    #if not api_key:
    #    print("\n⚠️  AVISO: OPENAI_API_KEY não encontrada nas variáveis de ambiente.")
    #    print("   Por favor, configure a variável de ambiente OPENAI_API_KEY com sua chave da OpenAI.")
    #    print("   Exemplo: export OPENAI_API_KEY='sua-chave-aqui'")
    #    return None
    
    try:
        # carregando a chave api_key "OPEN_AI_CHATGPT_FIAP_TECHCHALLENGER_2_LLM_API_KEY" configurada em .env e disponível como variável de ambiente
        from dotenv import load_dotenv
        load_dotenv()    
        
        # carregando a chave diretamente da variável de ambiente do S.O
        client = OpenAI(api_key=os.getenv("OPEN_AI_CHATGPT_FIAP_TECHCHALLENGER_2_LLM_API_KEY"))
        
        # Codifica as imagens
        print("\n📊 Codificando imagens para análise pelo LLM...")
        img1_base64 = encode_image(imagem1_path)
        img2_base64 = encode_image(imagem2_path)
        img3_base64 = encode_image(imagem3_path)
        
        # Prompt detalhado para análise
        prompt = """Você é um especialista em Machine Learning, Algoritmos Genéticos e análise de resultados de modelos de classificação.

Analise as três imagens fornecidas que mostram os resultados de um experimento de otimização de modelo de Regressão Logística usando Algoritmos Genéticos para diagnóstico de diabetes.

As imagens mostram:
1. Gráfico de evolução linear da função de fitness ao longo das gerações
2. Matrizes de confusão comparando o modelo original vs o modelo otimizado por algoritmos genéticos
3. Gráfico de barras comparativo mostrando Recall, Falsos Positivos e Falsos Negativos

Gere um relatório técnico completo e profissional em português brasileiro que inclua:

1. **Resumo Executivo**: Visão geral dos resultados obtidos
2. **Análise da Evolução do Algoritmo Genético**: Interpretação do gráfico de evolução da fitness
3. **Análise Comparativa dos Modelos**: Comparação detalhada entre modelo original e otimizado
4. **Análise das Matrizes de Confusão**: Interpretação dos resultados de classificação
5. **Análise das Métricas de Performance**: Recall, Falsos Positivos, Falsos Negativos
6. **Conclusões e Insights**: Principais descobertas e recomendações
7. **Impacto Prático**: Significado dos resultados para o diagnóstico de diabetes

Seja detalhado, técnico e forneça insights valiosos sobre a otimização realizada."""
        
        print("🤖 Enviando imagens para análise pelo LLM (GPT-4 Vision)...")
        
        response = client.chat.completions.create(
            model="gpt-4o",  # ou "gpt-4-turbo" se disponível
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img1_base64}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img2_base64}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img3_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4000
        )
        
        analise = response.choices[0].message.content
        print("✅ Análise gerada com sucesso pelo LLM!")
        return analise
        
    except Exception as e:
        print(f"\n❌ Erro ao comunicar com o LLM: {str(e)}")
        return None

def gerar_relatorio_pdf(analise_texto, output_path, dados_ranking=None):
    """
    Gera um relatório em PDF com a análise fornecida pelo LLM.
    """
    if not analise_texto:
        print("⚠️  Não foi possível gerar o PDF: análise não disponível.")
        return False
    
    try:
        print(f"\n📄 Gerando relatório PDF: {output_path}")
        
        # Cria o documento PDF
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Estilos
        styles = getSampleStyleSheet()
        
        # Título principal
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor='#2c3e50',
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Subtítulo
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=14,
            textColor='#34495e',
            spaceAfter=20,
            alignment=TA_CENTER
        )
        
        # Corpo do texto
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            textColor='#2c3e50',
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            leading=14
        )
        
        # Título do relatório
        story.append(Paragraph("Relatório de Análise de Resultados", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Subtítulo
        story.append(Paragraph("Regressão Logística e Algoritmos Genéticos", subtitle_style))
        story.append(Paragraph("Tech Challenger 2 - Fase 2", subtitle_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Data
        data_atual = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        story.append(Paragraph(f"<i>Gerado em: {data_atual}</i>", styles['Normal']))
        story.append(Spacer(1, 0.4*inch))
        
        # Divide o texto da análise em seções e formata markdown
        linhas = analise_texto.split('\n')
        i = 0
        while i < len(linhas):
            linha = linhas[i].strip()
            
            if not linha:
                story.append(Spacer(1, 0.1*inch))
                i += 1
                continue
            
            # Detecta títulos markdown (# ## ###)
            if linha.startswith('###'):
                titulo = linha.replace('###', '').strip()
                if titulo:
                    story.append(Spacer(1, 0.15*inch))
                    story.append(Paragraph(f"<b>{titulo}</b>", styles['Heading3']))
                    story.append(Spacer(1, 0.08*inch))
            elif linha.startswith('##'):
                titulo = linha.replace('##', '').strip()
                if titulo:
                    story.append(Spacer(1, 0.2*inch))
                    story.append(Paragraph(f"<b>{titulo}</b>", styles['Heading2']))
                    story.append(Spacer(1, 0.1*inch))
            elif linha.startswith('#'):
                titulo = linha.replace('#', '').strip()
                if titulo:
                    story.append(Spacer(1, 0.25*inch))
                    story.append(Paragraph(f"<b>{titulo}</b>", styles['Heading1']))
                    story.append(Spacer(1, 0.15*inch))
            else:
                # Processa texto com formatação markdown
                texto_formatado = linha
                # Converte **texto** para <b>texto</b>
                texto_formatado = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', texto_formatado)
                # Converte *texto* para <i>texto</i> (mas não se já foi processado como negrito)
                texto_formatado = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', texto_formatado)
                # Escapa caracteres especiais HTML, mas preserva tags que criamos
                texto_formatado = texto_formatado.replace('&', '&amp;')
                # Restaura as tags HTML que criamos após escape
                texto_formatado = texto_formatado.replace('&amp;lt;b&amp;gt;', '<b>').replace('&amp;lt;/b&amp;gt;', '</b>')
                texto_formatado = texto_formatado.replace('&amp;lt;i&amp;gt;', '<i>').replace('&amp;lt;/i&amp;gt;', '</i>')
                texto_formatado = texto_formatado.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
                texto_formatado = texto_formatado.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
                
                if texto_formatado.strip():
                    story.append(Paragraph(texto_formatado, body_style))
            
            i += 1
        
        # Adiciona informações do ranking se disponível
        if dados_ranking is not None and len(dados_ranking) > 0:
            story.append(PageBreak())
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph("<b>Top 5 Configurações do Algoritmo Genético</b>", styles['Heading2']))
            story.append(Spacer(1, 0.2*inch))
            
            for i, row in enumerate(dados_ranking.head(5).iterrows(), 1):
                idx, data = row
                story.append(Paragraph(
                    f"<b>Rank {i}:</b> Fitness={data['fitness']:.1f}, "
                    f"Recall={data['recall']:.2%}, FP={int(data['fp'])}, FN={int(data['fn'])}, "
                    f"Acc={data['acc']:.2%}, C={data['c']:.3f}, Solver={data['solver']}, "
                    f"IQR={data['iqr']:.2f}, Features={int(data['features'])}",
                    body_style
                ))
                story.append(Spacer(1, 0.1*inch))
        
        # Constrói o PDF
        doc.build(story)
        print(f"✅ Relatório PDF gerado com sucesso: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao gerar PDF: {str(e)}")
        return False

# Executa a análise e geração do relatório
print("\n" + "="*110)
print("INICIANDO ANÁLISE COM LLM E GERAÇÃO DE RELATÓRIO PDF")
print("="*110)

imagens = [
    'grafico_evolucao_linear.png',
    'matrizes_confusao_antes_depois.png',
    'grafico_barras_comparativo.png'
]

# Verifica se todas as imagens existem
imagens_existentes = [img for img in imagens if os.path.exists(img)]
if len(imagens_existentes) == 3:
    analise = analisar_resultados_com_llm(imagens[0], imagens[1], imagens[2])
    
    if analise:
        # Prepara dados do ranking para incluir no PDF
        df_ranking_pdf = pd.DataFrame(ranking_data).sort_values(by='fitness', ascending=False).drop_duplicates(subset=['c', 'iqr']).head(5)
        
        sucesso = gerar_relatorio_pdf(
            analise,
            'Relatorio_Resultado_TechChallenger2_Regressao_logistica_e_Algorimos_geneticos.pdf',
            df_ranking_pdf
        )
        
        if sucesso:
            print("\n✅ Processo concluído com sucesso!")
        else:
            print("\n⚠️  Relatório PDF não pôde ser gerado.")
    else:
        print("\n⚠️  Análise não foi gerada. Verifique a configuração da API key.")
else:
    print(f"\n⚠️  Algumas imagens não foram encontradas. Encontradas: {len(imagens_existentes)}/3")
    print(f"   Imagens esperadas: {imagens}")

print("="*110)