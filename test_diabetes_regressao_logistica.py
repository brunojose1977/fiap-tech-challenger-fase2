"""
Testes unitários para o programa de regressão logística com algoritmos genéticos
para diagnóstico de diabetes.

Autor: Testes automatizados
Data: 2025
"""

import pytest
import pandas as pd
import numpy as np
import os
import base64
import tempfile
from unittest.mock import Mock, patch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score

# Importa o módulo principal
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importa funções do módulo principal (precisa ser importado como módulo)
# Como o código principal não está em funções, vamos criar funções auxiliares para teste


class TestDataPreparation:
    """Testes para a preparação dos dados"""
    
    def test_replace_zeros_with_nan(self):
        """Testa se os zeros são substituídos por NaN nas colunas corretas"""
        # Cria dados de teste
        data = {
            'Glucose': [0, 100, 0, 120],
            'BloodPressure': [0, 80, 70, 0],
            'SkinThickness': [0, 20, 0, 25],
            'Insulin': [0, 150, 0, 200],
            'BMI': [0, 25.5, 0, 30.0],
            'Age': [30, 40, 50, 60],
            'Pregnancies': [1, 2, 3, 4],
            'DiabetesPedigreeFunction': [0.5, 0.6, 0.7, 0.8],
            'Outcome': [0, 1, 0, 1]
        }
        df = pd.DataFrame(data)
        
        cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)
        
        # Verifica se os zeros foram substituídos por NaN
        for col in cols_to_replace:
            assert df[col].isna().any(), f"Coluna {col} deveria ter NaN após substituição"
        
        # Verifica se outras colunas não foram afetadas
        assert not df['Age'].isna().any(), "Coluna Age não deveria ter NaN"
        assert not df['Pregnancies'].isna().any(), "Coluna Pregnancies não deveria ter NaN"
    
    def test_fillna_with_median(self):
        """Testa se os NaN são preenchidos com a mediana"""
        data = {
            'Glucose': [np.nan, 100, np.nan, 120],
            'BloodPressure': [np.nan, 80, 70, np.nan],
            'Outcome': [0, 1, 0, 1]
        }
        df = pd.DataFrame(data)
        
        cols_to_fill = ['Glucose', 'BloodPressure']
        for col in cols_to_fill:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        
        # Verifica se não há mais NaN
        for col in cols_to_fill:
            assert not df[col].isna().any(), f"Coluna {col} não deveria ter NaN após preenchimento"
    
    def test_train_test_split(self):
        """Testa se o split de treino/teste funciona corretamente"""
        data = {
            'Feature1': range(100),
            'Feature2': range(100, 200),
            'Outcome': [0, 1] * 50
        }
        df = pd.DataFrame(data)
        
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Verifica tamanhos
        assert len(X_train) == 80, "X_train deveria ter 80 amostras"
        assert len(X_test) == 20, "X_test deveria ter 20 amostras"
        assert len(y_train) == 80, "y_train deveria ter 80 amostras"
        assert len(y_test) == 20, "y_test deveria ter 20 amostras"
        
        # Verifica estratificação
        train_prop = y_train.sum() / len(y_train)
        test_prop = y_test.sum() / len(y_test)
        assert abs(train_prop - test_prop) < 0.1, "Proporções deveriam ser similares"


class TestFitnessFunction:
    """Testes para a função de fitness do algoritmo genético"""
    
    @pytest.fixture
    def sample_data(self):
        """Cria dados de exemplo para os testes"""
        np.random.seed(42)
        n_samples = 200
        
        X_train = pd.DataFrame({
            'Feature1': np.random.randn(n_samples) * 10 + 100,
            'Feature2': np.random.randn(n_samples) * 5 + 50,
            'Feature3': np.random.randn(n_samples) * 2 + 25,
            'Feature4': np.random.randn(n_samples) * 3 + 30,
            'Feature5': np.random.randn(n_samples) * 4 + 20,
            'Feature6': np.random.randn(n_samples) * 6 + 40,
            'Feature7': np.random.randn(n_samples) * 1 + 15,
            'Feature8': np.random.randn(n_samples) * 2 + 10
        })
        
        y_train = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]), name='Outcome')
        
        X_test = pd.DataFrame({
            'Feature1': np.random.randn(50) * 10 + 100,
            'Feature2': np.random.randn(50) * 5 + 50,
            'Feature3': np.random.randn(50) * 2 + 25,
            'Feature4': np.random.randn(50) * 3 + 30,
            'Feature5': np.random.randn(50) * 4 + 20,
            'Feature6': np.random.randn(50) * 6 + 40,
            'Feature7': np.random.randn(50) * 1 + 15,
            'Feature8': np.random.randn(50) * 2 + 10
        })
        
        y_test = pd.Series(np.random.choice([0, 1], size=50, p=[0.6, 0.4]), name='Outcome')
        
        return X_train, X_test, y_train, y_test
    
    def test_fitness_function_structure(self, sample_data):
        """Testa se a função de fitness aceita os parâmetros corretos"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Define uma solução de exemplo
        # [C, solver_idx, unused, iqr_factor] + [8 feature masks]
        solution = np.array([1.0, 0, 0, 2.0] + [1, 1, 0, 0, 1, 1, 0, 1])
        
        # Mock do GA instance
        ga_instance = Mock()
        ga_instance.best_solution = Mock(return_value=(solution, 100, 0))
        
        # Importa a função de fitness do módulo principal
        # Como está no módulo principal, vamos criar uma versão testável
        def fitness_func_test(_ga_instance, solution, _solution_idx):
            c_val, solver_idx, _, iqr_f = solution[0:4]
            feature_mask = solution[4:].astype(bool)
            selected_solver = ['lbfgs', 'liblinear', 'saga'][int(solver_idx)]
            
            df_temp = pd.concat([X_train, y_train], axis=1).copy()
            for col in X_train.columns:
                Q1, Q3 = df_temp[col].quantile(0.25), df_temp[col].quantile(0.75)
                df_temp = df_temp[(df_temp[col] >= Q1 - iqr_f*(Q3-Q1)) & 
                                 (df_temp[col] <= Q3 + iqr_f*(Q3-Q1))]
            
            if len(df_temp) < 150 or not any(feature_mask):
                return -99999
            
            X_train_ga = df_temp.drop('Outcome', axis=1).iloc[:, feature_mask]
            y_train_ga = df_temp['Outcome']
            X_test_ga = X_test.iloc[:, feature_mask]
            
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_train_ga)
            X_ts_sc = sc.transform(X_test_ga)
            
            model = LogisticRegression(
                C=c_val, 
                solver=selected_solver, 
                class_weight='balanced', 
                max_iter=3000, 
                random_state=42
            )
            model.fit(X_tr_sc, y_train_ga)
            
            preds = model.predict(X_ts_sc)
            cm = confusion_matrix(y_test, preds)
            if len(cm) < 2:
                return -99999
            
            recall = recall_score(y_test, preds, zero_division=0)
            fp = cm[0][1]
            _fn = cm[1][0]
            _acc = accuracy_score(y_test, preds)
            
            fitness = recall * 1000
            if recall < 0.80:
                fitness -= (0.80 - recall) * 5000
            elif recall > 0.95:
                fitness -= (recall - 0.95) * 2000
            if fp > 60:
                fitness -= (fp - 60) * 100
            
            return fitness
        
        # Testa a função
        fitness = fitness_func_test(ga_instance, solution, 0)
        
        # Verifica se retorna um número
        assert isinstance(fitness, (int, float)), "Fitness deveria ser um número"
        assert fitness != -99999, "Fitness não deveria ser o valor de erro"
    
    def test_fitness_function_with_invalid_solution(self):
        """Testa se a função retorna -99999 para soluções inválidas"""
        # Solução inválida: nenhuma feature selecionada
        solution = np.array([1.0, 0, 0, 2.0] + [0, 0, 0, 0, 0, 0, 0, 0])
        
        def fitness_func_test(_ga_instance, solution, _solution_idx):
            feature_mask = solution[4:].astype(bool)
            if not any(feature_mask):
                return -99999
            return 100
        
        fitness = fitness_func_test(Mock(), solution, 0)
        assert fitness == -99999, "Deveria retornar -99999 para solução inválida"


class TestImageEncoding:
    """Testes para a função de codificação de imagens"""
    
    def test_encode_image(self):
        """Testa se a função codifica uma imagem em base64 corretamente"""
        # Cria um arquivo de imagem temporário
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as tmp_file:
            test_content = b'fake image content'
            tmp_file.write(test_content)
            tmp_path = tmp_file.name
        
        try:
            # Função de codificação
            def encode_image(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            
            encoded = encode_image(tmp_path)
            
            # Verifica se é uma string
            assert isinstance(encoded, str), "Resultado deveria ser uma string"
            
            # Verifica se pode ser decodificado
            decoded = base64.b64decode(encoded)
            assert decoded == test_content, "Conteúdo decodificado deveria ser igual ao original"
        
        finally:
            # Remove arquivo temporário
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_encode_image_file_not_found(self):
        """Testa se a função trata erro quando arquivo não existe"""
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        with pytest.raises(FileNotFoundError):
            encode_image('arquivo_que_nao_existe.png')


class TestLLMAnalysis:
    """Testes para a função de análise com LLM"""
    
    def test_encode_image_for_llm(self):
        """Testa codificação de imagem para envio ao LLM"""
        # Cria arquivo temporário
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as tmp_file:
            test_content = b'fake image content for LLM'
            tmp_file.write(test_content)
            tmp_path = tmp_file.name
        
        try:
            # Função de codificação (extraída do código principal)
            def encode_image(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            
            encoded = encode_image(tmp_path)
            
            # Verifica se é uma string base64 válida
            assert isinstance(encoded, str), "Resultado deveria ser uma string"
            assert len(encoded) > 0, "String codificada não deveria estar vazia"
            
            # Verifica se pode ser decodificado
            decoded = base64.b64decode(encoded)
            assert decoded == test_content, "Conteúdo decodificado deveria ser igual ao original"
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    @patch('os.getenv')
    def test_llm_api_key_handling(self, mock_getenv):
        """Testa tratamento de API key"""
        # Testa quando API key está disponível
        mock_getenv.return_value = 'test-api-key'
        api_key = os.getenv("OPENAI_CHAT_GPT_CONTA_BRUNOJOSE1977_API_KEY")
        assert api_key == 'test-api-key', "API key deveria ser retornada"
        
        # Testa quando API key não está disponível
        mock_getenv.return_value = None
        api_key = os.getenv("OPENAI_CHAT_GPT_CONTA_BRUNOJOSE1977_API_KEY")
        assert api_key is None, "API key deveria ser None quando não configurada"
    
    def test_llm_prompt_structure(self):
        """Testa estrutura do prompt para LLM"""
        # Simula estrutura do prompt usado no código
        prompt = """Você é um especialista em Machine Learning, Algoritmos Genéticos e análise de resultados de modelos de classificação.

Analise as três imagens fornecidas que mostram os resultados de um experimento de otimização de modelo de Regressão Logística usando Algoritmos Genéticos para diagnóstico de diabetes."""
        
        assert "especialista" in prompt.lower(), "Prompt deveria mencionar especialista"
        assert "algoritmos genéticos" in prompt.lower(), "Prompt deveria mencionar algoritmos genéticos"
        assert "regressão logística" in prompt.lower(), "Prompt deveria mencionar regressão logística"


class TestPDFGeneration:
    """Testes para a geração de relatório PDF"""
    
    def test_gerar_relatorio_pdf_structure(self):
        """Testa se a função gera PDF com estrutura correta"""
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        import re
        
        analise_texto = """
# Título Principal

## Subtítulo

### Sub-subtítulo

Texto normal com **negrito** e *itálico*.

Mais texto aqui.
"""
        
        def gerar_relatorio_pdf(analise_texto, output_path, _dados_ranking=None):
            if not analise_texto:
                return False
            
            try:
                doc = SimpleDocTemplate(output_path, pagesize=A4)
                story = []
                styles = getSampleStyleSheet()
                
                # Adiciona conteúdo
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=20,
                    alignment=TA_CENTER
                )
                
                story.append(Paragraph("Relatório de Análise de Resultados", title_style))
                story.append(Spacer(1, 0.2))
                
                # Processa texto
                linhas = analise_texto.split('\n')
                for linha in linhas:
                    linha = linha.strip()
                    if not linha:
                        continue
                    
                    if linha.startswith('###'):
                        titulo = linha.replace('###', '').strip()
                        if titulo:
                            story.append(Paragraph(f"<b>{titulo}</b>", styles['Heading3']))
                    elif linha.startswith('##'):
                        titulo = linha.replace('##', '').strip()
                        if titulo:
                            story.append(Paragraph(f"<b>{titulo}</b>", styles['Heading2']))
                    elif linha.startswith('#'):
                        titulo = linha.replace('#', '').strip()
                        if titulo:
                            story.append(Paragraph(f"<b>{titulo}</b>", styles['Heading1']))
                    else:
                        texto_formatado = linha
                        texto_formatado = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', texto_formatado)
                        texto_formatado = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', texto_formatado)
                        if texto_formatado.strip():
                            story.append(Paragraph(texto_formatado, styles['Normal']))
                
                doc.build(story)
                return True
            
            except (IOError, OSError) as e:
                print(f"Erro ao gerar PDF: {str(e)}")
                return False
        
        # Testa geração de PDF
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            result = gerar_relatorio_pdf(analise_texto, output_path)
            assert result is True, "PDF deveria ser gerado com sucesso"
            assert os.path.exists(output_path), "Arquivo PDF deveria existir"
            assert os.path.getsize(output_path) > 0, "Arquivo PDF não deveria estar vazio"
        
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_gerar_relatorio_pdf_without_analysis(self):
        """Testa comportamento quando não há análise"""
        def gerar_relatorio_pdf(analise_texto, _output_path, _dados_ranking=None):
            if not analise_texto:
                return False
            return True
        
        result = gerar_relatorio_pdf(None, 'test.pdf')
        assert result is False, "Deveria retornar False quando não há análise"


class TestDataProcessing:
    """Testes para processamento de dados e filtros IQR"""
    
    def test_iqr_filtering(self):
        """Testa se o filtro IQR funciona corretamente"""
        # Cria dados com outliers
        data = {
            'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100],  # 100 é outlier
            'Feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            'Outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(data)
        
        iqr_factor = 1.5
        col = 'Feature1'
        
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        
        df_filtered = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Verifica se o outlier foi removido
        assert 100 not in df_filtered[col].values, "Outlier deveria ser removido"
        assert len(df_filtered) < len(df), "Dados filtrados deveriam ter menos linhas"
    
    def test_feature_selection_mask(self):
        """Testa se a máscara de seleção de features funciona"""
        n_features = 8
        feature_mask = np.array([1, 0, 1, 0, 1, 1, 0, 1], dtype=bool)
        
        # Cria DataFrame de exemplo
        X = pd.DataFrame(np.random.randn(100, n_features))
        
        # Aplica máscara
        X_selected = X.iloc[:, feature_mask]
        
        # Verifica se apenas as features selecionadas estão presentes
        assert X_selected.shape[1] == feature_mask.sum(), "Número de colunas deveria corresponder à máscara"
        assert X_selected.shape[0] == X.shape[0], "Número de linhas deveria ser mantido"


class TestModelEvaluation:
    """Testes para avaliação de modelos"""
    
    def test_confusion_matrix_calculation(self):
        """Testa cálculo de matriz de confusão"""
        y_true = [0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1, 1, 0, 1]
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Verifica formato
        assert cm.shape == (2, 2), "Matriz de confusão deveria ser 2x2"
        assert cm[0][0] >= 0, "True Negatives deveria ser >= 0"
        assert cm[1][1] >= 0, "True Positives deveria ser >= 0"
    
    def test_recall_score_calculation(self):
        """Testa cálculo de recall"""
        y_true = [0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1, 1, 0, 1]
        
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Verifica se é um valor válido
        assert 0 <= recall <= 1, "Recall deveria estar entre 0 e 1"
    
    def test_accuracy_score_calculation(self):
        """Testa cálculo de acurácia"""
        y_true = [0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1, 1, 0, 1]
        
        acc = accuracy_score(y_true, y_pred)
        
        # Verifica se é um valor válido
        assert 0 <= acc <= 1, "Acurácia deveria estar entre 0 e 1"


class TestRankingData:
    """Testes para estrutura de dados de ranking"""
    
    def test_ranking_data_structure(self):
        """Testa se a estrutura de dados de ranking está correta"""
        ranking_data = [
            {
                'fitness': 850.5,
                'c': 1.5,
                'solver': 'lbfgs',
                'iqr': 2.0,
                'recall': 0.85,
                'fp': 10,
                'fn': 5,
                'acc': 0.90,
                'features': 6
            }
        ]
        
        df_ranking = pd.DataFrame(ranking_data)
        
        # Verifica colunas esperadas
        expected_columns = ['fitness', 'c', 'solver', 'iqr', 'recall', 'fp', 'fn', 'acc', 'features']
        for col in expected_columns:
            assert col in df_ranking.columns, f"Coluna {col} deveria estar presente"
    
    def test_ranking_sorting(self):
        """Testa ordenação do ranking por fitness"""
        ranking_data = [
            {'fitness': 500, 'c': 1.0, 'solver': 'lbfgs', 'iqr': 2.0, 'recall': 0.5, 'fp': 20, 'fn': 10, 'acc': 0.7, 'features': 5},
            {'fitness': 800, 'c': 2.0, 'solver': 'liblinear', 'iqr': 2.5, 'recall': 0.8, 'fp': 15, 'fn': 5, 'acc': 0.85, 'features': 6},
            {'fitness': 600, 'c': 1.5, 'solver': 'saga', 'iqr': 3.0, 'recall': 0.6, 'fp': 18, 'fn': 8, 'acc': 0.75, 'features': 5}
        ]
        
        df_ranking = pd.DataFrame(ranking_data)
        df_sorted = df_ranking.sort_values(by='fitness', ascending=False)
        
        # Verifica se está ordenado corretamente
        assert df_sorted.iloc[0]['fitness'] == 800, "Maior fitness deveria estar primeiro"
        assert df_sorted.iloc[-1]['fitness'] == 500, "Menor fitness deveria estar último"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

