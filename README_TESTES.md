# Testes Unitários - Regressão Logística com Algoritmos Genéticos

Este documento descreve os testes unitários criados para o programa `diabetes_regressao_logistica_com_algoritmos_geneticos.py`.

## Estrutura dos Testes

Os testes estão organizados em classes que cobrem diferentes aspectos do programa:

### 1. `TestDataPreparation`
Testa a preparação e pré-processamento dos dados:
- Substituição de zeros por NaN nas colunas específicas
- Preenchimento de valores faltantes com a mediana
- Divisão de dados em treino e teste com estratificação

### 2. `TestFitnessFunction`
Testa a função de fitness do algoritmo genético:
- Estrutura e parâmetros da função
- Validação de soluções inválidas
- Cálculo correto do fitness

### 3. `TestImageEncoding`
Testa a codificação de imagens em base64:
- Codificação correta de arquivos de imagem
- Tratamento de erros quando arquivo não existe

### 4. `TestLLMAnalysis`
Testa a integração com o LLM (OpenAI):
- Análise bem-sucedida com API key válida
- Tratamento quando API key não está disponível
- Uso de mocks para evitar chamadas reais à API

### 5. `TestPDFGeneration`
Testa a geração de relatórios em PDF:
- Estrutura e formatação do PDF
- Processamento de texto markdown
- Tratamento quando não há análise disponível

### 6. `TestDataProcessing`
Testa processamento de dados:
- Filtro IQR para remoção de outliers
- Seleção de features usando máscaras booleanas

### 7. `TestModelEvaluation`
Testa avaliação de modelos:
- Cálculo de matriz de confusão
- Cálculo de recall
- Cálculo de acurácia

### 8. `TestRankingData`
Testa estrutura de dados de ranking:
- Estrutura correta dos dados
- Ordenação por fitness

## Instalação

Para instalar as dependências necessárias para os testes:

```bash
pip install -r REQUIREMENTS.txt
```

Ou instalar apenas o pytest:

```bash
pip install pytest pytest-cov
```

## Executando os Testes

### Executar todos os testes

```bash
pytest test_diabetes_regressao_logistica.py -v
```

### Executar testes específicos

```bash
# Executar apenas testes de preparação de dados
pytest test_diabetes_regressao_logistica.py::TestDataPreparation -v

# Executar apenas testes de função de fitness
pytest test_diabetes_regressao_logistica.py::TestFitnessFunction -v

# Executar um teste específico
pytest test_diabetes_regressao_logistica.py::TestDataPreparation::test_replace_zeros_with_nan -v
```

### Executar com cobertura de código

```bash
pytest test_diabetes_regressao_logistica.py --cov=. --cov-report=html
```

Isso gerará um relatório HTML em `htmlcov/index.html`.

### Executar com saída detalhada

```bash
pytest test_diabetes_regressao_logistica.py -v -s
```

O flag `-s` mostra as saídas de print durante os testes.

## Estrutura dos Testes

Cada teste segue o padrão AAA (Arrange, Act, Assert):

1. **Arrange**: Prepara os dados e configurações necessárias
2. **Act**: Executa a função ou código a ser testado
3. **Assert**: Verifica se o resultado está correto

## Mocks e Fixtures

Os testes utilizam:
- **unittest.mock**: Para mockar dependências externas (OpenAI API, arquivos, etc.)
- **pytest.fixture**: Para criar dados de teste reutilizáveis
- **tempfile**: Para criar arquivos temporários durante os testes

## Notas Importantes

1. **Dados de Teste**: Os testes não dependem do arquivo `datasets/diabetes.csv` real. Eles criam dados sintéticos para garantir isolamento e reprodutibilidade.

2. **API Externa**: Os testes que envolvem chamadas à API da OpenAI usam mocks para evitar chamadas reais e custos.

3. **Arquivos Temporários**: Os testes que criam arquivos temporários os removem automaticamente após a execução.

4. **Random State**: Os testes usam `random_state=42` para garantir reprodutibilidade.

## Melhorias Futuras

Algumas melhorias que podem ser implementadas:

1. **Testes de Integração**: Criar testes que executem o fluxo completo do programa
2. **Testes de Performance**: Adicionar testes para verificar tempo de execução
3. **Testes de Validação**: Validar formatos de saída (PDF, imagens)
4. **Testes Parametrizados**: Usar `@pytest.mark.parametrize` para testar múltiplos cenários
5. **CI/CD**: Integrar os testes em um pipeline de CI/CD

## Troubleshooting

### Erro: ModuleNotFoundError
Certifique-se de que todas as dependências estão instaladas:
```bash
pip install -r REQUIREMENTS.txt
```

### Erro: ImportError
Verifique se o Python está usando o ambiente virtual correto:
```bash
source env/bin/activate  # Linux/Mac
# ou
env\Scripts\activate  # Windows
```

### Testes falhando
Verifique se os dados de teste estão corretos e se as funções do módulo principal não foram alteradas de forma incompatível.

## Contribuindo

Ao adicionar novas funcionalidades ao programa principal, lembre-se de:
1. Criar testes correspondentes
2. Manter a cobertura de código acima de 80%
3. Seguir o padrão AAA nos testes
4. Documentar casos de teste complexos

