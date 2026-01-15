--------------------------
## TechChallenger Fase 1
--------------------------

    Trabalho de entrega da Fase 2 da Pós Graduação IA para Devs da FIAP

    Integrantes do grupo
    --------------------

    -Adalberto Ferreira de Albuquerque Neto (RM368178)
        adalbertonet@outlook.com

    - Bruno José e Silva (RM367064)
        brunojose1977@yahoo.com.br

    - Elton de Souza Machado Simão (RM368289)
        tonsoumasi@gmail.com

    - Lucas Varisco Mendes Bezerra (RM368587)
        lucasv.mendes@hotmail.com

---------------------------
## Introdução e Objetivos
---------------------------

    Um grande hospital universitário busca implementar um sistema inteligente de suporte ao diagnóstico, capaz de auxiliar médicos e equipes clínicas na análise inicial de exames e no processamento de dados médicos.

    ### Na Fase 1
    
    Foi feito o desenvolvimento e a validação de um modelo de algoritmo preditivo baseado em Machine Learning para auxiliar no diagnóstico de diabetes. 

    O objetivo principal consiste no treinamento do algoritmo com dados históricos e prever, com base em medições de diagnóstico, se uma paciente tem diabetes.

    ## Na Fase 2

    Estamos utilizando Algoritmos Genéticos para explorar e maximizar o uso dos metaparametros em busca de maior otimização dos resultados.

    ### Integração com LLM para Análise Automática

    O programa agora inclui integração com LLM (GPT-4 Vision) para análise automática dos resultados. Após a execução dos algoritmos genéticos e geração dos gráficos, o sistema:

    1. Envia as três imagens geradas (evolução linear, matrizes de confusão e gráfico comparativo) para o LLM
    2. Solicita uma análise técnica detalhada dos resultados
    3. Gera automaticamente um relatório em PDF com a análise


-----------------------
## Dataset Utilizado
-----------------------

    O projeto utilizou o Dataset sobre diabetes.

    Fonte:  N. Inst. of Diabetes & Diges. & Kidney Dis.

    Amostras: O dataset é composto por 768 observações de pacientes.

    Este conjunto de dados é de origem do Instituto Nacional de Diabetes e Doenças Digestivas e Renais (National Institute of Diabetes and Digestive and Kidney Diseases). 

### Conteúdo

    Em particular, todas as pacientes aqui são do sexo feminino, com pelo menos 21 anos de idade e de ascendência indígena Pima.

    Mais detalhes sobre a tribo indígena Pima podem ser encontrados em: 

    - https://www.britannica.com/topic/Pima-people

    - https://www.kaggle.com/uciml/pima-indians-diabetes-database

-------------------------
## Guia de Configuração
-------------------------
# Criar um ambiente virtual 
python3 -m venv env

# Rodar o ambiente virtual
source ./env/bin/activate

# Instalar todas as dependencias
pip install -r REQUIREMENTS.txt

# Configurar a Chave API OPENAI CHATGPT

vi .env
  
OPEN_AI_CHATGPT_FIAP_TECHCHALLENGER_2_LLM_API_KEY='COLOQUE AQUI A SUA CHAVE OPENAI CHATGPT, EXEMPLO: sk-proj-kT7DzVWH-yNj9-mqrXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    
-------------------------
## Guia de Execução
-------------------------    
# Executar cobertura de testes
   
### rodar todos os testes 
   pytest test_diabetes_regressao_logistica.py -v
   
### Executar testes específicos
   pytest test_diabetes_regressao_logistica.py::TestDataPreparation -v
  
### Executar testes com cobertura
    pytest test_diabetes_regressao_logistica.py --cov=. --cov-report=html

# Programa principal
    diabetes_regressao_logistica_com_algoritmos_geneticos.py
    
--------------------------------------------------------------------------------------------
## Arquivos gerados pelo Programa e utilizado pelo LLM ChatGPT para gerar o relatório final
--------------------------------------------------------------------------------------------     
    grafico_barras_comparativo.png
    grafico_evolucao_linear.png
    matrizes_confusao_antes_depois.png

--------------------------------------------------------------------------------------------
## Relatório PDF geradopelo LLM ChatGPT com base nas imagens geradas pelo programa
--------------------------------------------------------------------------------------------   
    Relatorio_Resultado_TechChallenger2_Regressao_logistica_e_Algorimos_geneticos.pdf

