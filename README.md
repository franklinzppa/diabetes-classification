# diabetes-classification
<div align="center">
  
![diabetes](img/header.jpeg)
  
</div>

Esse projeto de **classificação** tem como objetivo prever o diagnóstico de diabetes a partir um conjunto de informações de saúde de um paciente. Utilizei o **CRISP-DM** como metodologia de trabalho durante a execução do projeto. Ao final, coloquei o modelo em produção.

## Entendimento do problema e questão de negócio

* Após uma exploração do contexto do projeto, criei uma **questão de negócio** a ser resolvida: melhorar as métricas de classificação a fim de reduzir os custos por pacientes, com relação a um modelo heurístico simples.

* Em especial, a principal métrica a ser otimizada nesse contexto de negócio é o **recall**. Durante o processo de modelagem, otimizaremos o AUC, para permitir que no final ajustemos o recall sem grandes perdas de precisão. Desejamos obter um recall de aproximadamente 80%.

* Também desejamos uma vez classificada a base de teste, ranquear os resultados e selecionar os 10% pacientes mais prováveis a serem diabéticos. 

* O modelo heurístico que criei para *baseline* se baseia em achar o melhor valor de corte de glicose para classificar pacientes diabéticas.


<div align="center">

### Métricas do modelo heurístico

Acurácia   | Precisão | **Recall** | F1
---------  | -------- | ------ | -----
75.8%      | 67.0%    | **60.2%**  | 63.4%

</div>

----------------------------


## Exploração dos dados e tratamento

* Na **análise exploratória** além das ferramentas do `Pandas` para determinar por exemplo, correlações entre as variáveis. Construi gráficos usando `Matplotlib` e `Seaborn`.

* Utilizei a **redução de dimensionalide** t-SNE para visualizar a separabilidade dos dados.

* O principal problema a ser resolvido nos tratamentos dos dados foram os dados faltantes. Utilizei como ferramentas o `KNNImputer`, `KBinsDiscretizer`, `QuantileTransformer` nessa fase de tratamento.

* Dei bastante atenção para evitar alguns erros comums que percebi em outros projetos, como o **data leakage** causado por fazer alterações nos dados antes da separação de treino e teste.

----------------------------

## Modelos de Machine Learning

* Alguns dos modelos de ML utilizados foram `LogisticRegression`, `SVM`, `RandomForestClassifier` e ainda modelos de votação como o `VotingClassifier`. Além da separação **treino-teste**, usei **validação cruzada estratificada de 5 folds** para avaliar os modelos visto que nosso dataset é pequeno. Os modelos tiveram um ajuste fino com `GridSearchCV` e `RandomizedSearchCV`.

* Obtive as **features importances** a partir dos modelos treinados. Features como *glucose*, *BMI* e *Age* se mostraram features importantes.

* Como técnicas para lidar com o desbalanceamento, testei técnicas como `NearMiss`, `SMOTE` e ajuste do `class_weight` dos modelos, sendo este último a solução utilizada.

* Construi **pipelines** para automatizar e organizar o fluxo dos dados utilizando `Pipeline`, `FunctionTransfomer` e `ColumnTransfomer`.

* Testei a seleção de features utilizando o `SelectKBest`.

* Ao final do projeto conseguimos uma melhora significativa das métricas, principalmente o **recall**. Em particular, reduzimos bastante o número de **falsos negativos**, o que é importante dentro do contexto do problema. Além disso, obtivemos uma redução de custo por paciente com o nosso modelo.

<div align="center">

### Métricas do modelo de Machine Learning (Logistic Regression)

AUC | Acurácia   | Precisão | **Recall** | F1
----| -----  | -------- | ------ | -----
85.4% | 76.3%      | 66.0%    | **79.0%**  | 72.0%

</div>

* O recall desejado foi obtido com o ajuste do threshold de classificação, que foi definido utilizando a análise da **curva ROC** e curvas **recall-precisão**. 

* Apresentamos a curva de ganho e a curva lift da nossa solução.

* Ranqueamos a base de teste e obtemos uma taxa de alvo no primeiro decil de 23%. Isso significa que os top 10% apontados pelo modelo compreende 23% dos pacientes diabéticos do conjunto de teste. Isso permite pensar em soluções aplicáveis de acordo com as capacidades.

* Além das melhores métricas, o modelo costruído se traduz em uma **economia de 42.4% por paciente**.

## Deploy

* O modelo (pipeline de tratamento + Log Reg) foi colocado em produção utilizando o **HugginFace** e **Gradio** e pode ser acessado nesse [link](https://huggingface.co/spaces/frank-ferreira/diabetes-clf-deploy).
