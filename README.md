# diabetes-classification

Esse projeto de **classficação** tem como objetivo prever o diagnóstico de diabetes dado um conjunto de informações de saúde de um paciente. Utilizei o **CRISP-DM** como metodologia de trabalho durante a execução do projeto. 

Após uma exploração do contexto do projeto, criei uma questão de negócio a ser resolvida: melhorar as métricas de classificação a fim de reduzir os custos por pacientes, com relação a um modelo de previsão simples (que também foi construído no projeto).

Na análise exploratória além das ferramentas do `Pandas` para determinar por exemplo, correlações entre as variáveis. Construi gráficos usando `Matplotlib` e `Seaborn`.

O principal problema a ser resolvido nos tratamentos dos dados foram os dados faltantes. Utilizei como ferramentas o `KNNImputer`, `KBinsDiscretizer`, `QuantileTransformer` nessa fase de tratamento.

Alguns dos modelos de ML utilizados foram `LogisticRegression`, `SVM`, `RandomForestClassifier` e ainda modelos de votação como o `VotingClassifier`. Além da separação **treino-teste**, usei **validação cruzada** para avaliar os modelos. Os modelos tiveram um ajuste fino com `GridSearchCV` e `RandomizedSearchCV`.

Construi **pipelines** para automatizar e organizar o fluxo dos dados utilizando `Pipeline`, `FunctionTransfomer` e `ColumnTransfomer`.

>> Ao final do projeto conseguimos uma melhora significativa das métricas, principalmente o **recall** e ainda uma redução de custo por paciente com o nosso modelo.
