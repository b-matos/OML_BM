# Avaliação do módulo de Operacionalização de Machine Learning - Projecto Individual

## Rumos Bank going live

The Rumos Bank é um banco que tem perdido bastante dinheiro devido à quantidade de créditos que fornece e que não são pagos dentro do prazo devido. 

Depois do banco te contratar, como data scientist de topo, para ajudares a prever os clientes que não irão cumprir os prazos, os resultados exploratórios iniciais são bastante promissores!

Mas o banco está algo receoso, já que teve uma má experiência anterior com uma equipa de data scientists, em que a transição dos resultados iniciais exploratórios até de facto conseguirem ter algo em produção durou cerca de 6 meses, bem acima da estimativa inicial.

Por causa desta prévia má experiência, o banco desta vez quer ter garantias que a passagem dos resultados iniciais para produção é feita de forma mais eficiente. O objetivo é que a equipa de engenharia consegue colocar o vosso modelo em produção em dias em vez de meses!

## Avaliação

Os componentes que vão ser avaliados neste projecto são:

* Todas as alterações que fazem são trackeadas num repositório do github
* `README.md` atualizado
* Ambiente do projecto (conda.yaml) definido de forma adequada
* Runs feitas no notebook `rumos_bank_leading_prediction.ipynb` estão documentadas, reproduzíveis, guardadas e facilmente comparáveis
* Os modelos utilizados estão registados e versionados num Model Registry
* O melhor modelo está a ser servido num serviço - não precisa de UI
* O serviço tem testes
* O serviço está containerizado
* O container do serviço é built, testado e enviado para um container registry num pipeline de CICD

Garantam que tanto o repositório do github como o package no github estão ambos públicos!

### Data limite de entrega

01/05/2025

Deve ser enviada, até à data limite de entrega, um link para o vosso github (tem de estar público). Podem enviar este link para o meu email `lopesg.miguel@gmail.com` ou slack.

# Introdução
Este projecto foi divido em 8 pontos, sendo que os mesmos não foram executados de forma sequencial.

Os passos são:
- Criar Repositório
- Readme atualizado
- Ambiente Conda
- MLFlow Tracking Server
- Serviço API
- Testes
- Serviço conteinereizado
- Pipeline CICD

## Problema
The Rumos Bank é um banco que tem perdido bastante dinheiro devido à quantidade de créditos que fornece e que não são pagos dentro do prazo devido.

Depois do banco te contratar, como data scientist de topo, para ajudares a prever os clientes que não irão cumprir os prazos, os resultados exploratórios iniciais são bastante promissores!

Mas o banco está algo receoso, já que teve uma má experiência anterior com uma equipa de data scientists, em que a transição dos resultados iniciais exploratórios até de facto conseguirem ter algo em produção durou cerca de 6 meses, bem acima da estimativa inicial.

Por causa desta prévia má experiência, o banco desta vez quer ter garantias que a passagem dos resultados iniciais para produção é feita de forma mais eficiente. O objetivo é que a equipa de engenharia consegue colocar o vosso modelo em produção em dias em vez de meses!

# 1 - Criar Repositório
Este projecto está disponivel no repositório: https://github.com/b-matos/OML_BM.git


# 2 - 'README.md' atualizado
Um ficheiro Readme foi criado para uma melhor compreensão da estrutura do projeto.


# 3 - Ambiente Conda
Para o projeto foi criado um ambiente Conda com todas as dependências e de seguida exportado:
```
conda env export --file conda.yaml --no-builds
```

Para criar um ambiente com as mesmas configurações, usar o código:
````
conda create -f conda.yaml
````

De modo a que fique ativo, é necessário executar:
```
conda activate OML_Latest
```


# 4 - MLFlow Tracking Server
Para criar o MLFlow Tracking server, registei o modelo e os seus parametros, assim como as métricas de avaliação.

Para vizualizar o servidor executar:
```
mlflow ui --backend-store-uri ../mlruns
```

(`../mlruns` deverá ser substituída pela localização das runs)

## Model Registry
Os modelos foram registados ao correr o notebook: rumos_bank_lending_prediction.ipynb

Nesse notebook são testados e registados os seguintes modelos:
- Logistic Regression
- KNN
- SVM
- Decision Tree
- Random Forest
- Neural Networks

Após analise, foi escolhido o Random Forest como modelo a utilizar (@champion)

Mais tarde foi criado um modelo com pipeline, apesar do random forest não precisar de normalização dos dados.

## MLFlow Server


Para correr o serviço MLFlow Tracking Server localmente é necessário executar os seguintes comandos:
```
### % MLFLOW_TRACKING_URI=../mlruns

% mlflow server --port 5001 --backend-store-uri ../mlruns --artifacts-destination ../mlruns   
```

# 5 - Serviço API
Para expor o modelo registado numa API foi utilizada a framework FastAPI.

```
python src/app/main.py
```
(Para que o serviço funcione, é necessário que o MLFlow Tracking Server esteja a correr)

O serviço tem 4 endpoints:
- /default_payment_prediction
- /model_metrics
- /model_params
- /model_metadata

Para analizar a documentação do que ficou exposto, basta aceder a:
http://127.0.0.1:5002/docs

(A porta 5002 poderá ser configurável no ficheiro config/app.json)

# 6 - Testes
Neste projecto foram criados testes ao modelo (MLFlow) e ao serviço (FastAPI).

**test_model.py:**
- test_model_out
- test_model_dir
- test_model_out_shape

**test_service.py:**
- test_default_payment_prediction
- test_default_payment_prediction_invalid_data
- test_default_payment_prediction_missing_data
- test_get_model_metrics
- test_get_model_params
- test_get_model_metadata

# 7 - Serviço Conteinereizado

# 8 - Pipeline