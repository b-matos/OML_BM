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

Para desativar:
```
conda deactivate
```


# 4 - MLFlow Tracking Server
Para criar o MLFlow Tracking server, registei o modelo e os seus parametros, assim como as métricas de avaliação.

Ao longo do projecto foram usadas duas formas de aceder às runs: localmente e através de um servidor.

## Localmente
Para visualizar as runs é necessário executar:
```
mlflow ui --backend-store-uri ./mlruns
```
(Substituir ```./mlruns``` pela localização das runs)

Tipicamente a UI fica disponivel na porta 5000, sendo que para visualizar, basta aceder a: http://127.0.0.1:5000

## MLFlow Server

Para correr o serviço MLFlow Tracking Server localmente é necessário executar os seguintes comandos:
```
### % MLFLOW_TRACKING_URI=./mlruns

% mlflow ui --port 5001 --backend-store-uri ./mlruns --artifacts-destination ./mlruns   
```

(Substituir ```./mlruns``` pela localização das runs)

Neste caso a UI estara disponivel em: http://127.0.0.1:5001


## Model Registry
Os modelos foram registados ao correr o notebook: `rumos_bank_lending_prediction_register_mlruns`

Nesse notebook são testados e registados os seguintes modelos:
- Logistic Regression
- KNN
- SVM
- Decision Tree
- Random Forest
- Neural Networks

Uma vez que o modelo Random Forest ocupa demasiado espaço e não é possivel passar para o Github, para este projeto foi discartado tendo sido escolhido o Neural Networks como modelo a utilizar (@champion).

Para isso, foi um modelo com pipeline, para que os dados não tenham de ser tratados antes de utilizar o modelo.


# 5 - Serviço API
Para expor o modelo registado numa API foi utilizada a framework FastAPI.

Para correr esta API, basta executar o seguinte comando:

```
python src/app/main.py
```
(Para que o serviço funcione, é necessário que o [MLFlow Tracking Server](#4---mlflow-tracking-server) esteja a correr)

Foram criados cinco endpoints para o serviço:

**/default_payment_prediction**

    Prediction endpoint that processes input data and returns a model prediction.
<br>

**/model_metrics**

    Endpoint to retrieve the metrics of the model.
<br>

**/model_params**

    Endpoint to retrieve the parameters of the model.
<br>

**/model_metadata**

    Endpoint to retrieve the metadata of the model.
<br>

**/health**

    Health check endpoint to verify if the service is running.
<br>

Para analizar a documentação do serviço que ficou exposto, basta aceder a: http://127.0.0.1:5002/docs

(A porta 5002 poderá ser configurável no ficheiro `config/app.json`)

# 6 - Testes
Neste projecto foram criados testes ao modelo (MLFlow) e ao serviço (FastAPI).

**test_model.py:**
- test_model_out_1: test prediction 1
- test_model_out_0: test prediction 0
- test_model_out_shape: test shape of prediction
<br>
<br>

**test_service.py:**

*test_default_payment_prediction*

    Test for the /default_payment endpoint with valid input data.
    It should return a prediction in the response.
<br>

*test_default_payment_prediction_invalid_data*

    Test for the /default_payment endpoint with invalid input data.
    It should return a 422 Unprocessable Entity status code.
<br>

*test_default_payment_prediction_missing_data*

    Test for the /default_payment endpoint with missing input data.
    It should return a 422 Unprocessable Entity status code.
<br>

*test_get_model_metrics*

    Test for the /model endpoint.
    It should return the model metrics.
<br>

*test_get_model_params*

    Test for the /model endpoint.
    It should return the model params.
<br>

*test_get_model_metadata*

    Test for the /model endpoint.
    It should return the model metadata.
<br>

# 7 - Serviço Conteinereizado



Para correr em modo conteinereizado, é necessário correr o seguinte comando:
```
docker compose up
```


# 8 - Pipeline