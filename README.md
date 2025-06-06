# Operacionalização de Machine Learning - Projecto Individual - Rumos Bank Live

## Problema
The Rumos Bank é um banco que tem perdido bastante dinheiro devido à quantidade de créditos que fornece e que não são pagos dentro do prazo devido.

Depois do banco te contratar, como data scientist de topo, para ajudares a prever os clientes que não irão cumprir os prazos, os resultados exploratórios iniciais são bastante promissores!

Mas o banco está algo receoso, já que teve uma má experiência anterior com uma equipa de data scientists, em que a transição dos resultados iniciais exploratórios até de facto conseguirem ter algo em produção durou cerca de 6 meses, bem acima da estimativa inicial.

Por causa desta prévia má experiência, o banco desta vez quer ter garantias que a passagem dos resultados iniciais para produção é feita de forma mais eficiente. O objetivo é que a equipa de engenharia consegue colocar o vosso modelo em produção em dias em vez de meses!

## Estrutura
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

# 1 - Criar Repositório
Este projecto está disponivel no repositório: https://github.com/b-matos/OML_BM.git


# 2 - 'README.md' atualizado
Um ficheiro README.md foi criado para uma melhor compreensão da estrutura do projeto.


# 3 - Ambiente Conda
Para o projeto foi criado um ambiente Conda com todas as dependências e de seguida exportado:
```
% conda env export --file conda.yaml --no-builds
```

Para criar um ambiente com as mesmas configurações, usar o código:
````
% conda create -f conda.yaml
````

De modo a que fique ativo, é necessário executar:
```
% conda activate OML_Latest
```

Para desativar:
```
% conda deactivate
```

Deste modo, fica garantida a reprodutibilidade ao nível dos pacotes de Python usados.

# 4 - MLFlow Tracking Server
De modo a que se tenha reprodutibilidade ao nível do modelo, foi criado um MLFlow Tracking Server.

Para criar o MLFlow Tracking server, registei o modelo e os seus parametros, assim como as métricas de avaliação.

Ao longo do projecto foram usadas duas formas de aceder às runs: localmente e através de um servidor.

## Localmente
Não é necessário ter o UI a correr, bastando definir o caminho para guardar as runs nos respetivos notebooks:
```
uri = "../mlruns"

Path(uri).mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(uri)
```

No entanto, para visualizar as runs no frontend MLFlow é necessário executar:
```
mlflow ui --backend-store-uri ./mlruns
```
(Substituir ```./mlruns``` pela localização das runs)

Tipicamente a UI fica disponivel na porta 5000, sendo que para visualizar, basta aceder a: http://127.0.0.1:5000

## MLFlow Server

Para correr o serviço MLFlow Tracking Server localmente é necessário executar os seguintes comandos:
```
% mlflow ui --port 5001 --backend-store-uri ./mlruns --artifacts-destination ./mlruns   
```

(Substituir ```./mlruns``` pela localização das runs)

Neste caso a UI estara disponivel em: http://127.0.0.1:5001

Para guardar as runs recorrendo a este servidor, tem de se apontar para o caminho da seguinte forma nos notebooks:
```
from pathlib import Path

uri = "http://0.0.0.0:5001"

mlflow.set_tracking_uri(uri)
```

## Model Registry
Os modelos foram registados ao correr o notebook: `rumos_bank_lending_prediction_register_mlruns`

Nesse notebook são testados e registados os seguintes modelos:
- Logistic Regression
- KNN
- SVM
- Decision Tree
- Random Forest (*)
- Neural Networks

Uma vez que o modelo Random Forest ocupa demasiado espaço e não é possivel passar para o Github, para este projeto foi discartado tendo sido escolhido o Neural Networks como modelo a utilizar (@champion). O modelo @champion pode ser alterado no UI do MLFLow.

Para isso, foi registado um modelo com pipeline para que os dados não tenham de ser tratados antes de utilizar o modelo.

(* Random Forest foi excluido deste projeto devido ao tamanho ocupado pelo modelo)

# 5 - Serviço API
Para expor o modelo registado numa API foi utilizada a framework FastAPI, garantindo assim que o modelo usado possa ser reproduzido para aplicações e relatórios.

Para correr esta API, basta executar o seguinte comando:

```
% python src/app/main.py
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

Para analizar a documentação do serviço que ficou exposto, basta aceder a: http://127.0.0.1:5002/docs. Aqui podem ser testados todos os endpoints.

(A porta 5002 poderá ser configurável no ficheiro `config/app.json`)

# 6 - Testes
Neste projecto foram criados testes ao modelo (MLFlow) e ao serviço (FastAPI).

**test_model.py:**

*test_model_out_1*

    test prediction 1
<br>

*test_model_out_0*

    test prediction 0
<br>

*test_model_out_shape*

    test shape of prediction
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


## Testing
Para correr os testes, basta usar o comando:
```
% pytest
```
Este comando irá procurar os ficheiros python começados em `test_` e irá executá-los.

Nota: Tanto o [MLFlow Tracking Server](#4---mlflow-tracking-server) como o [Serviço API](#5---serviço-api) têm de estar ativos antes de executar o comando de teste..

# 7 - Serviço Conteinereizado
De modo a que seja garantida a reprodutibilidade ao nível do sistema operativo, foram utilizados contentores docker.

Foram criados dois ficheiros para contruir as imagens a serem utilizadas para correr o modelo e serviço.

Para criar a imagem do MLFlow Tracking Server é necessário correr o comando:
```
% docker build -t mlflow-tracking-server:latest -f Dockerfile.Model
```

Analogamente, para o serviço:
```
% docker build -t default-payment-prediction-service:latest -f Dockerfile.Service
```

Para correr os contentores:
```
% docker run -d mlflow-tracking-server
% docker run -d default-payment-prediction-service
```

Alternativamente, é possível executar apenas o comando abaixo para criar e correr os containers:
```
% docker compose up
```


# 8 - Pipeline
Por último foi definido um pipeline de cicd (no ficheiro `.github/workflows/cicd.yaml`) para que sejam garantidos testes minimos antes de ser feito o merge com o branch `main`.

Para o executar, basta criar um pull request para main, o pipeline será executado automaticamente.

Este pipeline é constituido pelos seguintes passos:
- Obter repositório
- Configurar Docker
- Iniciar serviços
- Criar ambiente para executar testes
- Health Check Service
- Health Check Model
- Executar testes
- Iniciar sessão no repositório
- Enviar imagem para o repositório