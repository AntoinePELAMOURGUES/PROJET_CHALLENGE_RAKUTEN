<h1>Challenge Rakuten France Multimodal Product Data Classification</h1>

<img src="./img/rakuten.png" alt="image_rakuten" width="1000" height="350">

<h2>:diamond_shape_with_a_dot_inside:Contexte</h2>

<p>Ce challenge porte sur le thème de la classification multimodale (texte et image) des codes types de produits à grande échelle où l'objectif est de prédire le code type de chaque produit tel que défini dans le catalogue de Rakuten France.</p>
<p>Le catalogage des listes de produits via la catégorisation des titres et des images est un problème fondamental pour tout marché de commerce électronique, avec des applications allant de la recherche personnalisée et des recommandations à la compréhension des requêtes.</p>
<p>Les approches de catégorisation manuelles et basées sur des règles ne sont pas évolutives puisque les produits commerciaux sont organisés en plusieurs classes. Le déploiement d'approches multimodales serait une technique utile pour les entreprises de commerce électronique, car elles ont du mal à catégoriser les produits en fonction des images et des étiquettes des commerçants et à éviter les duplications, en particulier lorsqu'elles vendent à la fois des produits neufs et d'occasion auprès de commerçants professionnels et non professionnels, comme le fait Rakuten.</p>
<p>Les progrès dans ce domaine de recherche ont été limités en raison du manque de données réelles provenant de catalogues commerciaux réels. Le défi présente plusieurs aspects de recherche intéressants en raison de la nature intrinsèquement bruyante des étiquettes et des images des produits, de la taille des catalogues de commerce électronique modernes et de la distribution déséquilibrée typique des données.</p>

<h2>:diamond_shape_with_a_dot_inside:Objectif</h2>

<p>L'objectif de ce défi de données est la classification multimodale à grande échelle (texte et image) des données de produits en codes de types de produits. Par exemple, dans le catalogue Rakuten France, un produit avec une désignation ou un titre français « Klarstein Présentoir 2 Montres Optique Fibre » associé à une image et parfois à une description complémentaire. Ce produit est classé sous le code de type de produit 1500. Il existe d'autres produits avec des titres, des images et des descriptions possibles différents, qui se trouvent sous le même code de type de produit. Compte tenu de ces informations sur les produits, comme l'exemple ci-dessus, ce défi propose de modéliser un classificateur pour classer les produits dans son code type de produit correspondant.</p>

<h2>:diamond_shape_with_a_dot_inside:Data description</h2>

<p>Pour ce challenge, Rakuten France propose environ 99 000 listes de produits au format CSV, y compris le train (84 916) et l'ensemble de test (13 812). L'ensemble de données comprend les désignations de produits, les descriptions de produits, les images de produits et leur code de type de produit correspondant. Les données sont réparties selon deux critères, formant quatre ensembles distincts : formation ou test, entrée ou sortie.</p>

<ul type="disc">
  <li>X_train.csv : fichier d'entrée de formation</li>
  <li>Y_train.csv : fichier de sortie de formation</li>
  <li>X_test.csv : fichier d'entrée de test </li>
</ul>

<p>De plus, le fichier images.zip est fourni contenant toutes les images. La décompression de ce fichier fournira un dossier nommé images avec deux sous-dossiers nommés image_training et image_test, contenant respectivement des images de formation et de test. </p>

<h2>:diamond_shape_with_a_dot_inside:Resultats obtenus concernant le texte</h2>

<h3>Vectorisation par sac de mots</h3>

| Model          | Train Accuracy ± StdDev | Validation Accuracy ± StdDev | Train Weighted F1 Score ± StdDev | Validation Weighted F1 Score ± StdDev | Time                |
| ------------- | ---------------------- | ---------------------------- | ---------------------------------- | ------------------------------------ | ------------------- |
| K-Neighbors   | 69.5%                  | 59.5%                        | 71.2%                             | 61.3%                                | 5 seconds           |
| Logistic Regr.| 70.0%                  | 67.7%                        | 71.2%                             | 61.3%                                | 5 seconds           |
| Random Forest  | 71.1%                  | 70.9%                        | 71.7%                             | 71.5%                                | 4 ms                |
| XGBoost       | 82.9%                  | 72.9%                        | 83.7%                             | 73.7%                                | 1.5 minutes        |
| Neural Netw.  | 80.3%                  | 75.6%                        | 80.4%                             | 75.5%                                | 3 minutes           |
| Linear SVC    | 71.7%                  | 70.0%                        | 71.3%                             | 69.6%                                | 1 second            |
| Naive Bayes   | 67.8%                  | 66.0%                        | 65.2%                             | 63.3%                                | 0.1 seconds         |

<h3>Vectorisation par plongement lexical</h3>

| Model         | Accuracy | Top 3 Accuracy | Weight F1 |
| ------------- | -------- | -------------- | --------- |
| SVC           | 65%      | 84%            | 63%       |
| Logistic Regr.| 62%      | 81%            | 61%       |
| Neural Netw.  | 65%      | 84%            | 64%       |

<h3>LLM</h3>

| Metric        | eval\_loss | eval\_accuracy  | eval\_f1    | eval\_precision   |
| ------------- | ---------- | --------------- | ----------- | ----------------- |
| train         | 0.33       | 90.3%           | 88%         | 90%               |
| val           | 0.74       | 79.1%           | 76.9%       | 79.5%             |
| test          | 0.75       | 78.7%           | 75.9%       | 78.4%             |

<h2>:diamond_shape_with_a_dot_inside:Resultats obtenus concernant les images</h2>

| Model            | Accuracy | F1Score Weighted | Delay (ms/step) |
| ---------------- | -------- | --------------- | --------------- |
| VGG16            | 63.8%    | 63.2%           | 10              |
| EfficientNet B1   | 53.1%    | 53.7%           | 9               |
| EfficientNet V2   | 62%      | 61%             | 46              |

<h2>:diamond_shape_with_a_dot_inside:Resultats obtenus par fusion texte et image</h2>

| Model                                  | Merge Type | Text Score | Image Score | Fusion Score |
| ------------------------------------- | ---------- | ---------- | ----------- | ------------ |
| Bert + LSTM + InceptionV3               | Early      | 0.742      | 0.628       | 0.83         |
| LSTM + EfficientNetB4                   | Early      | 0.813      | 0.606       | 0.52\*       |
| Bert + EfficientNetV2L                  | Score level | 0.848      | 0.67        | 0.8704        |

<h2>:diamond_shape_with_a_dot_inside:Création d'une API</h2>

<p>Nous avons créé une API utilisant pour frontend Streamlit et pour backend Fastapi. Nous souhaitions ainsi finaliser le projet en ayant une interface proche de celle des sites de vente en ligne. L'utilisateur choisit sa langue, le modèle qu'il souhaite utiliser puis va saisir le titre et la description de l'objet qu'il souhaite vendre. Il doit ensuite uploader la photographie de l'objet mis en vente. En retour, il reçoit les 3 classes les plus probables correspondantes à son objet :</p>

<img src="./img/page1_app.PNG" alt="image_app_1">

<img src="./img/page2_app.PNG" alt="image_app_2">


<h2>:diamond_shape_with_a_dot_inside:Guide de démarrage rapide</h2>

Vous souhaitez vendre un objet ? Nous vous aiderons à choisir sa classe ! 

Suivez ces étapes simples pour lancer rapidement l'application en local.

##  🛠️ Prérequis

Assurez-vous d'avoir installé les éléments suivants sur votre machine :

1. [Python](https://www.python.org/) >= 3.9
2. [Docker Desktop](https://docs.docker.com/desktop/)

## :computer: Installation

### Clonage du repository

Clonnez ce repository sur votre machine locale :

```bash
git clone https://github.com/dongnold/dst-sept23-rakuten.git
```

## :wrench: Configuration de Git LFS

Notre repository contient des modèles Machine Learning volumineux gérés par Git LFS. Après avoir cloné le repository, assurez-vous d'être connecté à Git LFS pour pouvoir télécharger les modèles.

### Instructions :

1. Naviguez jusqu'à la racine du répertoire cloné :

```bash
cd {REPOSITORY}
```

2. Connectez-vous à Git LFS :

```bash
git lfs login
```

  Entrez vos identifiants GitHub lorsque demandé.

3. Téléchargez les modèles Machine Learning :

```bash
git lfs pull
```

4. Construction des images Docker
   
Naviguez jusqu'à la racine du répertoire cloné et construisez les images Docker nécessaires pour l'application Backend et Frontend :

```bash
cd {REPOSITORY}
docker-compose up --build
```

Ouvrez ensuite un navigateur Web et visitez http://localhost:8000 pour afficher l'interface Streamlit.

De plus, vous pouvez envoyer des requêtes HTTP vers notre API FastAPI en visitant http://localhost:8001. Consultez la documentation OpenAPI Swagger UI pour connaître les points de terminaison disponibles.

## :skull: Arrêt de l'application

Appuyez sur Ctrl+C dans le terminal pour arrêter l'application. Confirmez l'opération si nécessaire.

















