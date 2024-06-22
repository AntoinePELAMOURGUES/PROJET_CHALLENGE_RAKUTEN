<h1>Challenge Rakuten France Multimodal Product Data Classification</h1>

<img src="./img/rakuten.png" alt="image_rakuten" width="400" height="200">

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
