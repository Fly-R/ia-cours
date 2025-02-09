# Projet SMART - Rapport

L’objectif du projet SMART est développer un projet en Python capable d’utiliser la
Computer Vision pour reconnaitre automatiquement un ensemble défini de produits. La
détection se fera soit sur une image, soit sur une vidéo, soit sur le flux caméra d’un
ordinateur en temps réel.

## Les premiers tests

### Experiment n°1 ([real_exp_10_yolo_s](https://app.picsellia.com/0192f6db-86b6-784c-80e6-163debb242d5/project/01936425-704a-7d93-b1bb-1b1641812ba4/experiment/0194d7c2-3718-76f0-b280-26eb28edd7cc/))

#### Paramètres de l'entrainement
- Modèle : YOLOv11-S
- epochs : 400
- Paramètres par défaut

#### Résultats : 
| Métrique   | Valeur finale | Maximum atteint pendant l'entraînement   |
|------------|---------------|------------------------------------------|
| Précision  | 0.84          | 0.89                                     |
| Recall     | 0.65          | 0.70                                     |
| mAP50      | 0.76          | 0.77                                     |
| mAP50-95   | 0.62          | -                                        |

Les différentes loss pendant le training : 

| box_loss                                                         | cls_loss                                                         | dfl_loss                                                         |
|------------------------------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------|
| <img src="img/exp1/box_loss.png" width="350" height="300"> | <img src="img/exp1/cls_loss.png" width="350" height="300"> | <img src="img/exp1/dfl_loss.png" width="350" height="300"> |


Une analyse plus détaillée de la box loss montre que la validation loss commence à diverger de la training loss autour 
de la 30ème epoch. Cela suggère que le modèle commence à sur-apprendre, et qu'un arrêt précoce pourrait être envisagé 
pour éviter le sur-ajustement.

<img src="img/exp1/box_loss_annoted.png" width="350" height="300">

### Experiment n°2 ([real_exp_11_yolo_l](https://app.picsellia.com/0192f6db-86b6-784c-80e6-163debb242d5/project/01936425-704a-7d93-b1bb-1b1641812ba4/experiment/0194e272-62f8-7d4b-82ce-21cd06f6100e/))

#### Paramètres de l'entrainement
- Modèle : YOLOv11-L
- epochs : 50
- Paramètres par défaut

Les différentes loss pendant le training : 

| box_loss                                                   | cls_loss                                                   | dfl_loss                                                   |
|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|
| <img src="img/exp2/box_loss.png" width="350" height="300"> | <img src="img/exp2/cls_loss.png" width="350" height="300"> | <img src="img/exp3/dfl_loss.png" width="350" height="300"> |

On a choisi de tester ce modèle (un peu pour le fun) car c'est un modèle assez lourd. Comme dans l'experiment précédente,
on a vu qu'il n'était pas forcément nécessaire de faire beaucoup d'epochs, il serait possible de train celui "rapidement".

Finalement, le training a été plutôt long et les résultats ne sont pas très bons. La précision et le recall sont très faibles 
et meriteraient de faire plus d'epochs mais le cout en ressource est trop important pour le faire donc on peut laisser 
tomber ce modèle.

La matrice de confusion est aussi catastrophique

<img src="img/exp2/conf_matrix.png" width="350" height="300"> 


### Experiment n°3 ([real_exp_12_yolo_m](https://app.picsellia.com/0192f6db-86b6-784c-80e6-163debb242d5/project/01936425-704a-7d93-b1bb-1b1641812ba4/experiment/0194e4dd-d8d8-7061-ab50-9b34e8e59523/))

#### Paramètres de l'entrainement
- Modèle : YOLOv11-M
- epochs : 100
- Paramètres par défaut

#### Résultats : 
| Métrique   | Valeur finale | Maximum atteint pendant l'entraînement |
|------------|---------------|----------------------------------------|
| Précision  | 0.72          | 0.80 (12ème epoch)                     |
| Recall     | 0.42          | 0.49 (90ème epoch)                     |
| mAP50      | 0.53          | -                                      |
| mAP50-95   | 0.40          | -                                      |

Les différentes loss pendant le training : 

| box_loss                                                   | cls_loss                                                   | dfl_loss                                                   |
|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|
| <img src="img/exp3/box_loss.png" width="350" height="300"> | <img src="img/exp3/cls_loss.png" width="350" height="300"> | <img src="img/exp3/dfl_loss.png" width="350" height="300"> |  

On a choisi le modèle M qui semble être le plus prometteur. Les entraînements précédents avec ce modèle ont donné 
de bons résultats.
Dans un premier temps, on a testé avec 100 epochs pour observer les performances et envisager des améliorations ultérieures.

Toutes les validation loss approchent leurs training loss, mais la précision et le recall restent faibles et en pleine 
croissance. Il pourrait être pertinent de reprendre ce modèle pour lancer 100 nouvelles epochs et tenter de l'améliorer.

<img src="img/exp3/precision.png" width="350" height="300">
<img src="img/exp3/recall.png" width="350" height="300">

### Experiment n°4 et n°5 ([real_exp_13_yolo_m](https://app.picsellia.com/0192f6db-86b6-784c-80e6-163debb242d5/project/01936425-704a-7d93-b1bb-1b1641812ba4/experiment/0194e552-fd8a-7653-a191-b30dfbc1eb09/), [real_exp_14_yolo_m](https://app.picsellia.com/0192f6db-86b6-784c-80e6-163debb242d5/project/01936425-704a-7d93-b1bb-1b1641812ba4/experiment/0194e6dd-dae1-75cf-a3bd-bb0d53054008/))

#### Paramètres de l'entrainement n°4
- Modèle : experiment n°3
- epochs : 100
- patience : 20
- Paramètres par défaut

Les différentes loss pendant le training de l'experiment n°3 :

| box_loss                                                   | cls_loss                                                   | dfl_loss                                                   |
|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|
| <img src="img/exp4/box_loss.png" width="350" height="300"> | <img src="img/exp4/cls_loss.png" width="350" height="300"> | <img src="img/exp4/dfl_loss.png" width="350" height="300"> |  

<img src="img/exp4/conf_matrix.png" width="350" height="300"> 

#### Paramètres de l'entrainement n°5
- Modèle : experiment n°4
- epochs : 100
- patience : 20
- Paramètres par défaut

Les différentes loss pendant le training de l'experiment n°5 : 

| box_loss                                                   | cls_loss                                                   | dfl_loss                                                   |
|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|
| <img src="img/exp5/box_loss.png" width="350" height="300"> | <img src="img/exp5/cls_loss.png" width="350" height="300"> | <img src="img/exp5/dfl_loss.png" width="350" height="300"> |  

<img src="img/exp5/conf_matrix.png" width="350" height="300"> 


Les deux experiments ont été faites sur 100 epochs avec une patience de 20. L'experiment n°4 a fait les 100 epochs mais 
l'experiment n°5 n'a fait que 60. 
On voit que sur l'experiment n°4, les loss commencent déjà à s'éloigner et c'est encore plus marquant sur la n°5.

Finalement le résultat sur ces experiments n'est pas vraiment satisfaisant. 

On va tenter une nouvelle experiment depuis le modèle M de base en modifiant la learning rate pour limiter les variations 
au niveau des validation loss. Pour la learning rate finale, on peut la modifier légèrement car on voit que sur les 
training précédent, elles se stabilisaient vers la fin du training.

### Résumé des experiments
<img src="img/schemas/test-part-schema.png" >

## Des résultats intéressants

### Experiment n°6 ([real_exp_15_yolo_m_lr](https://app.picsellia.com/0192f6db-86b6-784c-80e6-163debb242d5/project/01936425-704a-7d93-b1bb-1b1641812ba4/experiment/0194e726-a3fe-7ae3-a710-baf712db1ed0/))

#### Paramètres de l'entrainement n°6
- Modèle : YOLOv11-M
- epochs : 200
- patience : 20
- lr0 = 0.00179
- lrf = 0.01518

Les valeurs des learning rates proviennent d'un début d'optimisation avec la méthode `model.tune()` de Ultralytics, bien 
que nous n'ayons plus les paramètres exacts utilisés pour cette phase d'ajustement. Les résultats obtenus semblent 
plutôt bon pour être une base de travail.

<img src="img/exp6/tune_graph.png" width="500" height="300">

Le fait de baisser la learning rate a permis de stabiliser un peu plus les variations des validations loss

| box_loss                                                   | cls_loss                                                   | dfl_loss                                                   |
|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|
| <img src="img/exp6/box_loss.png" width="350" height="300"> | <img src="img/exp6/cls_loss.png" width="350" height="300"> | <img src="img/exp6/dfl_loss.png" width="350" height="300"> |  

<img src="img/exp6/conf_matrix.png" width="350" height="300"> 
<img src="img/exp6/F1_curve.png" width="350" height="300">

Les premiers résultats sont plutôt satisfaisants avec une fitness de 77.6%, le meilleur score obtenu jusqu'à présent.

Cependant, le modèle rencontre des difficultés avec la reconnaissance des mikados et des capsules, souvent confondus 
avec l'arrière-plan. Une prochaine étape consisterait à poursuivre l'entraînement sur plus d'epochs et d'intégrer des 
techniques de data augmentation pour améliorer ces détections.


### Experiment n°7 ([real_exp_16_yolo_m_augm](https://app.picsellia.com/0192f6db-86b6-784c-80e6-163debb242d5/project/01936425-704a-7d93-b1bb-1b1641812ba4/experiment/0194e7f9-6718-76eb-814f-bb28a78a81d5/))

#### Paramètres de l'entrainement n°7
- Modèle : experiment n°6
- epochs : 100
- patience : 50
- lr0 = 0.000895
- lrf = 0.00759
- mixup = 0.3
- mosaic = 0.3

L'ajustement en divisant par 2 les learning rates vise à éviter d’effacer les progrès du modèle. L'ajout de mixup et 
mosaic permet d'augmenter la complexité des images d'entraînement, en combinant plusieurs images en une et en générant 
des mosaïques.

Les différentes loss pendant le training :

| box_loss                                                   | cls_loss                                                   | dfl_loss                                                   |
|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|
| <img src="img/exp7/box_loss.png" width="350" height="300"> | <img src="img/exp7/cls_loss.png" width="350" height="300"> | <img src="img/exp7/dfl_loss.png" width="350" height="300"> |  


<img src="img/exp7/conf_matrix.png" width="350" height="300">

Les améliorations sont globalement légères, sauf pour les Mikados, qui perdent 0.13 en score. 

| Classe         | Exp n°6 | Exp n°7 | Evolution |
|----------------|---------|---------|-----------|
| Mikado         | 1.0     | 0.87    | - 0.13    |  
| Kinder pingui  | 0.78    | 0.86    | + 0.08    |  
| Kinder country | 0.60    | 0.70    | + 0.10    |  
| Kinder tronky  | 0.80    | 0.78    | - 0.02    |  
| Tic Tac        | 0.95    | 0.95    | =         |  
| Sucette        | 0.72    | 0.75    | + 0.03    |  
| Capsule        | 0.56    | 0.56    | =         |  
| Pepito         | 0.77    | 0.80    | + 0.03    |  
| Bouteille      | 0.82    | 0.83    | + 0.01    |  
| Canette        | 0.76    | 0.74    | - 0.02    |  

| Experiment | Fitness | 
|------------|---------|
| exp 6      | 77.6 %  | 
| exp 7      | 78.4 %  |

Une prochaine experiment pourrait être envisagée en réutilisant le modèle de l'experiment 6 avec d'autres paramètres de 
data augmentation.

### Experiment n°8 ([real_exp_17_yolo_m_augm](https://app.picsellia.com/0192f6db-86b6-784c-80e6-163debb242d5/project/01936425-704a-7d93-b1bb-1b1641812ba4/experiment/0194e9ad-af59-70aa-b3f6-ff9284c81521/))

#### Paramètres de l'entrainement n°8
- Modèle : experiment n°6
- epochs : 30
- patience : 10
- lr0 = 0.000895
- lrf = 0.00759
- translate= 0.1
- mosaic = 0.1,
- scale=0.5, 
- shear=10, 
- flipud=0.5

Les différentes loss pendant le training :

| box_loss                                                   | cls_loss                                                   | dfl_loss                                                   |
|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|
| <img src="img/exp8/box_loss.png" width="350" height="300"> | <img src="img/exp8/cls_loss.png" width="350" height="300"> | <img src="img/exp8/dfl_loss.png" width="350" height="300"> |  

<img src="img/exp8/conf_matrix.png" width="350" height="300">

| Classe         | Exp n°6 | Exp n°8 | Evolution |
|----------------|---------|---------|-----------|
| Mikado         | 1.0     | 1.0     | =         |  
| Kinder pingui  | 0.78    | 0.86    | + 0.08    |  
| Kinder country | 0.60    | 0.70    | + 0.10    |  
| Kinder tronky  | 0.80    | 0.78    | - 0.02    |  
| Tic Tac        | 0.95    | 0.95    | =         |  
| Sucette        | 0.72    | 0.76    | + 0.04    |  
| Capsule        | 0.56    | 0.69    | + 0.13    |  
| Pepito         | 0.77    | 0.73    | - 0.04    |  
| Bouteille      | 0.82    | 0.85    | + 0.03    |  
| Canette        | 0.76    | 0.79    | + 0.03    | 

| Experiment | Fitness | 
|------------|---------|
| exp 6      | 77.6 %  | 
| exp 7      | 78.4 %  |
| exp 8      | 81.1 %  |


Avec ces paramètres, on obtient de bien meilleur résultat. Certes, il y a des classes qui ont diminué, mais la diminution
est bien faible par rapport aux gains sur les autres classes.

On pourrait essayer une dernière experiment en augmentant les paramètres de celle-ci et en supprimant la patience
pour voir ce que cela donne si l'on va au bout des 30 epochs sur le modèle de l'experiment 6.


### Experiment n°9 ([real_exp_18_yolo_m_augm](https://app.picsellia.com/0192f6db-86b6-784c-80e6-163debb242d5/project/01936425-704a-7d93-b1bb-1b1641812ba4/experiment/0194e9ec-cbdd-78d9-9e68-6f235e19a0b0/))

#### Paramètres de l'entrainement n°9
- Modèle : experiment n°6
- epochs : 30
- patience : 0
- lr0 = 0.000895
- lrf = 0.00759
- translate= 0.1
- mosaic = 0.1,
- scale=0.5, 
- shear=10, 
- flipud=0.5

<img src="img/exp9/conf_matrix.png" width="350" height="300">

| Classe         | Exp n°6 | Exp n°9 | Evolution |
|----------------|---------|---------|-----------|
| Mikado         | 1.0     | 0.87    | - 0.13    |  
| Kinder pingui  | 0.78    | 0.84    | + 0.06    |  
| Kinder country | 0.60    | 0.65    | + 0.05    |  
| Kinder tronky  | 0.80    | 0.78    | - 0.02    |  
| Tic Tac        | 0.95    | 0.84    | - 0.11    |  
| Sucette        | 0.72    | 0.69    | - 0.03    |  
| Capsule        | 0.56    | 0.59    | + 0.03    |  
| Pepito         | 0.77    | 0.76    | - 0.01    |  
| Bouteille      | 0.82    | 0.78    | - 0.04    |  
| Canette        | 0.76    | 0.72    | - 0.04    | 

| Experiment | Fitness | 
|------------|---------|
| exp 6      | 77.6 %  | 
| exp 7      | 78.4 %  |
| exp 8      | 81.1 %  |
| exp 9      | 75.2 %  |

Avec cette experiment, le modèle est bon que l'original.


### Résumé des experiments
<img src="img/schemas/result-part-schema.png">

## Conclusion
Nous avons décidé de conserver le modèle de l’expérimentation n°8, qui offre les meilleurs résultats obtenus jusqu’à 
présent. Il serait possible de poursuivre le fine-tuning pour encore améliorer les performances, mais le modèle actuel 
est déjà plutôt correct.

Une autre méthode aurait été d’utiliser `model.tune()` de Ultralytics de manière plus poussée pour rechercher 
automatiquement les meilleurs hyper-paramètres. Nous avons testé cette méthode, mais cette approche demande 
énormément de ressources, ce qui n'était pas envisageable dans notre cas (à moins de louer 4 RTX 4090, mais bon ... 💸🐀).

Finalement, partir des paramètres par défaut proposés par Ultralytics, qui sont vraiment bon, et ajouter des data 
augmentations nous a permis d’obtenir un modèle aux performances plutôt correctes sans nécessiter un tuning automatique 
trop coûteux.

|                  | Lien                                                                                                                                                                   | 
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Meilleur modèle  | [model_latest](https://app.picsellia.com/0192f6db-86b6-784c-80e6-163debb242d5/model/01936429-42c2-7152-8cd1-ba068fa9d87a/version/0194e9b9-36a7-7580-ae8d-b307e8a03c0d) | 
| Autre bon modèle | [model_latest](https://app.picsellia.com/0192f6db-86b6-784c-80e6-163debb242d5/model/01936429-42c2-7152-8cd1-ba068fa9d87a/version/0194c29a-b7fc-709f-8a81-af438610aa38) | 

