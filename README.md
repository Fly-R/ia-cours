# Projet SMART - Guide d'utilisation

Le projet contient 2 pipelines:
- Une pipeline de training ([train.model.py](train.model.py))
- Une pipeline d'inférence ([exec.model.py](exec.model.py))

## Installation des dépendances

### Installation des packages Python
```bash
pip3 install -r requirements.txt
```
Ce projet utilise **Black**, **Flake8** et **pre-commit** pour garantir un code propre et structuré.

### Installation pour CUDA
- Mettre à jour les drivers NVIDIA avec la dernière version disponible
- Installer CUDA toolkit 12.8 ou autres

Mettre à jour la version de PyTorch selon la version de CUDA installée. Utiliser la commande pip disponible sur la page
de [PyTorch](https://pytorch.org/) pour obtenir la commande d'installation adéquate.
La version de CUDA toolkit doit toujours être supérieure à la version de PyTorch utilisée.
- CUDA toolkit 12.8 > PyTorch CUDA 12.6 ✅
- CUDA toolkit 12.5 < PyTorch CUDA 12.6 ❌

### Installation pour MPS
Rien à faire


## Configuration du projet
La configuration du projet se fait via les fichiers de configuration xml dans le dossier [config](config/).

### Configuration globale
`picsellia.config.xml` : Contient les informations de connexion à l'API Picsellia.
```xml
<config>
    <api_token></api_token>
    <organization_name></organization_name>
    <project_name>Groupe_8</project_name>
</config>
```
- `api_token` : Token d'authentification pour l'API Picsellia.
- `organization_name` : Nom de l'organisation sur Picsellia.
- `project_name` : Nom du projet sur Picsellia.

### Configuration du training
`train.config.xml` : Contient les informations pour paramétrer le training du modèle.
```xml
<config>
    <experiment_name></experiment_name>
    <dataset_path></dataset_path>
    <dataset_version_id></dataset_version_id>
    <send_metrics_on_epoch_end>False</send_metrics_on_epoch_end>
</config>
```
- `experiment_name` : Nom de l'experiment sur Picsellia. Si l'expérience n'existe pas, elle sera créée automatiquement.
- `dataset_path` : Localisation du dataset sur le disque une fois téléchargé et formatté.
- `dataset_version_id` : ID de la version du dataset sur Picsellia. Le dataset sera automatiquement attaché à l'experiment si ce n'est pas déjà le cas.
- `send_metrics_on_epoch_end` : Envoi les métriques à Picsellia à la fin de chaque epoch pour avoir un feedback du training en continu.
Peut être désactivé pour n'enregistrer les métriques qu'à la fin du training, notamment en cas de problème de timeout lors de l'envoi des métriques, ce qui stoppe le processus de training.

### Configuration de l'inférence
`exec.config.xml` : Contient les informations pour paramétrer le type d'inférence du modèle.
```xml
<config>
    <inference_type>images</inference_type>
    <experiment_name></experiment_name>
    <source></source>
</config>
```

- `inference_type` : Type d'inférence à réaliser. Types disponibles :
  - `images` : analyse les images du dossier renseigné dans `source`.
  - `video` : analyse la vidéo renseignée dans `source`.
  - `stream` : lance un stream de la webcam et analyse en temps réel.
- `experiment_name` : Nom de l'experiment sur Picsellia pour récupérer le modèle associé.
- `source` : Localisation de la source pour l'inférence. Soit un chemin vers un dossier d'images ou une vidéo.


## Utilisation

### Training

Une fois la configuration faite, il suffit de lancer le script `train.model.py` pour lancer le training du modèle.
```bash
python3 train.model.py
```

Etapes du training:
- Créer l'experiment sur Picsellia si il n'existe pas.
- Associe le dataset à l'experiment si ce n'est pas déjà le cas.
- Génère le dataset en local s'il n'existe pas déjà.
  - Génère 3 splits : train, val et test. Selon un ratio 60-20-20.
  - Télécharge les images et annotations depuis Picsellia.
  - Organise les images et annotations selon la structure YOLO.
  - Génère le fichier de configuration YOLO pour le training.
- Détecte le type de device pour le training (`CUDA`, `MPS` ou `CPU`).
- Lance le training du modèle.
- Valide le modèle sur le split de test et envoi les evaluations dans l'experiment Picsellia.

Métriques :
- `precision` : Pourcentage d’objets correctement détectés parmi toutes les détections du modèle. Une précision
élevée signifie que le modèle fait peu d’erreurs en détectant des objets qui n’existent pas (fausses détections)
- `recall` : Pourcentage d’objets réellement présents qui ont été détectés par le modèle. Un recall élevé
signifie que le modèle ne manque presque aucun objet
- `mAP50` : Moyenne des précisions obtenues pour toutes les classes d’objets, en considérant une
prédiction correcte si l’intersection entre la boîte détectée et la boîte réelle est d’au moins 50%
- `mAP50_95` : Similaire à mAP50, mais il prend en compte plusieurs niveaux de précision en variant le seuil
d’intersection (IoU) entre 50% et 95%. Cela donne une évaluation plus complète de la qualité du modèle
- `box_loss` : Mesure l’erreur dans la position et la taille des boîtes détectées par rapport aux vraies boîtes. Plus
la perte est faible, plus la détection des objets est précise
- `cls_loss` : Indique si le modèle a bien identifié la classe des objets détectés. Une grande perte signifie que le
modèle fait souvent des erreurs en confondant les objets
- `dfl_loss` : Aide à améliorer la précision des boîtes détectées en affinant leur position. Cette métrique est utilisée
dans des modèles avancés comme YOLOv8 pour rendre la détection encore plus précise

### Inference

Pour utiliser un modèle présent sur Picsellia, il suffit de lancer le script `exec.model.py` avec la configuration adéquate.
```bash
python3 exec.model.py
```

Etapes de l'inférence :
- Récupère la dernière version du modèle associé à l'experiment sur Picsellia et la télécharge en local dans le dossier `./models/{experiment_name}/best.pt`.
- Détecte le type de device pour l'inférence (`CUDA`, `MPS` ou `CPU`).
- Lance l'inférence selon le type renseigné
  - Stream : ouvre une fenetre en affiche le flux de la webcam avec la détection en temps réel.
  - Vidéo : ouvre une fenetre et affiche la vidéo avec la détection en temps réel.
  - Images : analyse les images dans le dossier renseigné et place les bounding boxes, puis les sauvegarde dans le dossier
  `./exec/{experiment_name}/predict`.
