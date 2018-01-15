# Answers

## Question :

Pour la Bibliothèque Nationale de France, vous devez mettre au point un classifeur de page afin de prédire pour chaque page d’un manuscrit son type. Développez votre code python dans un fichier Examen/classify_pages.py. Décrivez les étapes de la mise au point dans un fichier Examen/reponses.md. Mettez à jour le fichier requirements.txt si besoin.

Prenez soin de bien définir les différents ensembles (train/validation/test). L’objectif est de pouvoir estimer la performance du classifieur (taux de classification correcte) sur des nouveaux manuscrits.

Donnez les résultats de toutes vos expériences, même les résutats négatifs.

Faites une analyse votre résultat final, telle qu’elle pourrait être transmise à votre client.

Commitez votre code et vos réponses dans git, mais ne commitez pas les images.


Packages used :
- PIL
- TQDM
- JOBLIB
- SCIKITLEARN

Possible optimizations :
- OpenCL
- CUDA for image resizing
- More ram for feature vectors

## Step 1
Loading our .jpg files or pickle file if the features have been pre-processed
NOTE : feat2.pkl is a subresolution of 8x8 px for the 38006 images

## Step 2
Depending of the argument given to the parser, we either execute Linear SVM or KNN classification
The --opt works for the KNN and tests values from 1 to 9 for k, giving the accuracies and the best one with the k value
The SVC is simple, C is by default 1, degree 3 and gamma is set to auto, there is still a possibility for an --opt option and doing a cross validation
with several folds to tune it

## Results
KNN gets 94% precision with k = 1 and SVC 88%