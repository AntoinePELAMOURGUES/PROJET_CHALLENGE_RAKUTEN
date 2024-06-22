import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
import numpy as np
import cv2 # opencv
import datetime
import time
import tqdm
import html
import sys
import os
import re
from langdetect import detect
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from bs4 import BeautifulSoup
import sklearn
from glob import glob
import pathlib
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import itertools
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from glob import glob

from matplotlib.image import imread
import tensorflow as tf
import h5py
import pickle
from joblib import dump, load
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# FONCTION DE DETECTION DE LA LANGUE
def detect_my_langue(text):
    try:
        return detect(text)
    except:
        return 'inconnu'

# FONCTION DE NETTOYAGE DE NOTRE COLONNE TEXTE   
def pretraitement_texte(texte, langue):    
    preprocess_list = []
    lemmatizer = WordNetLemmatizer()             
    for mots, langues in zip(texte, langue) :
        if langues == "english":        
            stop_words = set(nltk.corpus.stopwords.words('english'))        
        else:
            stop_words = set(nltk.corpus.stopwords.words('french'))           
        # Suppression des balises HTML
        mots = BeautifulSoup(mots, 'html.parser').get_text()
        # Remplacement des apostrophes et ° par des espaces
        mots = mots.replace("'", " ")
        mots = mots.replace("°", " ")
        # Convertir en minuscules
        mots = mots.lower()        
        # Supprimer les chiffres en conservant années (4chiffres qui se suivent et n°de série)
        mots = re.sub(r'\b(?!(\d{4}|[A-Z]{3}\d{3})\b)\d+\b', '', mots)
        # Supprimer la ponctuation
        mots = "".join([i for i in mots if i not in string.punctuation])
        # Tokeniser
        tokenize_mots = word_tokenize(mots, language=langues)
        # Supprimer les stopwords  et prendre les mots de plus de 2 CAractères
        mots = [i for i in tokenize_mots if i not in stop_words and len(i) > 2] 
        # Lemmatisation
        mots = [lemmatizer.lemmatize(i) for i in mots]
        # Rejoindre les mots traités
        mots_clean = ' '.join(mots)
        preprocess_list.append(mots_clean)
    return preprocess_list

# FONCTION DE PRE-PROCESSING DU TEXTE
def pre_processing_texte(x_train_csv, y_train_csv):
    '''
    Fonction utiliser afin de pretraiter la partie texte de notre dataframe
    '''
    start_time = datetime.datetime.now()
    formatted_time = start_time.strftime("%H:%M:%S")
    print(f"Début du préprocessing à {formatted_time}")
    print("Nous chargeons les dataframes et récupérons les liens des mimages correspondantes afin de créer un dataframe preprocessé")
    # lecture de nos fichier x_train et y_train / recuperation des filepaths de nos images
    dfx = pd.read_csv(x_train_csv, index_col = [0])
    dfy = pd.read_csv(y_train_csv, index_col = [0])
    dfz = filepath_train() 
    print(f"Shape de notre dataframe avant préprocessing : {dfx.shape}")
    # concatenation de nos 2 dataframes par les index
    df_i = pd.merge(left = dfx, right = dfy, left_index = True, right_index = True, how = 'inner')
     # concatenation de nos 2 dataframes dfx et dfz - texte + filepath
    df = pd.merge(left = df_i, right = dfz, left_on = 'imageid', right_on = 'imageid', how = 'inner')
    # REMPLACER DES NaN de descirpion en valeur nulle
    df.fillna({'description':''}, inplace=True)
    df = df.dropna(axis=0)
    # CONCATENER NOS 2 COLONNES DE TEXTES 
    df['produit'] = df['designation'] + " " + df['description']
    # SUPPRESION DES 1414 VALEURS EN DOUBLE DANS NOTRE DF X_train
    duplicates = df.duplicated('produit')
    nb_duplicates = (duplicates.sum())
    print("Après avoir concaténé la colonne designation et description, nous avons constaté la présence de",
          str(nb_duplicates), "doublons. Ils sont supprimés !")
    df.drop(df[duplicates].index, inplace=True)
    # SUPPRESSION DES COLONNES DESIGNATION ET DEXCRIPTION
    df = df.drop(['description', 'designation'], axis = 1)
    # DETECTION DES LANGUES
    df['langue'] = df['produit'].apply(detect_my_langue)
    print("Il y a {} langues détectées au sein du datastet".format(df['langue'].nunique()))
    print("Nous conservons le français et l'anglais")
    df = df.loc[(df.langue == 'fr') | (df.langue == 'en')]
    # Passage de nos colonnes produits en string
    df[['produit','langue']] = df[['produit','langue']].astype(str)
    df['langue'] = df['langue'].str.replace('en', 'english', regex=False)
    df['langue'] = df['langue'].str.replace('fr', 'french', regex=False)
    df['produit_clean'] = pretraitement_texte(df.produit, df.langue)
    df = df.drop(['productid_x','imageid','productid_y', 'produit', 'langue'], axis =1)
    end_time = datetime.datetime.now()
    formatted_time_end = end_time.strftime("%H:%M:%S")
    preprocessing_duration = end_time - start_time
    print(f"Preprocessing terminé {formatted_time_end}. Durée totale: {preprocessing_duration}.")
    print(f"Shape de notre dataframe après preprocessing : {df.shape}")
    df.head()
    return df

# FONCTION RECUPERATION FILEPATH IMAGE

def filepath_train():    
    '''
    Fonction qui va récupérer les filepaths de nos images et les ajouter à notre df
    '''
    # Trouver tous les chemins vers les fichiers qui finissent par .jpg
    train_filepath = [os.path.join('./images/train', fn) for fn in os.listdir('./images/train') if fn.lower().endswith('.jpg')]
    # Remplacer les \\ par /
    train_filepath = list(map(lambda x : [x, x.split('/')[2]], train_filepath))
    # Créer un DataFrame pandas
    df_train = pd.DataFrame(train_filepath, columns=['filepath', 'nameLabel'])
    # Extraction des données images et produitid pour merging
    # TRAIN - Récupération de nos id et product pour joonction avec nos labels
    df_train['filepath_1'] = df_train['filepath'].str.replace("./images/train/image_", "")
    df_train['filepath_1'] = df_train['filepath_1'].str.replace("product_", "")
    df_train['filepath_1'] = df_train['filepath_1'].str.replace(".jpg", "")
    # CREATION D UNE COLONNE PRODUCTID ET D UNE IMAGE ID
    df_train[['imageid','productid']] = df_train.filepath_1.str.split('_', expand=True)
    df_train['imageid'] = df_train['imageid'].astype('int')
    df_train = df_train.drop(['nameLabel', 'filepath_1'], axis =1)
    return df_train

# MATRICE DE CONFUSION

def plot_confusion_matrice(y_test, y_pred, test_generator):    
    cnf_matrix = confusion_matrix(y_test, y_pred)
    classes = list(test_generator.class_indices.keys())
    plt.figure(figsize = (20,20))    
    plt.imshow(cnf_matrix, interpolation='nearest',cmap='Blues')    
    plt.title("Matrice de confusion")    
    plt.colorbar()    
    tick_marks = np.arange(len(classes))    
    plt.xticks(tick_marks, classes)    
    plt.yticks(tick_marks, classes)    
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):    
        plt.text(j, i, cnf_matrix[i, j], horizontalalignment="center",    
                 color="white" if cnf_matrix[i, j] > ( cnf_matrix.max() / 2) else "black")    
    plt.ylabel('Vrais labels')    
    plt.xlabel('Labels prédits')    
    plt.show()

# FONCTION DE PREDICTION EFFNETb1

def predict_effnet(test_generator, model_effnet):
    '''
    Fonction qavec comme arguments:
        test_generator : dataframe de test 
        model_effnet : modèle utilisé
        
    Fonction qui retourne:
        1. y_prob = les probabilités d'appartenir à chaque classe
        2. y_pred = l'index de la meilleure prédiction de chacune des images dans y_prob
        3. y_test = la classe réelle de chacune des images
        4. class_labels = les labels des classes
     '''   
    # Probabilités renvoyées par le modèle
    start_time = datetime.datetime.now()
    formatted_time = start_time.strftime("%H:%M:%S")
    print(f"Recupération des probabilités renvoyés par le modèke pour les {len(test_generator)} images à {formatted_time}")
    y_prob = model_effnet.predict(test_generator)
    end_time = datetime.datetime.now()
    formatted_time = end_time.strftime("%H:%M:%S")
    preprocessing_duration = end_time - start_time
    print(f"Fin de la récupération à {formatted_time}")
    print(f"Délai du modèle pour la prédiction :  {preprocessing_duration}")
    y_pred = tf.argmax(y_prob, axis=-1).numpy()
    # Vraies étiquettes
    y_test = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())   
    # Calcul accuracy
    print ("ACCURACY DU MODELE : ", accuracy_score(y_test, y_pred), end = "\n\n")
    # Calcul f1_score
    print("F1_SCORE_WEIGHTED : ", f1_score(y_test, y_pred, average = 'weighted'), end = "\n\n")
    # Rapport de classification
    print(classification_report(y_test, y_pred, target_names = class_labels), end = "\n\n")     
    # Matrice de confusion
    print("Matrice de confusion :", end = "\n\n")  
    plot_confusion_matrice(y_test, y_pred, test_generator)
    return y_prob, y_pred, y_test, class_labels


# FONCTION DE PREDICTION RANDOMFOREST

def predict_randomforest(model, X_test, y_test, save_filepath= None, class_labels=None):
    
    '''
    Fonction qui prend en argument :
        model : model entrainé
        X_test : test_features
        y_pred = l'index de la meilleure prédiction
        y_test = la classe réelle de chacune des images
        save_filepath : chemin d'accès pour la sauvegarde
        
    Fonction qui renvoie :
        y_pred = étiquettes prédites
        y_prob = probabilités estimées pour chaque instance de test
    
     '''   
    start_time = datetime.datetime.now()
    formatted_time = start_time.strftime("%H:%M:%S")
    print(f"Début de la prédiction à {formatted_time}")
    X_test = np.asarray(X_test)    
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    class_labels = list(model.classes_.astype(str))
    end_time = datetime.datetime.now()
    formatted_time = end_time.strftime("%H:%M:%S")
    preprocessing_duration = end_time - start_time
    print(f"Fin de la récupération à {formatted_time}")
    print(f"Délai du modèle pour la prédiction :  {preprocessing_duration}") 
    # Calcul accuracy
    print ("ACCURACY DU MODELE : ", accuracy_score(y_test, y_pred), end = "\n\n")
    # Calcul f1_score
    print("F1_SCORE_WEIGHTED : ", f1_score(y_test, y_pred, average = 'weighted'), end = "\n\n")
    # Rapport de classification
    print(classification_report(y_test, y_pred, target_names = class_labels), end = "\n\n")     
    # Matrice de confusion
    print("Matrice de confusion :", end = "\n\n")  
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (20,20))    
    plt.imshow(cnf_matrix, interpolation='nearest',cmap='Blues')    
    plt.title("Matrice de confusion")    
    plt.colorbar()    
    tick_marks = np.arange(len(class_labels))    
    plt.xticks(tick_marks, class_labels)    
    plt.yticks(tick_marks, class_labels)    
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):    
        plt.text(j, i, cnf_matrix[i, j], horizontalalignment="center",    
                 color="white" if cnf_matrix[i, j] > ( cnf_matrix.max() / 2) else "black")    
    plt.ylabel('Vrais labels')    
    plt.xlabel('Labels prédits')    
    plt.show()
    print(f"\n\n Sauvegarde de notre modèle dans le répertoire {save_filepath}")
    # Création automatique du répertoire si celui-ci n'existe pas encore
    if save_filepath is not None and not os.path.exists(save_filepath):
        os.mkdir(save_filepath)
    filename = 'randomForestClassifier.pkl'
    # Combinaison des chemins pour former le chemin absolu
    abs_path = os.path.join(save_filepath, filename)
    # Enregistrement du modèle dans le répertoire models
    if save_filepath is not None:
        with open(abs_path, 'wb') as f:
            pickle.dump(model, f)
    return y_pred, y_prob

# MULTIMODAL

# FONCTION QUI PERMET DE CALCULER ET RECUPERER LES PROBABILITES D APPARTENIR A UNE CLASSE POUR LE TEXTE - MODELE BERT

def prob_text(model_txt, texte_column):
    ''' 
    Fonction qui prend en entrée une colonne texte et renvoie une liste des probabilités d'appartenir à chacune de nos classes
    '''
    list_outputs_text = []
    for texte in texte_column:
        # Encoder le texte avec le tokenizer
        encoded_input = tokenizer(texte, return_tensors="pt", max_length=512)
        # Obtenir les probabilités pour chaque classe
        with torch.no_grad():
            output = model_txt(**encoded_input, return_dict=True)
            probabilities = output.logits.softmax(-1)
            list_outputs_text.append(probabilities)
    id2label = dict(model_txt.config.id2label)
    return list_outputs_text, id2label

# FONCTION QUI PERMET DE CALCULER ET RECUPERER LES PROBABILITES D APPARTENIR A UNE CLASSE POUR LES IMAGES

def prob_img(model_img, df_test):
    ''' 
    Fonction qui prend en entrée une colonne avec chemin d'accès des images et renvoie une liste des probabilités d'appartenir à chacune de nos classes
    '''
    list_outputs_img = []
    from tensorflow.keras.applications.vgg16 import preprocess_input
    test_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
    test_generator = test_data_generator.flow_from_dataframe(
                        dataframe=df_test,
                        x_col = "filepath",
                        y_col = "prdtypecode",
                        target_size= (224,224),
                        batch_size= 1,
                        class_mode= 'categorical',
                        shuffle = False, 
                        )
    # Probabilités renvoyées par le modèle
    y_prob = model_img.predict(test_generator, batch_size=1)  
    # Prédiction de la classe
    y_pred = tf.argmax(y_prob, axis=-1).numpy()
    class_labels = list(test_generator.class_indices.keys())   
    return y_prob, y_pred, class_labels

def best_classes(list_outputs_text, list_outputs_img, model_txt):
    '''
    Fonction qui va permettre d'additionner les prob texte et probs images et ressortir la meilleure
    '''
    list_outputs_text = [i.numpy() for i in list_outputs_text]
    id2label = dict(model_txt.config.id2label)
    best_labels = []
    for i, j in zip(list_outputs_text, list_outputs_img):
        sum_prob = i + j
        sum_pred = np.argmax(sum_prob, axis = -1)
        best_label = id2label[int(sum_pred)]
        best_labels.append(best_label)




   