import asyncio
import requests
import pandas as pd
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import pathlib


# nltk.download('punkt_tab')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

url_base="https://www.amazon.com/s?i=videogames-intl-ship&bbn=16225016011&rh=n%3A468642%2Cp_123%3A184411%257C218247%257C220854%257C221831%257C248671%257C358345%257C381900%257C395698&dc&qid=1747921882&rnid=85457740011&ref=sr_nr_p_123_8&ds=v1%3AjUsyaOpJbf2kGD1P%2FZJAX0AoksAUZDpNwVbC14qa6KY&page={page_num}"


# Générer un User-Agent aléatoire
ua = UserAgent()
user_agent = ua.random
headers = {"User-Agent": user_agent}
# Nettoyage du texte
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text): 
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def get_text_if_not_none(elements) :
    return [el.text.strip() for el in elements] if elements else []

async def scrap_page(page):

    # name

    prod_el = "h2.a-size-medium.a-spacing-none.a-color-base.a-text-normal"
    produits = await page.locator(prod_el).all_inner_texts()
    produits = [i.strip() for i in produits]
    print("name :", produits)

    # price

    prix_el = "span.a-price-whole"
    prix_tot = await page.locator(prix_el).all_inner_texts() # class_ pour pas confondre avec une classe poo, nous c une classe css on veut 
    prix = [i.strip() for i in prix_tot]
    print("price :", prix)

    # Constructeur
    cons_el = "div.a-row.a-size-base.a-color-secondary"
    cons = await page.locator(cons_el).all_inner_texts()
    constructeur = [i.strip() for i in cons]
    constructeur = [i.replace('by','') for i in cons if i.startswith("by ")]
    print("construct :", constructeur)

    # avis
    reviews_el = "span.a-size-base.s-underline-text"
    reviews_number = await page.locator(reviews_el).all_inner_texts()
    reviews = [i.strip() for i in reviews_number]
    print("reviews :", reviews)


    print(len(produits), len(prix), len(constructeur), len(reviews)) 
    max_len = max(len(produits), len(prix), len(constructeur), len(reviews))  # Trouver la plus grande longueur

    # Compléter les listes trop courtes
    produits += [""] * (max_len - len(produits)) # produits += incrémente le resultat (il ajoute des élément à la liste plutot que de la remplacer comme l'uarait fait un = simple
    prix += [""] * (max_len - len(prix))
    constructeur += [""] * (max_len - len(constructeur))
    reviews += [""] * (max_len - len(reviews))
        

    # Stocker dans une seule ligne du DataFrame

    return pd.DataFrame({
        "Product name": produits,
        "Price": prix,
        "Constructeur": constructeur,
        "Avis": reviews
    })


# Tentative de scraping avec requests
dataframes = []
try:
    
    for page_num in range(1, 6):  # pour 5 pages max
        url = url_base.format(page_num=page_num)

        print(f"\n[INFO] Scraping page {page_num}: {url}")
        # Vérification du statut HTTP avec Requests
        response = requests.get(url, headers=headers) # on recupere notre url qu'on stocke dans une variable response 
        # response.encoding = "utf-8" # pr éviter les erreurs de caractères tels que les "À@" à la place des caratères avec acccent
        response.encoding = response.apparent_encoding # le mieux c de recuperer directement l'encoding de la page 

        if response.status_code == 200 :
            html = response.text
        
            f= open("page.html", "w") # f pour file et "w" pour write on se met en mode écriture et on écris dans le fichier "products.html"
            f.write(html) # bah on ecris du coup le contenue de notre variable html
            f.close() 

            soup = BeautifulSoup(html, 'html5lib')

            prod_el = soup.find_all("h2", {"class": "a-size-medium a-spacing-none a-color-base a-text-normal"}) # on recherche les h2 de classe... 
            produits = get_text_if_not_none(prod_el) # ".text" pour recuperer que le name sans la balise et tt...
            print("name :", produits)


            prix_el = soup.find_all("span", {"class": "a-price-whole"}) # on recherchep les span de classe... 
            prix = get_text_if_not_none(prix_el)
            print("price :", prix)

            cons_el = soup.find_all("div", {"class": "a-row a-size-base a-color-secondary"}) 
            constructeur = get_text_if_not_none(cons_el)
            constructeur = [i.replace('by','') for i in constructeur if i.startswith("by ")] #recupère uniquement le texte commençant par "by "
            print("construct :", constructeur)

            reviews_el = soup.find_all('span',{'class':'a-size-base s-underline-text'})
            reviews = get_text_if_not_none(reviews_el)
            print("reviews :", reviews)

            print(len(produits), len(prix), len(constructeur), len(reviews)) 
            max_len = max(len(produits), len(prix), len(constructeur), len(reviews))  # Trouver la plus grande longueur

            # Compléter les listes trop courtes
            produits += [""] * (max_len - len(produits)) # produits += incrémente le resultat (il ajoute des élément à la liste plutot que de la remplacer comme l'uarait fait un = simple
            prix += [""] * (max_len - len(prix))
            constructeur += [""] * (max_len - len(constructeur))
            reviews += [""] * (max_len - len(reviews))
        
            df_page = pd.DataFrame({
                "Product name": produits,
                "Price": prix,
                "Constructeur": constructeur,
                "Avis": reviews
            })
            dataframes.append(df_page)

            if not all([produits, constructeur, prix, reviews]):
                raise Exception("Contenu incomplet, probablement généré par JavaScript")
      
        else :
            print("ERREUR:", response.status_code) 
            raise Exception("Erreur HTTP")
            # Convertir en DataFrame
    df = pd.concat(dataframes, ignore_index=True) # Liste avec un seul products

    df.to_csv("products.csv", index=False, encoding="utf-8",  header=not pd.io.common.file_exists("products.csv"), mode='a')
except Exception as e:
    print(f"\n[INFO] Passage à Playwright : {e}")
    async def main():
        all_data = []
        # Configuration des options du navigateur
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(user_agent=user_agent)
            page = await context.new_page()
            playwright_data = []
            for page_num in range(1, 6):  # pour 5 pages max
                url = url_base.format(page_num=page_num)
                print(f"\n[INFO] Scraping page {page_num}: {url}")

                await page.goto(url, wait_until='domcontentloaded', timeout=60000)  # Attends le chargement complet de la page
                df_page = await scrap_page(page)
                playwright_data.append(df_page)

                html = await page.content()

                with open("page.html", "w", encoding="utf-8") as f:
                    f.write(html)
            


            # browser = await p.chromium.launch(headless=False)
            # context = await browser.new_context()
            # # Convertir le chemin local en URL fichier
            # path = pathlib.Path("page.html").resolve().as_uri()
            # page = await context.new_page()
            # # Charger le fichier local
            # await page.goto(path, wait_until='domcontentloaded', timeout=60000)
            # data = await scrap_page(page)


            # Fermer Playwright
            await browser.close()

            global df
            df = pd.concat(playwright_data, ignore_index=True)

            df.to_csv("products.csv", index=False, encoding="utf-8",  header=not pd.io.common.file_exists("products.csv"), mode='a')

    # Lancer l'event loop
    asyncio.run(main())
  

print("[INFO] Fin d'extraction \n[INFO] Début de l'analyse des données")


df = pd.read_csv("products.csv")
# Appliquer le nettoyage
df['Cleaned_Product_Name'] = df['Product name'].astype(str).apply(clean_text)

# Supprimer les doublons et lignes vides
df.drop_duplicates(inplace=True) #doublons
df.dropna(how='all', inplace=True)  # Ligne où tout est NaN
print (df)

# Nettoyage robuste des prix
df['Price'] = (
    df['Price']
    .astype(str)                             # S'assure que tout est string
    .str.replace(r'[^\d.]', '', regex=True)  # Garde uniquement chiffres et points
    .str.extract(r'(\d+\.?\d*)')[0]          # Extrait un nombre avec ou sans virgule flottante
    .astype(float)                           # Convertit en float
)
df = df.dropna(subset=['Constructeur'])  # Supprime les constructeur NaN
df = df.dropna(subset=['Price'])  # Supprime les prix NaN
df['Price'] = df['Price'].round(0).astype(int) # Pour arrondir

# Calculer le prix moyen par constructeur
df_mean_price = df.groupby('Constructeur', as_index=False)['Price'].mean().sort_values(by='Price', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Price', y= 'Constructeur', hue='Constructeur', legend=False, data=df_mean_price, palette='viridis')
plt.title("Prix moyen des produits par constructeur")
plt.xlabel("Prix moyen ($)")
plt.ylabel("Constructeur")
plt.tight_layout()
plt.show()

# Top marques les plus fréquentes
df['Constructeur'] = df['Constructeur'].str.strip()
print(df['Constructeur'].value_counts().head(3))

# Nettoyage des avis (extrait les nombres)
df['Avis'] = df['Avis'].str.replace(',', '', regex=False).str.extract(r'(\d+)').astype(float)
df['Avis'] = pd.to_numeric(df['Avis'], errors='coerce')

# Produits les plus p opulaires
top_reviewed = df.sort_values(by="Avis", ascending=False).head(3)
print(top_reviewed[['Cleaned_Product_Name', 'Avis', 'Price']])

# qualité prix : plus d'avis positifs pour un prix bas
df['Avis'] = df['Avis'].astype(float)
df['Score'] = df['Avis'] / df['Price']

constructeurs_scores = df.groupby('Constructeur')['Score'].mean().sort_values(ascending=False).head(3)
print(constructeurs_scores)

# TF-IDF vectorization
# Transformer les noms de produits textuels en vecteurs numériques exploitables par les modèles de machine learning 
vectorizer = TfidfVectorizer() # prend automatiquement les mots plus importants dans les textes
X = vectorizer.fit_transform(df['Cleaned_Product_Name']) # transforme chaque texte en un vecteur sparse 
print(X)

# Clustering KMeans
scores = [] # évalue la qualité du clustering, c’est-à-dire à quel point les produits d’un même cluster sont similaires entre eux et différents des autres clusters.
for k in range(2, 6):  # on determine k necessaire (nombre de clusters)
    model = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append((k, score))
#   Il renvoie un score entre -1 et 1 :

# ≈1 : très bon clustering (produits bien séparés en groupes distincts)

# ≈0 : les clusters se chevauchent

# <0 : mauvais clustering (produits plus proches d'autres clusters que du leur)

# Trouver le k avec le meilleur score silhouette
best_k = max(scores, key=lambda x: x[1])[0]
print(f"[INFO] Meilleur k choisi automatiquement : {best_k}")

# Appliquer KMeans avec le meilleur k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(X)
print(df)

# Réduction de dimension pour visualisation

X_dense = X.toarray()
n_samples = X_dense.shape[0]  # nombre de lignes/documents

if n_samples > 100:
    print(f"[INFO] Beaucoup de données ({n_samples}), PCA + t-SNE activés")
    X_reduced = PCA(n_components=50).fit_transform(X_dense)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X_reduced)
else:
    print(f"[INFO] Peu de données ({n_samples}), t-SNE simple activé")
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)# n_components=2 pr une projection en 2D
    X_embedded = tsne.fit_transform(X_dense)# X est une matrice TF-IDF (sparse matrix), donc .toarray() la convertit en matrice NumPy dense.
print(X_embedded)

# Ajout des dimensions TSNE au DataFrame
df['TSNE-1'] = X_embedded[:, 0] # axe x 
df['TSNE-2'] = X_embedded[:, 1] # axe y
print(df)

# Visualisation
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='TSNE-1', y='TSNE-2', hue='Cluster', palette='Set2', s=100)
plt.title("Visualisation 2D des clusters de produits (KMeans + TSNE)")
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()
