# Analyse & Clustering de Produits Amazon

Projet de scraping Amazon et d’analyse sémantique des titres de produits, suivi d’un **clustering non supervisé** pour grouper les produits similaires.

---

## Objectifs

- Scraper automatiquement les produits (titre, prix, note).
- Nettoyer les textes à l’aide de techniques NLP (lemmatisation, stopwords...).
- Vectoriser les titres avec **TF-IDF**.
- Regrouper les produits par similarité avec **K-Means**.
- Visualiser les clusters en 2D avec **t-SNE**.

---

## Fonctionnalités

- Scraping hybride (`requests` + `Playwright`) pour gérer les contenus statiques et dynamiques.
- Nettoyage linguistique des titres (en français).
- Vectorisation avec sélection automatique des 1000 mots les plus pertinents.
- Détection automatique du meilleur `k` via le *silhouette score*.
- Visualisation des clusters (réduction de dimension avec PCA + t-SNE).

---

## Deux scripts disponibles

- `scraper_one_url.py` → Scraper une seule page Amazon.
- `scraper_multiple_pages.py` → Scraper plusieurs pages automatiquement.

---

## Prérequis

### Installation

Assurez-vous d’avoir Python **3.8+** puis installez les dépendances :

```bash
pip install requests beautifulsoup4 pandas nltk scikit-learn matplotlib seaborn fake-useragent playwright
playwright install
```
### Ressources NLTK (à installer une seule fois)

Dans le code, retirez le # devant ces lignes uniquement lors de la première exécution :

```python
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
```
Une fois les ressources installées, remettez les # pour éviter une nouvelle installation inutile.

## Mode Debug & Utilisation locale

Vous pouvez enregistrer la page HTML Amazon dans un fichier local page.html :

```python
# with open("page.html", "w", encoding="utf-8") as f:
#     f.write(await page.content())
```

Cela permet ensuite de scraper hors ligne, en lisant depuis ce fichier :

```python
# browser = await p.chromium.launch(headless=False)
# context = await browser.new_context()
# # Convertir le chemin local en URL fichier
# path = pathlib.Path("DKLE.html").resolve().as_uri()
# page = await context.new_page()
# # Charger le fichier local
# await page.goto(path, wait_until='domcontentloaded', timeout=60000)
# data = await scrap_page(page)
```

Pour scraper localement : activer open("page.html") et désactiver page.goto(...).

Pour scraper en direct depuis Amazon : activer page.goto(...) et désactiver open("page.html").

N’activez jamais les deux en même temps.

## Avertissement légal
Ce projet est à but éducatif. Le scraping de plateformes comme Amazon est soumis à leurs Conditions Générales d’Utilisation.
Utilisez ce script de manière responsable.
