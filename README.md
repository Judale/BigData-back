# Big Data

Ce document explique comment installer et lancer le projet Flask sur macOS/Linux et Windows.

---

## Prérequis

- **Python 3.6+** installé
- **Git** (optionnel, pour le versionnage)
- (Optionnel) **Virtualenv** ou module `venv` de Python

---

## Installation des dépendances

### Sur macOS / Linux

```bash
# 1. Créer et activer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 2. Mettre à jour pip (recommandé)
pip install --upgrade pip

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Sur Windows (PowerShell)

```powershell
# 1. Créer et activer un environnement virtuel
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Mettre à jour pip (recommandé)
pip install --upgrade pip

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## Configuration des variables d’environnement

### Sur macOS / Linux

```bash
export FLASK_APP=app.py
export FLASK_ENV=development
```

### Sur Windows (PowerShell)

```powershell
$Env:FLASK_APP = "app.py"
$Env:FLASK_ENV = "development"
```

---

## Lancer l’application

```bash
flask run
```

- Par défaut, l’application est accessible à `http://127.0.0.1:5000/`
---

## Structure du projet (exemple)

```
mon-projet-flask/
├── app.py
├── requirements.txt
├── venv/
├── templates/
├── instance/
├── backend/
└── static/
```

