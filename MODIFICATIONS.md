# Modifications par rapport au code original

## 1. requirements.txt

**Ajouts :**
- `transformers==4.25.1` — version originale du projet (était non pinionné, incompatible avec les versions récentes qui requièrent torch >= 2.4)
- `scikit-learn` — manquant, utilisé dans `src/dataset.py` (`from sklearn.model_selection import train_test_split`)
- `tqdm` — manquant, utilisé dans `src/preprocess_swde.py`

**Installation supplémentaire (non dans requirements.txt) :**
- `lxml_html_clean` — `lxml` a séparé son module `clean` dans un package à part depuis une version récente. Nécessaire pour `src/html_utils.py` (`from lxml.html.clean import Cleaner`). À ajouter dans requirements.txt.

---

## 2. Bugs identifiés (non encore corrigés)

### Bug critique — `src/data_collator.py` lignes 133-150

`DataCollatorForDOMNodeMask.torch_call` ne passe pas les features DOM au modèle.

**Code actuel (bugué) :**
```python
def torch_call(self, examples):
    input_ids = [e["input_ids"] for e in examples]
    node_ids  = [e["node_ids"]  for e in examples]  # utilisé uniquement pour le masquage
    # parent_node_ids, sibling_node_ids, depth_ids, tag_ids → jamais récupérés
    ...
    return {"input_ids": inputs, "labels": labels}
    # node_ids, parent_node_ids, sibling_node_ids, depth_ids, tag_ids → tous absents
```

**Conséquence :** Le modèle reçoit `None` pour les 5 features DOM (P0–P4), qui sont remplacées par des valeurs de padding dans `DOMLMEmbeddings.forward()` (lignes 166–175 de `modeling_domlm.py`). Les `TreePositionEmbeddings` n'apprennent jamais les vraies positions du DOM. Le modèle s'entraîne comme un RoBERTa standard.

**Même bug présent dans :** `tf_call` (ligne 152) et `numpy_call` (ligne 171).

**Correction à appliquer :**
```python
dom_keys = ["node_ids", "parent_node_ids", "sibling_node_ids", "depth_ids", "tag_ids"]
batch = {"input_ids": inputs, "labels": labels}
for key in dom_keys:
    vals = [e[key] for e in examples]
    batch[key] = _torch_collate_batch(vals, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
return batch
```

---

### Bug 2 — `src/dataset.py` ligne 7

`SWDEDataset` n'accepte qu'un seul domaine à la fois (`domain="university"` par défaut). Le papier pré-entraîne sur les 8 domaines.

**Correction à appliquer :** Accepter une liste de domaines et agréger les fichiers.

---

### Bug 3 — Pages SWDE : BOM + déclaration XML

Les fichiers `.htm` du dataset SWDE contiennent un BOM (`﻿`) et une déclaration XML (`<?xml version="1.0"?>`) en début de fichier qui cassent `lxml.html.soupparser.fromstring()` dans `src/html_utils.py`.

**Erreur produite :**
```
ValueError: Invalid PI name 'b'xml''
```

**Correction à appliquer dans `src/preprocess_swde.py`** (avant d'appeler `extract_features`) :
```python
import re
html = html.lstrip('﻿')                   # supprimer BOM
html = re.sub(r'<\?xml[^>]*\?>', '', html)     # supprimer <?xml ...?>
```

---

### Bug 4 — `src/train.py` lignes 60-61

```python
bf16 = True,   # requiert GPU Ampere (A100, A4000, RTX 30xx+)
tf32 = True,   # requiert GPU Ampere uniquement
```

Si le GPU n'est pas Ampere, remplacer par `fp16 = True` (comme dans le notebook `train_mlm.ipynb`).

---

## 3. Nouveaux fichiers ajoutés

### `debug_pipeline.py`

Script de débogage qui trace le pipeline complet en 5 étapes :
1. Lecture d'un fichier HTML SWDE
2. `extract_features()` → contenu d'un `.pkl`
3. Entrée du DataCollator
4. Sortie du DataCollator (ce que le modèle reçoit réellement)
5. Vérification des features DOM manquantes

**Usage :**
```bash
python debug_pipeline.py
```
