import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.domlm import DOMLMConfig
from src.preprocess import extract_features
from src.data_collator import DataCollatorForDOMNodeMask
from transformers import AutoTokenizer

HTML_FILE = ROOT / "data/swde_html/university/university-collegeprowler(2000)/0000.htm"
CONFIG_FILE = ROOT / "domlm-config/config.json"

# ── ETAPE 1 : lecture du HTML ──────────────────────────────────────────────
print("\n" + "="*60)
print("ETAPE 1 — Lecture du fichier HTML")
print("="*60)
with open(HTML_FILE, "r") as f:
    html = f.read()
print(f"Taille du HTML : {len(html)} caractères")
print(f"Extrait du HTML : {html[:300]}...")

# Nettoyage nécessaire pour les pages SWDE : BOM + déclaration XML
import re
html = html.lstrip('﻿')                          # supprimer BOM
html = re.sub(r'<\?xml[^>]*\?>', '', html)            # supprimer <?xml ...?>
print(f"Après nettoyage  : {html[:300]}...")

# ── ETAPE 1b : vérifier get_cleaned_body ──────────────────────────────────
from src.html_utils import get_cleaned_body
body = get_cleaned_body(html)
print(f"get_cleaned_body() retourne : {body}")
if body is None:
    print("  → body est None sur ce fichier, on utilise un HTML de test à la place")
    html = """<html><body>
        <div class="movie-info">
            <h2 class="movie-title">Blade Runner <span class="release-year">(1982)</span></h2>
            <div class="director">Directed by <a href="#">Ridley Scott</a></div>
            <span class="genres">Genres - Science Fiction</span>
        </div>
    </body></html>"""
    print(f"HTML de remplacement utilisé ({len(html)} caractères)")

# ── ETAPE 2 : extract_features → ce que le .pkl contiendrait ──────────────
print("\n" + "="*60)
print("ETAPE 2 — extract_features() → contenu d'un .pkl")
print("="*60)
config = DOMLMConfig.from_json_file(str(CONFIG_FILE))
subtrees = extract_features(html, config)
print(f"Nombre de sous-arbres générés : {len(subtrees)}")

exemple = subtrees[0]
print(f"\nClés présentes dans un exemple : {list(exemple.keys())}")
for key, val in exemple.items():
    print(f"  {key:25s} → longueur={len(val)}, premiers éléments={val[:5]}")

# ── ETAPE 3 : ce que le DataCollator reçoit ───────────────────────────────
print("\n" + "="*60)
print("ETAPE 3 — Entrée du DataCollator (batch de 2 exemples)")
print("="*60)
batch_input = subtrees[:2]
print(f"Clés dans chaque exemple du batch : {list(batch_input[0].keys())}")

# ── ETAPE 4 : ce que le DataCollator retourne ─────────────────────────────
print("\n" + "="*60)
print("ETAPE 4 — Sortie du DataCollator (ce que le modèle reçoit)")
print("="*60)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
collator = DataCollatorForDOMNodeMask(tokenizer=tokenizer, mlm_probability=0.15)
batch_output = collator(batch_input)
print(f"Clés retournées au modèle : {list(batch_output.keys())}")
print()
for key, val in batch_output.items():
    print(f"  {key:25s} → shape={val.shape}")

# ── ETAPE 5 : ce qui manque ───────────────────────────────────────────────
print("\n" + "="*60)
print("ETAPE 5 — Features DOM manquantes")
print("="*60)
dom_features = ["node_ids", "parent_node_ids", "sibling_node_ids", "depth_ids", "tag_ids"]
for feat in dom_features:
    present = feat in batch_output
    print(f"  {feat:25s} → {'✅ présent' if present else '❌ ABSENT — le modèle recevra None → padding'}")
