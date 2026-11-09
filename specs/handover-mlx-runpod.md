# Handover : RAT-Bench sur macOS (MLX) avec repli RunPod

Statut : plan validé, aucun code écrit. Machine d'origine : Apple M1 Max, 32 Go RAM, arm64.

## Décisions prises

| Sujet | Choix |
|---|---|
| Modèle local | Llama-3.1-8B-Instruct, MLX 8 bits (~9 Go) |
| Attaquants | MLX local pour le développement, Claude (API) pour la mesure finale |
| Portée du premier run | Level 1, méthodes locales : `presidio`, `gliner`, `scrubadub`, `llama_basic` |
| Repli RunPod | Pod GPU complet, `vllm` réutilisé tel quel |
| GPU RunPod | RTX 4090 24 Go |
| Plafond de coût | 5 $ par run, arrêt automatique du pod |

## Blocages sur macOS (vérifiés dans le code)

1. `vllm` dans `pyproject.toml` : CUDA/Linux seulement. Utilisé par `pii_benchmark/anonymizers/vllm_engine.py`, `llama.py`, `llamarescriber.py`.
2. `pii_benchmark/attackers/llama.py` : pipeline `transformers`, modèle `meta-llama` protégé, lent sur MPS.
3. `pii_benchmark/anonymizers/textwash.py` : `cuda:0` codé en dur (option `cpu` existe).
4. `juliacall` / `juliapkg` (pour `correctmatch`) : Julia absent sur la machine d'origine.
5. Données absentes du repo : `benchmark/` et `data/pums/pums_pwgtp_sample_.csv` (Google Drive, voir README).
6. Le README cite `scripts/*/medical/all_levels.sh` : ces scripts n'existent pas. Seuls `scripts/{generate,anonymize,attack}/level_{1,2,3}.sh` existent.
7. `.python-version` demande 3.12.

## Étapes

1. Environnement : `vllm` sous marqueur `sys_platform == 'linux'`, ajout de `mlx-lm`, `uv sync` en Python 3.12, installer Julia, lancer `install_correctmatch.py`. (~20 min)
2. Backend MLX : créer `pii_benchmark/mlx_engine.py` (même interface que `vllm_engine.get_engine`, cache de processus). Brancher `llama.py` et `llamarescriber.py`. Créer `MlxAttacker`, l'enregistrer dans `pii_benchmark/attackers/get_attacker.py` (clé `mlx`). (~1 h)
3. Sélecteur `RAT_BACKEND` = `mlx` | `vllm` | `runpod`. Défaut : `mlx` sur macOS, `vllm` sur Linux avec CUDA. (~20 min)
4. `scripts/runpod/launch.sh` : créer le pod, `uv sync`, `rsync` du repo et de `data/pums/`, lancer le run, rapatrier `benchmark/`, arrêter le pod. Plafond 5 $, arrêt aussi en cas d'échec. (~1 h)
5. Données : télécharger `imperial-cpg/rat-bench` (Hugging Face) vers `benchmark/`. Déposer le CSV PUMS à la main.
6. Test rapide : level 1, 5 profils, `presidio` + `llama_basic` (MLX), attaquant `mlx`. Contrôle manuel de la sortie.
7. Run level 1 complet, en arrière-plan, reprise possible via les pickles existants.
8. Vérification : comparer les taux de ré-identification au papier et au leaderboard. Écart attendu : MLX 8 bits contre bf16 d'origine.

## À fournir sur la nouvelle machine

- CSV PUMS : `data/pums/pums_pwgtp_sample_.csv`
- Clé API Anthropic : `pii_benchmark/credentials.py` (copie de `credentials_example.py`, ignoré par git)
- Clé API RunPod : via `pass-cli`, jamais dans le repo
- Volume réseau RunPod (optionnel) pour garder le modèle entre deux pods
- Accès Hugging Face (modèle Llama protégé)

## Règle RunPod « local trop faible »

Bascule proposée si la RAM libre est inférieure à la taille du modèle, ou si le test rapide dépasse un temps limite. Aucune dépense sans accord de l'utilisateur.

## Fichiers CCPM

`ccpm.config.json` et `ccpm/*.json` sont générés par `/pm init`. Le champ `project.name` vaut `RAT-Bench-claude` (nom du dossier), alors que `ccpm/github.json` indique `rat-bench` (nom du dépôt).
