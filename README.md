# quantizerp : codes compagnons

Codes qui produisent toutes les figures numeriques du document
*Quantification en commande : de la dimension finie aux EDP*.

## Installation

```
pip install -r requirements.txt
```

## Utilisation

Tout regenerer d'un coup :

```
python run_all.py
```

Ou un script a la fois :

```
python quantized_state_feedback.py
```

Chaque script ecrit ses sorties dans `../figs/` (cree automatiquement).
Aucun argument, aucune configuration : les parametres sont en tete de fichier.

## Contenu

| Fichier | Section du document | Ce qu'il produit | Duree |
|---|---|---|---|
| `scalar_quantized_demo.py` | 1.8 | `scalar_quantized_demo.png` : systeme scalaire instable, trois valeurs de mu | < 5 s |
| `quantized_state_feedback.py` | 2.4 | `quantized_state_feedback.png` : systeme d'ordre 2, loi hybride a deux phases, trois scenarios ; imprime aussi les constantes certifiees kappa_P, gamma, Omega, T | < 5 s |
| `make_tradeoff_fig.py` | 2.4 | `tradeoff_eps.png` : compromis marge/vitesse, Omega, T et taux garanti en fonction d'epsilon | < 5 s |
| `transport_backstepping_demo.py` | Annexe A | `transport_backstepping_demo.png` : EDP de transport, boucle ouverte instable contre backstepping nominal | < 10 s |
| `transport_quantized_control.py` | 3.2 | `transport_quantized_control.png` : EDP de transport bouclee par backstepping quantifie, schema decentre amont | ~ 1 min |
| `delay_quantized_predictor.py` | 3.3 | `delay_quantized_predictor.png` et quatre fichiers `d_ret_*.txt` : cascade EDO-EDP avec retard d'entree, predicteur quantifie, reglage certifie contre reglage pratique | ~ 1 min |

## Trois pieges d'implementation

1. **CFL.** Avec un schema decentre amont, `dt <= dx` est obligatoire. Au-dela le
   schema explose et l'on croit a tort a une instabilite du systeme.
2. **Plancher sur mu.** On divise par `mu` a chaque pas de temps : il faut une
   borne inferieure, ne serait-ce que numerique.
3. **Ordre des operations.** Quantifier puis integrer (`integrale de k*q(u)`,
   mesures quantifiees) et integrer puis quantifier (`q(integrale de k*u)`,
   entree quantifiee) sont deux problemes distincts, regis par deux theoremes
   differents. Les scripts implementent le premier.

## Licence

Les codes sont libres de reutilisation. Merci de citer le document si vous vous
en servez dans un travail publie.
