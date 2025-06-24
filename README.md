# Facility Location Optimization

Ovaj projekt implementira i poredi dvije klasične metode za rješavanje problema lokacije skladišta (Facility Location Problem):

- **Branch and Bound (B&B)**
- **Cutting Plane (CP)**

Cilj je pronaći optimalan izbor skladišta i način opskrbe kupaca tako da se minimiziraju ukupni troškovi — uključujući fiksne troškove otvaranja skladišta i transportne troškove.

## Opis

Simulacija koristi nasumično generisane ulazne podatke:
- Fiksni troškovi za svaku lokaciju skladišta
- Transportni troškovi između svake lokacije i kupca (bazirano na udaljenosti)
- Potražnja kupaca

U poređenju se analizira vremenska složenost oba pristupa.
## Pokretanje

Za pokretanje koda potrebno je imati Python 3 i sljedeće biblioteke:

```bash
pip install numpy scipy

