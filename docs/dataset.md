# Dataset

75 foto's van een zak met 29 onderdelen van de docenten, in 2 scènes. Alle foto's zijn met een telefoon van bovenaf
gemaakt en verkleind naar 1536 x 2048.

## De onderdelen

8 klassen. Op elke foto liggen dezelfde onderdelen, dus de aantallen zijn per foto ongeveer gelijk.

| Klasse | Egaal | Dambord | Kop |
|---|---|---|---|
| ring | 113 | 392 | nee |
| zeskantbout | 86 | 277 | ja |
| insteekkoppeling | 75 | 270 | ja |
| moer | 75 | 224 | nee |
| inbusbout | 76 | 221 | ja |
| koppelbus | 38 | 112 | nee |
| slotbout | 32 | 93 | ja |
| knopschroef | 19 | 0 | ja |

Knopschroef komt alleen op de egale ondergrond voor, 1 per foto. Daar zijn er dus maar 19 van.

## Scène 1: egale ondergrond

19 foto's (`ps_iphone-16-pro_IMG_6001` t/m `6019`) op een egale lichte ondergrond.

- 6001 t/m 6010: de onderdelen liggen grotendeels los.
- 6011 t/m 6019: de onderdelen liggen op een stapel, tegen en over elkaar.
- 6005 t/m 6013: hebben een verlaagde exposure waarmee slecht licht wordt gesimuleerd.

## Scène 2: dambord

56 foto's (`board_IMG_...`) van dezelfde onderdelen op een vel met een dambordpatroon. Dat is veel drukker dan de egale
ondergrond, en sommige foto's zijn schuin, donker of onscherp. De foto's zijn gemaakt in 6 opstellingen: steeds de onderdelen
anders neergelegd, en daarvan een paar foto's.

- A en B: 6202 t/m 6221
- C: 6222 t/m 6226
- D, donker en E: 6227 t/m 6244 en 6293 t/m 6305

## Labels

Per foto een JSON-bestand in `data/labels/`. Per onderdeel een geroteerde box (midden, lengte en breedte, hoek), en bij
bouten en koppelingen ook waar de kop zit. De foto's zijn in CVAT gelabeld; de dambordfoto's zijn eerst voorgelabeld met
een al getraind model.

- `occluded`: het onderdeel is niet helemaal zichtbaar. Dat geldt voor 184 van de 514 onderdelen op de egale ondergrond.
  Op het dambord heb ik dit niet gelabeld.
- `_negeer`: een gebied dat niet meetelt, bijvoorbeeld een onderdeel dat te ver bedekt is om te herkennen (31 keer, alleen op
  het dambord).

## Verdeling

Staat in `data/split.yaml`. De regel: train, val en test delen nooit een dambord-opstelling. Anders test je of het model
een opstelling herkent die het al gezien heeft, en niet of het werkt op een nieuwe.

| | Egaal | Dambord | Waarvoor |
|---|---|---|---|
| train | 11 | 20 (A en B) | leren |
| val | 2 | 5 (C) | instellingen kiezen |
| test | 6 | 31 (D, donker en E) | alleen het eindcijfer |

In elk deel zitten zowel losse als gestapelde foto's van de egale ondergrond. De testset heb ik pas aan het eind gebruikt, zodat het
eindcijfer niet mooier is dan het echt is.
