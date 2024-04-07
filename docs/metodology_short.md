# Viri

- [Hugging Face](https://huggingface.co/)
- [Kaggle](https://www.kaggle.com/)
- [Članek SPROTNA ANALIZA SLIK VOZIL Z
METODAMI GLOBOKEGA UČENJA V
OGRODJU FLUTTER](https://rosus.feri.um.si/rosus2024/files/Shishkov-Sprotna.pdf)

# Overview

Naš projekt je namenjen uporabnikom, ki želijo dobiti pripravljen model za klasifikacijo slik iz različnih kategorij. Ta model bo sprejel poljubno sliko kot vhod in izdal pripadajočo kategorijo.

# Metodologija

Naša aplikacija bo sestavljena iz treh glavnih delov:

- Aplikacija za učenje modela za klasifikacijo slik
- Zaledna aplikacija, ki bo izvajala napovedi in klasifikacijo slik
- Mobilna aplikacija, ki bo uporabnikom omogočala zajem slik in njihovo napoved direktno v aplikaciji ali preko klica na zaledno aplikacijo

## Orodja v projektu

Načrtujemo uporabo naslednjih orodij:

- Za delo z umetno inteligenco bomo uporabili Python, obogaten s knjižnicami TensorFlow in/ali PyTorch.
- Za zaledno aplikacijo bomo tudi uporabili Python zaradi enostavne integracije naučenega modela v kodo. Za razvoj HTTP strežnika, ki bo omogočal komunikacijo prek API-ja, bomo uporabili ogrodje Flask ali Django. Za testiranje API-jev pa bomo uporabili aplikacijo Postman.
- Mobilno aplikacijo nameravamo razviti z ogrodjem Flutter, saj imamo v skupini izkušnje z njegovo uporabo. Za izvajanje napovedi modela pa bomo uporabili knjižnico TFLite.

## Načrt razvoja

Pripravili smo načrt inkrementalnega razvoja, ki nam bo omogočil učinkovito izdelavo projekta od začetka do konca.

Na začetku se osredotočamo na razvoj aplikacije za učenje modela, nato sledi razvoj zaledne aplikacije, in nazadnje mobilne aplikacije.

### Aplikacija za učenje modela

Gre za CLI aplikacijo, ki jo želimo uporabiti na močnem superračunalniku. V njej bomo implementirali celoten postopek učenja. CLI aplikacija bo prejela seznam kategorij, ki jih mora model pravilno klasificirati.

![CLI app architecture](/images/arhitektura_ai_horizontal.drawio.png)

### Zaledna aplikacija

Zaledna aplikacija je osrednji HTTP strežnik, ki sprejema zahteve od odjemalcev.

Načrtujemo vsaj tri končne točke (endpoints):

- list_trained_models: Vrne seznam vseh dostopnih modelov, ki jih uporabnik lahko uporabi za napovedovanje slik.
- train_model: Izvede zgoraj opisano CLI aplikacijo. Prejme seznam kategorij za učenje in vrne ime naučenega modela.
- classify_image: Ta končna točka izvaja napovedi slik in vrne rezultat klasifikacije.

### Mobilna aplikacija s Flutter

Mobilna aplikacija bo funkcionalnost strežnika vizualno prijetno prikazala. Poleg tega omogoča napovedovanje neposredno na napravi z uporabo knjižnice [TFLite](https://pub.dev/packages/tflite_flutter).

## Uporabniška imena za superračunalnik

- auser010 (Aleksandr Shishkov)
- auser011 (Mark Škof) 
- auser022 (Matija Bažec)
