# Viri

- [Hugging Face](https://huggingface.co/)
- [Kaggle](https://www.kaggle.com/)

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

Gre za CLI aplikacijo, ki jo želimo uporabiti na močnem superračunalniku.

V njej bomo implementirali celoten postopek učenja. CLI aplikacija bo prejela seznam kategorij, ki jih mora model pravilno klasificirati.

Arhitektura CLI aplikacije bo naslednja:

- Seznam kategorij pretvorimo v posebne 'prompts', ki jih bomo uporabili za generiranje sintetičnih slik. To bomo izvedli preko LLM modela. Načrtujemo uporabo ChatGPT prek API klicev ali modela, dostopnega preko Hugging Face, kot je na primer gorilla-llm/gorilla-openfunctions-v2.
- Generirane 'prompte' vstavimo v model za pretvorbo besedila v sliko, ki bo ustvaril nabor podatkov za učenje modela klasifikacije. Zaradi velikega obsega podatkov načrtujemo uporabo superračunalnika za pospešitev generiranja slik.
- Na ustvarjenem naboru podatkov bomo izučili model klasifikacije slik. Načrtujemo uporabo modela MobileNetV3, zaradi njegove visoke natančnosti in optimiziranosti za mobilne naprave.
- Kot izhod te aplikacije bomo dobili uteži in model klasifikacije na osnovi MobileNetV3.

![CLI app architecture](/images/arhitektura_ai.drawio.png)


### Zaledna aplikacija

Zaledna aplikacija je osrednji HTTP strežnik, ki sprejema zahteve od odjemalcev.

Načrtujemo vsaj tri končne točke (endpoints):

- list_trained_models: Vrne seznam vseh dostopnih modelov, ki jih uporabnik lahko uporabi za napovedovanje slik.
- train_model: Izvede zgoraj opisano CLI aplikacijo. Prejme seznam kategorij za učenje in vrne ime naučenega modela.
- classify_image: Ta končna točka izvaja napovedi slik in vrne rezultat klasifikacije.

### Mobilna aplikacija s Flutter

To je odjemalec, ki komunicira s strežnikom in vizualno prikaže možnosti vsake končne točke, ki jih ponuja strežnik.

Uporabnik lahko v aplikaciji vidi vse dostopne modele, ki jih je naučila CLI aplikacija. Lahko zahteva učenje novega modela in izvede klasifikacijo slike. Poleg tega omogoča zajem slik prek aplikacije.

Ena od funkcionalnosti aplikacije je tudi napovedovanje neposredno na napravi. Načrtujemo prenos [TFLite](https://pub.dev/packages/tflite_flutter) modela v aplikacijo in njegovo uporabo za napovedi.

## Uporabniška imena za superračunalnik

- auser010 (Aleksandr Shishkov)
- auser011 (Mark Škof) 
- auserXXX (Matija Bažec)
