# Probabilistička matrična faktorizacija za sisteme preporuke u oblasti muzike 🎧🎶🎵🎼

Projekat ima za cilj da predstavi primjenu i uporedi metode probabilističke matrične faktorizacije
za sisteme preporuke u oblasti muzike. Korišćeni su podaci iz baze *hetrec2011-lastfm-2k* koja sadrži 
informacije o popularnosti raznih izvođača po njihovoj slušanosti od strane korisnika, društvenoj mreži korsnika,
kao i rejtinzima i žanrovima koje su korisnici dodijelili izvođačima na onlajn muzičkoj platformi *Last.fm*.
U tu svrhu implementirana su dva naučna rada na ovu temu - [PMF](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26419875.pdf)
i [KPMF](https://tinghuiz.github.io/papers/sdm12_kpmf.pdf).

Projekat je rađen u sklopu kursa Mašinsko učenje na Matematičkom fakultetu, Univerziteta u Beogradu.
U izradi projekta su učestvovale Andrea Grbić (1036/21) i Ana Marija Đurčinova (1100/21). 
## Sadržaj projekta
#### 1. Teorijski uvod
U ovom poglavlju uvodimo osnovne elemente teorije probabilističke matrične faktorizacije za sisteme preporuke.
Prvo objašnjavamo osnovu metodu, a potom i njena uopštenja kroz kernelizovanu metodu matrične faktorizacije (KPMF)
i ograničene kernelizovane metode matrične faktorizacije (cKPMF). \
Sistemi preporuka su jedna od najuspješnijih primjena mašinskog učenja u praksi. 
Korišćenjem sistema muzičkih preporuka, muzičke platforme mogu da predvide, a zatim ponude
odgovarajuće izvođače i pjesme svojim korisnicima na osnovu svojstava muzike koju su prethodno slušali sami korisnici.
Upotrebom naprednijih modela koji su implementirani u projektu moguće je koristiti i informacije o društvenoj mreži korisnika 
na muzičkoj platformi u cilju pronalaženja sličnosti u muzičkom ukusu korisnika i njihovih prijatelja. 
Mogućnost efikasnog pružanja personalizovanih muzičkih preporuka je od suštinskog značaja za konkurentnost bilo koje 
platforme za slušanje muzike. 

#### 2. Skup podataka i vizuelizacija
Predstavljen je i analiziran skup podataka *hetrec2011-lastfm-2k* koji je korišćen u okviru projekta.
U svrhu smanjenja memorijske i vremenske složenosti, izdvojen je podskup od hiljadu najslušanijih izvođača na muzičkoj platformi.
Prikazane su sumarne statisitke, kao i histogrami rejtinga svih izvođača i korisnika. 
Primijećeno je da su podaci neizbalansirani zbog čega je bilo neophodno implementirati kompleksnije metode od osnovne 
probabilističke matrične faktorizacije koja nije efikasna u slučaju takvih podataka. 

#### 3. Trening modela, analiza rezultata i upoređivanje implementiranih metoda
Istrenirano je ukupno 7 modela.
Dobijeni su rezultati i predikcije svih modela, što je grafički i tabelarno prikazano. 
Modeli su međusobno upoređeni i donesen je zaključak o tome koji je model najbolji. \
Rad je implementiran u sklopu Python skriptova koji se nalaze u foleru *codes*, a svi rezultati su prikazani u finalnoj Jupyter demo svesci pod nazivom *03_demo_run* 
koja približava funkcionalnosti.



##### 4. Lista paketa i pokretanje
Prije pokretanja samog koda u Jupyter sveskama *02_skup_podataka_vizuelizacija* i *03_demo_run* neophodno je instalirati nestandardne biblioteke sledećim komandama:

* pip install tabulate 
* pip install node2vec

Lista svih potrebnih paketa:
* pandas
* numpy
* node2vec
* sklearn
* tabulate


#### 5. Literatura
* [Probabilistic Matrix Factorization for Music Recommendation](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26419875.pdf) 
* [Kernelized Probabilstic Matrix Factorization: Exploiting Graphs and Side Information](https://tinghuiz.github.io/papers/sdm12_kpmf.pdf)
