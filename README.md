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
U ovom poglavlju uvodimo osnovne elemente teorije matrične faktorizacije za sisteme preporuke.
Prvo objašnjavamo osnovu metodu, a potom i njena uopštenja kroz kernelizovanu metodu matrične faktorizacije
i ograničene kernelizovane metode matrične faktorizacije. \
Sistemi preporuka su jedna od najuspješnijih primjena mašinskog učenja u praksi. 
Korišćenjem sistema muzičkih preporuka, muzičke platforme mogu da predvide, a zatim ponude
odgovarajuće izvođače i pjesme svojim korisnicima na osnovu svojstava muzike koju su prethodno slušali sami korisnici.
Upotrebom naprednijih modela koji su implementirani u projektu moguće je koristiti i informacije o društvenoj mreži korisnika 
na muzičkoj platformi u cilju pronalaženja sličnosti u muzičkom ukusu korisnika i njihovih prijatelja. 
Mogućnost efikasnog pružanja personalizovanih muzičkih preporuka je od suštinskog značaja za konkurentnost bilo koje usluge slušanja muzike. 

#### 2. Skup podataka i vizuelizacija
Predstavljen je i analiziran skup podataka *hetrec2011-lastfm-2k* koji je korišćen u okviru projekta.
U svrhu smanjenja memorijske i vremenske složenosti, izdvojen je podskup od hiljadu najslušanijih izvođača na muzičkoj platformi.
Prikazane su sumarne statisitke, kao i histogrami rejtinga svih izvođača i korisnika. 
Primijećeno je da su podaci neizbalansirani zbog čega je bilo neophodno implementirati kompleksnije metode od osnovne 
probabilističke matrične faktorizacije koja nije efikasna u slučaju takvih podataka. 

#### 3. Trening modela
Istrenirana su dva različita modela ...
#### 4. Analiza rezultata i upoređivanje implementiranih metoda
