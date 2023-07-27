%_N_114000_MPF
;$PATH=/_N_WKS_DIR/_N_114000_WPD
;PP : CC BLOCKD56 22.06.2022
DEF REAL VEND
DEF REAL VAKT
DEF INT  KOPF_ANZAHL
DEF INT  KOPF_ABSTAND
MSG("PLANNAME           :114000")
;MATERIAL           :Al99
;TAFELNAME          :114000
;LAENGE          MM :120.00
;BREITE          MM :120.00
;DICKE           MM :0.80
;PLATTENFLAECHE  QM :0.01

R500=120.00
R501=120.00
R611=120.00
R612=120.00
R613=0.80
N5 $P_IFRAME=$P_UIFR[R910]
N10 R200=0
N15 G90 G00
N20 L210
N25 X0 Y0
N30 $P_PFRAME=CSCALE(X,R911,Y,R911)
N35 R915=4
N40 PARAMETER
R696=0
IF R698>1 GOTOF "TEIL0"<<R698
IF R699>1 GOTOF HPSTART
N45 G90 G00 X0 Y0
N50 HPSTART:
N55 M20
N60 M31
N65 M20
N70 M31
TEIL01:
STOPRE
R696=R696+R698
R698=1
N75 R401=10
N80 R402=10
N85 R403=0.00
N90 R404=1
N95 PART1
N100 L206
N105 M20 
N110 M30

%_N_PARAMETER_SPF
;$PATH=/_N_WKS_DIR/_N_114000_WPD
IF (R912==4) OR (R912==6)
GOTOF ENDE
ENDIF
R911=1.0                                ;Massstabsfaktor
R923=7                                  ;Anschussart 7=PUNKT, 8=BOHREN, 9 =Rotation
R928=1.00                               ;Rotationsradius 
R917=250.00                             ;Rotationsanschuss Vorschub mm/min
R929=1.00                               ;Rotationsanschusszeit
R914=1.00                               ;Punktanschusszeit
R913=0.00                               ;Druckanstiegszeit
R959=270.00                             ;Abrasiv beim Anschuss 
R935=15                                  ;Silovorwahl
R920=20                              ;Abheben Z am Schnittende
;
R946=1700                               ;Freifahren X am Programmende / Messen
;
R899=330                                ;Vorschub Reihe 1 sehr gut
R938=800                                ;Vorschub Reihe 2 gut
R939=1200                                ;Vorschub Reihe 3 mittelfein
R942=1600                                ;Vorschub Reihe 4 mittel
R943=3000                               ;Vorschub Reihe 5 grob
;
R931=200                                ;Vorschub in Ecken Reihe 1 sehr gut 
R940=520                                ;Vorschub in Ecken Reihe 2 gut
R941=780                                ;Vorschub in Ecken Reihe 3 mittelfein
R944=1040                                ;Vorschub in Ecken Reihe 4 mittel
R945=1950                                ;Vorschub in Ecken Reihe 5 grob
;
R932=1.00                               ;Schnittspalt Reihe 1 sehr gut
R933=1.00                               ;Schnittspalt Reihe 2 gut
R934=1.00                               ;Schnittspalt Reihe 3 mittelfein
R958=1.00                               ;Schnittspalt Reihe 4 mittel
R927=1.00                               ;Schnittspalt Reihe 5 grob
;
R947=3500.00                            ;Schneiddruck Reihe 1 sehr gut 
R949=3500.00                            ;Schneiddruck Reihe 2 gut
R951=3500.00                            ;Schneiddruck Reihe 3 mittelfein
R953=3500.00                            ;Schneiddruck Reihe 4 mittel
R955=3500.00                            ;Schneiddruck Reihe 5 grob
;
R948=340.00                             ;Abrasiv Reihe 1 sehr gut
R950=340.00                             ;Abrasiv Reihe 2 gut
R952=340.00                             ;Abrasiv Reihe 3 mittelfein
R954=340.00                             ;Abrasiv Reihe 4 mittel
R956=340.00                             ;Abrasiv Reihe 5 grob
;
R967=340.00                             ;Abrasiv in Ecken Reihe 1 sehr gut
R968=340.00                             ;Abrasiv in Ecken Reihe 2 gut
R969=340.00                             ;Abrasiv in Ecken Reihe 3 mittelfein
R970=340.00                             ;Abrasiv in Ecken Reihe 4 mittel
R971=340.00                             ;Abrasiv in Ecken Reihe 5 grob
;
R916=3500.00                            ;Anschussdruck 
;
R937=500                                ;Min. Druck Abrasiv ein 
;
;GRAVIER-PARAMETER Reihe 6  
R800=1000                            ;Vorschub 
R803=1000.00                            ;Schneiddruck 
R804=250.00                            ;Abrasiv 
;
;KOERN-PARAMETER Reihe 7 
R909=0.00                               ;Anschusszeit
R811=600.00                             ;Schneiddruck
R812=50.00                              ;Abrasiv 
;
;BOHRPARAMETER    
R985=0.00                               ;Vorschub   
R986=0.00                               ;Tiefe gesamt
R987=0.00                               ;Drehzahl  
R988=0.00                               ;Bohrtiefe 1
R989=0.00                               ;Bohrtiefe 2
;
;BOHREN KLEINER LOECHER
R571=1.00                               ;Rotationsvorschub
R570=1.00                               ;Rotationsradius
R572=0.00                               ;Rotationsanschusszeit
;
;
R996=R916                               ;Sichern des CAD/CAM Anschussdruckes 
R997=R914                               ;Sichern der CAD/CAM Punktanschusszeit
R998=R920                               ;Sichern der CAD/CAM Abhebehoehe Z
ENDE:
M17

%_N_PART1_SPF
;$PATH=/_N_WKS_DIR/_N_114000_WPD
;NEUES TEIL NR 1
;KOMMISSION :
;KEINE AUFTRAGSDATEN
N5 TRANS X=R401 Y=R402
N10 AROT RPL=R403
N15 IF R404==1 GOTOF LABEL1
N20 AMIRROR X0
N25 LABEL1:
R697=0
IF R699>1 GOTOF "CONTOUR"<<R699
CONTOUR1:
STOPRE
R697=R697+R699
R699=1
N30 G00 X72.000 Y50.600
N35 R957=3
R923=7
IF (R912<>4) AND (R912<>6)
R914=R997  ;
ENDIF
N40 L201
N45 VAKT=R939*0.6
N50 G41 G01 X74.393 Y50.600 F=VAKT FLIN H1
N55 G03 X73.760 Y55.551 I-24.393 J-0.600
N60 VAKT=R939*0.61
N65 G03 X25.600 Y50.000 I-23.760 J-5.551 F=VAKT FLIN
N70 G03 X25.657 Y48.335 I24.400 J0.000
N75 VAKT=R939*0.73
N80 G03 X25.827 Y46.677 I24.343 J1.665 F=VAKT FLIN
N85 VAKT=R939*0.88
N90 G03 X26.110 Y45.035 I24.173 J3.323 F=VAKT FLIN
N95 G03 X72.684 Y41.010 I23.890 J4.965 F=R939 FLIN H0
N100 VAKT=R939*0.88
N105 G03 X73.244 Y42.579 I-22.684 J8.990 F=VAKT FLIN H1
N110 VAKT=R939*0.73
N115 G03 X73.697 Y44.182 I-23.244 J7.421 F=VAKT FLIN
N120 VAKT=R939*0.61
N125 G03 X74.268 Y47.464 I-23.697 J5.818 F=VAKT FLIN
N130 VAKT=R939*0.72
N135 G03 X74.385 Y49.126 I-24.268 J2.536 F=VAKT FLIN
N140 VAKT=R939*0.6
N145 G03 X74.393 Y49.400 I-24.385 J0.874 F=VAKT FLIN
N150 G40 G01 X73.000 Y49.400
N155 L205
CONTOUR2:
STOPRE
R697=R697+R699
R699=1
N160 G00 X32.322 Y67.678
N165 R957=3
R923=7
IF (R912<>4) AND (R912<>6)
R914=R997  ;
ENDIF
N170 L201
N175 VAKT=R939*0.6
N180 G01 X31.143 Y68.857 F=VAKT FLIN H1
N185 VAKT=R939*0.72
N190 G01 X29.965 Y70.035 F=VAKT FLIN
N195 VAKT=R939*0.86
N200 G01 X28.786 Y71.214 F=VAKT FLIN
N205 G01 X0.000 Y100.000 F=R939 FLIN H0
N210 L205
CONTOUR3:
STOPRE
R697=R697+R699
R699=1
N215 G00 X103.000 Y44.400
N220 R957=3
R923=7
IF (R912<>4) AND (R912<>6)
R914=R997  ;
ENDIF
N225 L201
N230 VAKT=R939*0.6
N235 G41 G01 X100.600 Y44.400 F=VAKT FLIN H1
N240 G01 X100.600 Y42.733
N245 VAKT=R939*0.72
N250 G01 X100.600 Y41.067 F=VAKT FLIN
N255 VAKT=R939*0.86
N260 G01 X100.600 Y39.400 F=VAKT FLIN
N265 G01 X100.600 Y5.000 F=R939 FLIN H0
N270 VAKT=R939*0.86
N275 G01 X100.600 Y3.333 F=VAKT FLIN H1
N280 VAKT=R939*0.72
N285 G01 X100.600 Y1.667 F=VAKT FLIN
N290 VAKT=R939*0.6
N295 G01 X100.600 Y0.000 F=VAKT FLIN
N300 G01 X100.000 Y-0.600
N305 G01 X98.333 Y-0.600
N310 VAKT=R939*0.72
N315 G01 X96.667 Y-0.600 F=VAKT FLIN
N320 VAKT=R939*0.86
N325 G01 X95.000 Y-0.600 F=VAKT FLIN
N330 G01 X15.000 Y-0.600 F=R939 FLIN H0
N335 VAKT=R939*0.87
N340 G01 X13.333 Y-0.600 F=VAKT FLIN H1
N345 VAKT=R939*0.72
N350 G01 X11.667 Y-0.600 F=VAKT FLIN
N355 VAKT=R939*0.6
N360 G01 X10.000 Y-0.600 F=VAKT FLIN
N365 G02 X-0.600 Y10.000 I0.000 J10.600
N370 G01 X-0.600 Y11.667
N375 VAKT=R939*0.72
N380 G01 X-0.600 Y13.333 F=VAKT FLIN
N385 VAKT=R939*0.87
N390 G01 X-0.600 Y15.000 F=VAKT FLIN
N395 G01 X-0.600 Y95.000 F=R939 FLIN H0
N400 VAKT=R939*0.86
N405 G01 X-0.600 Y96.667 F=VAKT FLIN H1
N410 VAKT=R939*0.72
N415 G01 X-0.600 Y98.333 F=VAKT FLIN
N420 VAKT=R939*0.6
N425 G01 X-0.600 Y100.000 F=VAKT FLIN
N430 G01 X0.000 Y100.600
N435 G01 X1.667 Y100.600
N440 VAKT=R939*0.72
N445 G01 X3.333 Y100.600 F=VAKT FLIN
N450 VAKT=R939*0.86
N455 G01 X5.000 Y100.600 F=VAKT FLIN
N460 G01 X85.000 Y100.600 F=R939 FLIN H0
N465 VAKT=R939*0.87
N470 G01 X86.667 Y100.600 F=VAKT FLIN H1
N475 VAKT=R939*0.72
N480 G01 X88.333 Y100.600 F=VAKT FLIN
N485 VAKT=R939*0.6
N490 G01 X90.000 Y100.600 F=VAKT FLIN
N495 G02 X100.600 Y90.000 I0.000 J-10.600
N500 G01 X100.600 Y88.333
N505 VAKT=R939*0.72
N510 G01 X100.600 Y86.667 F=VAKT FLIN
N515 VAKT=R939*0.87
N520 G01 X100.600 Y85.000 F=VAKT FLIN
N525 G01 X100.600 Y49.200 F=R939 FLIN H0
N530 VAKT=R939*0.86
N535 G01 X100.600 Y47.533 F=VAKT FLIN H1
N540 VAKT=R939*0.72
N545 G01 X100.600 Y45.867 F=VAKT FLIN
N550 VAKT=R939*0.6
N555 G01 X100.600 Y45.600 F=VAKT FLIN
N560 G40 G01 X102.000 Y45.600
N565 L205
N570 MIRROR
N575 ROT
N580 R403=0
N585 R404=0
N590 G90
STOPRE
R697=1
N595 RET
