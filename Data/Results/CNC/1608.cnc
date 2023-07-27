
%
P1 (LANTEK)
N10 (Nukon Fiber Laser PA8000)
N11 (SHEET SIZE:[mm]: 246.178 x 102.063)
N12 (MATERIAL: Al99)
N13 (THICKNESS:[mm]: 0.8)
N14 (STRATEGY: Air/150)
N15 (DATE: 22/06/2022 12:36:57)
N16 (RUNTIME:[hour min sec]: 00:00:03)
N17 (SHEET WEIGHT:[kg]: 0.0542713)
N18 (FOCAL LENGTH: 0.00)
N19 (NOZZLE DIAMETER: 0.00)
*N20 P188=102 / (SHEET SIZE Y)
*N21 P187=246 / (SHEET SIZE X)
*N22 P193=1/(TOTAL NR OF PARTS IN JOB)
N23 G71 M271
*N24 P160=0
*N25 P166=0
*N26 P167=0
*N27 P171=0
*N28 P173=10/ (JOB BOX X MIN)
*N29 P174=10/ (JOB BOX Y MIN)
*N30 P175=92.067/ (JOB BOX X MAX)
*N31 P176=236.178/ (JOB BOX Y MAX)
*N32 P177=82.067/ (JOB WIDTH)
*N33 P183=226.178/ (JOB HEIGHT)
*N34 P150=185.467 /(LENGHT OF FIRST RAPID MOVEMENT)
*N35 P151=1 /(NUMBER OF CONTOURS)
*N36 P152=185.467 /(TOTAL RAPID MOVEMENT LENGHT)
*N37 P153=613.632 /(CUT1 LENGHT)
*N38 P154=0 /(CUT2 LENGHT)
*N39 P155=0 /(CUT3 LENGHT)
*N40 P156=0 /(FILM BURNING LENGHT)
*N41 P157=0 /(HOLE LENGHT)
*N42 P158=0 /(MARKING LENGHT)
*N43 P159=0
N44T01800800
*N45 IF P197=0 GO 47
N46 Q899990
N47 Q899991
*N48 P197=1, P180=0, P98=0
*N49 IF P198>0 GO 52
*N50 IF P199>0 GO 52
*N51 GO 55
*N52 P98 = P198*100000, P99 =P199*10000000000
*N53 P98 = P98+P99+1,P198=P198-1
*N54 GO P98
N55 G10(** start **)

N10000100001 (PART NAME:DXFPART7)
*N10000100002 P199=1
*N10000100003 P198=1
*N10000100004 P178=10, P179=10, P194=1
*N10000100005 P189=0, P190=0, P191=82.063, P192=226.178
N10000100006 Q899993
*N10000100007 P189=0, P190=0, P191=82.063, P192=226.178
N10000100008 M201 (CUT1)
*N10000100009 P198 = 1
N10000100010 G0 X88.922 Y162.76
N10000100011 Q899996
N10000100012 G41 D100
N10000100013 G1 X85.969 Y162.231 F=P101
*N10000100014 IF P108 = 0 GO 10000100016
N10000100015 G4 F=P108
N10000100016 G3 X91.886 Y130.784 I2.825 J-15.749 F=P102 U20 O=P215
N10000100017 G2 X74.396 Y60.911 I-121.823 J-6.639
N10000100018 G3 X65.287 Y47.874 I-5.261 J-6.023
N10000100019 G2 X13.131 Y10 I-95.224 J76.271
N10000100020 G1 X12.372 Y20.845
N10000100021 G1 X16.362 Y21.124
N10000100022 G1 X14.827 Y43.07
N10000100023 G1 X10.837 Y42.791
N10000100024 G1 X10 Y54.762
N10000100025 G1 X29.278 Y97.331
N10000100026 G3 X32.029 Y143.77 I-59.215 J26.814
N10000100027 G1 X12.651 Y160.204
N10000100028 G1 X11.917 Y170.708
N10000100029 G1 X23.785 Y173.012
N10000100030 G1 X23.087 Y182.988
N10000100031 G1 X11.014 Y183.617
N10000100032 G1 X10.229 Y194.849
N10000100033 G1 X15.666 Y195.51
N10000100034 G3 X20.474 Y211.238 I-1.096 J8.936
N10000100035 G1 X11.259 Y219.251
N10000100036 G1 X19.509 Y219.828
N10000100037 G1 X18.365 Y236.178
N10000100038 G2 X85.965 Y162.232 I-48.302 J-112.033 M380
*N10000100039 P123=85.965, P124=162.232
*N10000100040 P125=0.000, P126=0.000
*N10000100041 P150=0
N10000100042 G40
N10000100043 Q899997
N10000100044 (END OF PARTS)
*N10000100045 P193=0,P194=0,P197=0,P198=0,P199=0
N10000100046 Q899994
N10000100047 G53 (MACHINE COORDINATE SYSTEM ON)
N10000100048 M30
%
