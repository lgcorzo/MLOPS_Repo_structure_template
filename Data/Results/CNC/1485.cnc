
%
P1 (LANTEK)
N10 (Nukon Fiber Laser PA8000)
N11 (SHEET SIZE:[mm]: 120 x 120)
N12 (MATERIAL: Al99)
N13 (THICKNESS:[mm]: 0.8)
N14 (STRATEGY: Air/150)
N15 (DATE: 22/06/2022 12:35:11)
N16 (RUNTIME:[hour min sec]: 00:00:04)
N17 (SHEET WEIGHT:[kg]: 0.031104)
N18 (FOCAL LENGTH: 0.00)
N19 (NOZZLE DIAMETER: 0.00)
*N20 P188=120 / (SHEET SIZE Y)
*N21 P187=120 / (SHEET SIZE X)
*N22 P193=1/(TOTAL NR OF PARTS IN JOB)
N23 G71 M271
*N24 P160=0
*N25 P166=0
*N26 P167=0
*N27 P171=0
*N28 P173=10/ (JOB BOX X MIN)
*N29 P174=10/ (JOB BOX Y MIN)
*N30 P175=110/ (JOB BOX X MAX)
*N31 P176=110/ (JOB BOX Y MAX)
*N32 P177=100/ (JOB WIDTH)
*N33 P183=100/ (JOB HEIGHT)
*N34 P150=62.853 /(LENGHT OF FIRST RAPID MOVEMENT)
*N35 P151=2 /(NUMBER OF CONTOURS)
*N36 P152=162.049 /(TOTAL RAPID MOVEMENT LENGHT)
*N37 P153=600.209 /(CUT1 LENGHT)
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

N10000100001 (PART NAME:Pieza_prueba_Maxwell)
*N10000100002 P199=1
*N10000100003 P198=1
*N10000100004 P178=10, P179=10, P194=3
*N10000100005 P189=0, P190=0, P191=100, P192=100
N10000100006 Q899993
*N10000100007 P189=0, P190=0, P191=50, P192=50
N10000100008 M201 (CUT1)
*N10000100009 P198 = 1
N10000100010 G0 X44.444 Y44.444
N10000100011 Q899996
N10000100012 G41 D100
N10000100013 G1 X42.322 Y42.322 F=P101
*N10000100014 IF P108 = 0 GO 10000100016
N10000100015 G4 F=P108
N10000100016 G1 X10 Y10 F=P102 U20 O=P215 M380
*N10000100017 P123=10, P124=10
*N10000100018 P125=10.808, P126=109.192
*N10000100019 P150=99.195
N10000100020 G40
N10000100021 Q899997
*N10000200001 P189=0, P190=0, P191=100, P192=100
N10000200002 M201 (CUT1)
*N10000200003 P198=2
*N10000200004 P125=10.808, P126=109.192
N10000200005 Q899996
N10000200006 G41 D100
N10000200007 G1 X12.929 Y107.071 F=P101
*N10000200008 IF P108 = 0 GO 10000200010
N10000200009 G4 F=P108
N10000200010 G2 X20 Y110 I7.071 J-7.071 F=P102 U20 O=P215
N10000200011 G1 X110 Y110
N10000200012 G1 X110 Y20
N10000200013 G2 X100 Y10 I-10 J0
N10000200014 G1 X10 Y10
N10000200015 G1 X10 Y100
N10000200016 G2 X12.929 Y107.071 I10 J0 M380
*N10000200017 P123=12.929, P124=107.071
*N10000200018 P125=0.000, P126=0.000
*N10000200019 P150=0
N10000200020 G40
N10000200021 Q899997
N10000200022 (END OF PARTS)
*N10000200023 P193=0,P194=0,P197=0,P198=0,P199=0
N10000200024 Q899994
N10000200025 G53 (MACHINE COORDINATE SYSTEM ON)
N10000200026 M30
%
