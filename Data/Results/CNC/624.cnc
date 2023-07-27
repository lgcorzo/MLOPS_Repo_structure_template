
%
N10 (DATE: 06/22/22)
N11 (MATERIAL: Al99)
N12 (THICKNESS: 0.8)
N13 (SHEET SIZE [mm]:100 x 100)
*N10 P187=100
*N11 P188=100
*N12 P193=0
*N13 P194=0
N14 G71 M271
*N15 P173=-0.13
*N16 P174=-0.13
*N17 P175=100.13
*N18 P176=100.13
*N19 P177=100.26
*N20 P183=100.26
*N21 P150=73.345 /(LENGHT OF FIRST RAPID MOVEMENT)
*N22 P151=4 /(NUMBER OF CONTOURS)
*N23 P152=340.678 /(TOTAL RAPID MOVEMENT LENGHT)
*N24 P153=0 /(CUT1 LENGHT)
*N25 P154=0 /(CUT2 LENGHT)
*N26 P155=239.535 /(CUT3 LENGHT)
*N27 P156=0 /(CUT4 LENGHT)
*N28 P157=0 /(HOLE LENGHT)
*N29 P158=0 /(MARKING LENGHT)
N30 T00000200800
N31 Q899991
*N32 IF P197=0 GO 34
N33 Q899990
*N34 P197=1, P180=0
*N35 IF P198>0 GO 38
*N36 IF P199>0 GO 38
*N37 GO 41
*N38 P98 = P198*10000, P99 =P199*100000000
*N39 P98 = P98+P99+1
*N40 GO P98
N41 G10(** start **)

N100010001 (PART NAME:Pieza_prueba_Maxwell)
*N100010002 P199=1
*N100010003 P178=63.527
*N100010004 P179=36.657
*N100010005 P189=90
*N100010006 P190=90
*N100010007 P191=100
*N100010008 P192=100
*N100010009 P194=3
*N100010010 P198=1
*N100010011 P193=1
N100010012 Q899993
N100010013 G0 X63.527 Y36.657
N100010014 M203 (NORMAL INSIDE CUTTING)
*N100010015 P198 = 1
N100010016 Q899996
N100010017 G41 D100
N100010018 G1 X67.678 Y32.507 F=P101
*N100010019 IF P108 = 0 GO 100010021
N100010020 G4 F=P108
N100010021 G3 X32.414 Y67.585 I-17.677 J17.493 F=P102 U20
N100010022 G3 X67.585 Y32.414 I17.586 J-17.585
*N100010023 P123=67.585, P124=32.414
*N100010024 P125=32.322, P126=67.678
*N100010025 P150=49.87
N100010026 G40
N100010027 Q899997
*N100020001 P125=32.322, P126=67.678
N100020002 M203 (NORMAL INSIDE CUTTING)
*N100020003 P198 = P198+1
N100020004 Q899996
N100020005 G43 D100
N100020006 G1 X0 Y100 U20
*N100020007 P123=0, P124=100
*N100020008 P125=90, P126=100.13
*N100020009 P150=90
N100020010 G40
N100020011 Q899997
*N100030001 P125=90, P126=100.13
N100030002 M203 (NORMAL OUTSIDE CUTTING)
*N100030003 P198 = P198+1
N100030004 Q899996
N100030005 G41 D101
N100030006 G2 X100.13 Y90 I0 J-10.13 U20
*N100030007 P123=100.13, P124=90
*N100030008 P125=10, P126=-0.13
*N100030009 P150=127.463
N100030010 G40
N100030011 Q899997

N200010001 (PART NAME:Pieza_prueba_Maxwell)
*N200010002 P199=2
*N200010003 P178=10
*N200010004 P179=-0.13
*N200010005 P189=0
*N200010006 P190=0
*N200010007 P191=10
*N200010008 P192=10
*N200010009 P194=1
*N200010010 P198=1
*N200010011 P193=1
N200010012 Q899993
*N200010013 P125=10, P126=-0.13
N200010014 M203 (NORMAL OUTSIDE CUTTING)
*N200010015 P198 = 1
N200010016 Q899996
N200010017 G41 D101
N200010018 G2 X-0.13 Y10 I0 J10.131 U20
*N200010019 P150=0
N200010020 G40
N200010021 Q899997
N200010022 (END OF PARTS)
N200010023 G0 X0.000 Y0.000
N200010024 Q899994
*N200010025 P193=0,P194=0,P198=0,P199=0,P160=0
N200010026 G53 (MACHINE COORDINATE SYSTEM ON)
N200010027 M30
