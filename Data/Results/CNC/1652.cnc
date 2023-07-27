
%
P1 (LANTEK)
N10 (Nukon Fiber Laser PA8000)
N11 (SHEET SIZE:[mm]: 200 x 135)
N12 (MATERIAL: Al99)
N13 (THICKNESS:[mm]: 0.8)
N14 (STRATEGY: Air/150)
N15 (DATE: 22/06/2022 12:37:37)
N16 (RUNTIME:[hour min sec]: 00:00:10)
N17 (SHEET WEIGHT:[kg]: 0.05832)
N18 (FOCAL LENGTH: 0.00)
N19 (NOZZLE DIAMETER: 0.00)
*N20 P188=135 / (SHEET SIZE Y)
*N21 P187=200 / (SHEET SIZE X)
*N22 P193=1/(TOTAL NR OF PARTS IN JOB)
N23 G71 M271
*N24 P160=0
*N25 P166=0
*N26 P167=0
*N27 P171=0
*N28 P173=10/ (JOB BOX X MIN)
*N29 P174=10/ (JOB BOX Y MIN)
*N30 P175=125/ (JOB BOX X MAX)
*N31 P176=190/ (JOB BOX Y MAX)
*N32 P177=115/ (JOB WIDTH)
*N33 P183=180/ (JOB HEIGHT)
*N34 P150=99.687 /(LENGHT OF FIRST RAPID MOVEMENT)
*N35 P151=6 /(NUMBER OF CONTOURS)
*N36 P152=548.061 /(TOTAL RAPID MOVEMENT LENGHT)
*N37 P153=965.982 /(CUT1 LENGHT)
*N38 P154=0 /(CUT2 LENGHT)
*N39 P155=0 /(CUT3 LENGHT)
*N40 P156=0 /(FILM BURNING LENGHT)
*N41 P157=263.323 /(HOLE LENGHT)
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

N10000100001 (PART NAME:Pieza_prueba_Maxwell_2)
*N10000100002 P199=1
*N10000100003 P198=1
*N10000100004 P178=10, P179=10, P194=6
*N10000100005 P189=0, P190=0, P191=115, P192=180
N10000100006 Q899993
*N10000100007 P189=0, P190=0, P191=20, P192=20
N10000100008 M205 (SMALL HOLE)
*N10000100009 P198 = 1
N10000100010 G0 X95.05 Y30.05
N10000100011 Q899996
N10000100012 G41 D100
N10000100013 G1 X92.929 Y27.929 F=P101
*N10000100014 IF P108 = 0 GO 10000100016
N10000100015 G4 F=P108
*N10000100016 P123=92.929, P124=27.929
*N10000100017 P125=30.05, P126=30.05
*N10000100018 P150=62.915
N10000100019 G40
N10000100020 Q899997
*N10000200001 P189=0, P190=0, P191=20, P192=20
N10000200002 M205 (SMALL HOLE)
*N10000200003 P198=2
*N10000200004 P125=30.05, P126=30.05
N10000200005 Q899996
N10000200006 G41 D100
N10000200007 G1 X27.929 Y27.929 F=P102 U20 O=P215
*N10000200008 IF P108 = 0 GO 10000200010
N10000200009 G4 F=P108
*N10000200010 P123=27.929, P124=27.929
*N10000200011 P125=30.05, P126=160.05
*N10000200012 P150=132.138
N10000200013 G40
N10000200014 Q899997
*N10000300001 P189=0, P190=0, P191=20, P192=20
N10000300002 M205 (SMALL HOLE)
*N10000300003 P198=3
*N10000300004 P125=30.05, P126=160.05
N10000300005 Q899996
N10000300006 G41 D100
N10000300007 G1 X27.929 Y157.929 F=P101
*N10000300008 IF P108 = 0 GO 10000300010
N10000300009 G4 F=P108
*N10000300010 P123=27.929, P124=157.929
*N10000300011 P125=95.05, P126=160.05
*N10000300012 P150=67.155
N10000300013 G40
N10000300014 Q899997
*N10000400001 P189=0, P190=0, P191=20, P192=20
N10000400002 M205 (SMALL HOLE)
*N10000400003 P198=4
*N10000400004 P125=95.05, P126=160.05
N10000400005 Q899996
N10000400006 G41 D100
N10000400007 G1 X92.929 Y157.929 F=P102 U20 O=P215
*N10000400008 IF P108 = 0 GO 10000400010
N10000400009 G4 F=P108
*N10000400010 P123=92.929, P124=157.929
*N10000400011 P125=29.831, P126=93.182
*N10000400012 P150=90.408
N10000400013 G40
N10000400014 Q899997
*N10000500001 P189=0, P190=0, P191=92.284, P192=150
N10000500002 M201 (CUT1)
*N10000500003 P198=5
*N10000500004 P125=29.831, P126=93.182
N10000500005 Q899996
N10000500006 G41 D100
N10000500007 G1 X30.631 Y90.291 F=P101
*N10000500008 IF P108 = 0 GO 10000500010
N10000500009 G4 F=P108
N10000500010 G3 X56.773 Y64.148 I36.142 J10 F=P102 U20 O=P215
N10000500011 G1 X56.773 Y35.291
N10000500012 G3 X76.773 Y35.291 I10 J0
N10000500013 G1 X76.773 Y64.148
N10000500014 G3 X102.915 Y90.291 I-10 J36.143
N10000500015 G1 X112.915 Y90.291
N10000500016 G1 X112.915 Y110.291
N10000500017 G1 X102.915 Y110.291
N10000500018 G3 X76.773 Y136.433 I-36.142 J-10
N10000500019 G1 X76.773 Y165.291
N10000500020 G3 X56.773 Y165.291 I-10 J0
N10000500021 G1 X56.773 Y136.433
N10000500022 G3 X30.631 Y110.291 I10 J-36.142
N10000500023 G1 X20.631 Y110.291
N10000500024 G1 X20.631 Y90.291
N10000500025 G1 X30.631 Y90.291 M380
*N10000500026 P123=30.631, P124=90.291
*N10000500027 P125=15.201, P126=184.799
*N10000500028 P150=95.759
N10000500029 G40
N10000500030 Q899997
*N10000600001 P189=0, P190=0, P191=115, P192=180
N10000600002 M201 (CUT1)
*N10000600003 P198=6
*N10000600004 P125=15.201, P126=184.799
N10000600005 Q899996
N10000600006 G41 D100
N10000600007 G1 X17.322 Y182.678 F=P101
*N10000600008 IF P108 = 0 GO 10000600010
N10000600009 G4 F=P108
N10000600010 G2 X35 Y190 I17.678 J-17.678 F=P102 U20 O=P215
N10000600011 G1 X100 Y190
N10000600012 G2 X125 Y165 I0 J-25
N10000600013 G1 X125 Y35
N10000600014 G2 X100 Y10 I-25 J0
N10000600015 G1 X35 Y10
N10000600016 G2 X10 Y35 I0 J25
N10000600017 G1 X10 Y165
N10000600018 G2 X17.322 Y182.678 I25 J0 M380
*N10000600019 P123=17.322, P124=182.678
*N10000600020 P125=0.000, P126=0.000
*N10000600021 P150=0
N10000600022 G40
N10000600023 Q899997
N10000600024 (END OF PARTS)
*N10000600025 P193=0,P194=0,P197=0,P198=0,P199=0
N10000600026 Q899994
N10000600027 G53 (MACHINE COORDINATE SYSTEM ON)
N10000600028 M30
%
