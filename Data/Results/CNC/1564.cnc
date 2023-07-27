
%
P1 (LANTEK)
N10 (Nukon Fiber Laser PA8000)
N11 (SHEET SIZE:[mm]: 229.633 x 162.619)
N12 (MATERIAL: Al99)
N13 (THICKNESS:[mm]: 0.8)
N14 (STRATEGY: Air/150)
N15 (DATE: 22/06/2022 12:36:17)
N16 (RUNTIME:[hour min sec]: 00:00:12)
N17 (SHEET WEIGHT:[kg]: 0.0806603)
N18 (FOCAL LENGTH: 0.00)
N19 (NOZZLE DIAMETER: 0.00)
*N20 P187=230 / (SHEET SIZE X)
*N21 P188=163 / (SHEET SIZE Y)
*N22 P193=1/(TOTAL NR OF PARTS IN JOB)
N23 G71 M271
*N24 P160=0
*N25 P166=0
*N26 P167=0
*N27 P171=0
*N28 P173=10/ (JOB BOX X MIN)
*N29 P174=10/ (JOB BOX Y MIN)
*N30 P175=219.634/ (JOB BOX X MAX)
*N31 P176=152.619/ (JOB BOX Y MAX)
*N32 P177=209.634/ (JOB WIDTH)
*N33 P183=142.619/ (JOB HEIGHT)
*N34 P150=178.582 /(LENGHT OF FIRST RAPID MOVEMENT)
*N35 P151=9 /(NUMBER OF CONTOURS)
*N36 P152=683.564 /(TOTAL RAPID MOVEMENT LENGHT)
*N37 P153=960.585 /(CUT1 LENGHT)
*N38 P154=0 /(CUT2 LENGHT)
*N39 P155=0 /(CUT3 LENGHT)
*N40 P156=0 /(FILM BURNING LENGHT)
*N41 P157=187.646 /(HOLE LENGHT)
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

N10000100001 (PART NAME:DXFPAR0105)
*N10000100002 P199=1
*N10000100003 P198=1
*N10000100004 P178=10, P179=10, P194=9
*N10000100005 P189=0, P190=0, P191=209.633, P192=142.619
N10000100006 Q899993
*N10000100007 P189=0, P190=0, P191=9, P192=9
N10000100008 M205 (SMALL HOLE)
*N10000100009 P198 = 1
N10000100010 G0 X143.974 Y105.656
N10000100011 Q899996
N10000100012 G41 D100
N10000100013 G1 X141.853 Y107.777 F=P101
*N10000100014 IF P108 = 0 GO 10000100016
N10000100015 G4 F=P108
*N10000100016 P123=141.853, P124=107.777
*N10000100017 P125=155.446, P126=89.273
*N10000100018 P150=22.96
N10000100019 G40
N10000100020 Q899997
*N10000200001 P189=0, P190=0, P191=9, P192=9
N10000200002 M205 (SMALL HOLE)
*N10000200003 P198=2
*N10000200004 P125=155.446, P126=89.273
N10000200005 Q899996
N10000200006 G41 D100
N10000200007 G1 X153.324 Y91.394 F=P102 U20 O=P215
*N10000200008 IF P108 = 0 GO 10000200010
N10000200009 G4 F=P108
*N10000200010 P123=153.324, P124=91.394
*N10000200011 P125=172.648, P126=101.318
*N10000200012 P150=21.723
N10000200013 G40
N10000200014 Q899997
*N10000300001 P189=0, P190=0, P191=9, P192=9
N10000300002 M205 (SMALL HOLE)
*N10000300003 P198=3
*N10000300004 P125=172.648, P126=101.318
N10000300005 Q899996
N10000300006 G41 D100
N10000300007 G1 X170.526 Y103.439 F=P101
*N10000300008 IF P108 = 0 GO 10000300010
N10000300009 G4 F=P108
*N10000300010 P123=170.526, P124=103.439
*N10000300011 P125=161.176, P126=117.701
*N10000300012 P150=17.054
N10000300013 G40
N10000300014 Q899997
*N10000400001 P189=0, P190=0, P191=9, P192=9
N10000400002 M205 (SMALL HOLE)
*N10000400003 P198=4
*N10000400004 P125=161.176, P126=117.701
N10000400005 Q899996
N10000400006 G41 D100
N10000400007 G1 X159.055 Y119.822 F=P102 U20 O=P215
*N10000400008 IF P108 = 0 GO 10000400010
N10000400009 G4 F=P108
*N10000400010 P123=159.055, P124=119.822
*N10000400011 P125=190.256, P126=138.063
*N10000400012 P150=36.142
N10000400013 G40
N10000400014 Q899997
*N10000500001 P189=0, P190=0, P191=9, P192=9
N10000500002 M205 (SMALL HOLE)
*N10000500003 P198=5
*N10000500004 P125=190.256, P126=138.063
N10000500005 Q899996
N10000500006 G41 D100
N10000500007 G1 X188.135 Y140.184 F=P101
*N10000500008 IF P108 = 0 GO 10000500010
N10000500009 G4 F=P108
*N10000500010 P123=188.135, P124=140.184
*N10000500011 P125=201.728, P126=121.68
*N10000500012 P150=22.96
N10000500013 G40
N10000500014 Q899997
*N10000600001 P189=0, P190=0, P191=9, P192=9
N10000600002 M205 (SMALL HOLE)
*N10000600003 P198=6
*N10000600004 P125=201.728, P126=121.68
N10000600005 Q899996
N10000600006 G41 D100
N10000600007 G1 X199.606 Y123.801 F=P102 U20 O=P215
*N10000600008 IF P108 = 0 GO 10000600010
N10000600009 G4 F=P108
*N10000600010 P123=199.606, P124=123.801
*N10000600011 P125=130.086, P126=84.972
*N10000600012 P150=79.629
N10000600013 G40
N10000600014 Q899997
*N10000700001 P189=0, P190=0, P191=31.537, P192=67.425
N10000700002 M201 (CUT1)
*N10000700003 P198=7
*N10000700004 P125=130.086, P126=84.972
N10000700005 Q899996
N10000700006 G41 D100
N10000700007 G1 X128.366 Y87.43 F=P101
*N10000700008 IF P108 = 0 GO 10000700010
N10000700009 G4 F=P108
N10000700010 G3 X127.015 Y79.77 I3.154 J-4.506 F=P102 U20 O=P215
N10000700011 G2 X146.578 Y26.02 I-90.516 J-63.38
N10000700012 G3 X157.536 Y26.979 I5.479 J0.48
N10000700013 G3 X136.026 Y86.079 I-121.037 J-10.589
N10000700014 G3 X128.366 Y87.43 I-4.506 J-3.155 M380
*N10000700015 P123=128.366, P124=87.43
*N10000700016 P125=23.898, P126=131.73
*N10000700017 P150=113.473
N10000700018 G40
N10000700019 Q899997
*N10000800001 P189=0, P190=0, P191=70.134, P192=21.868
N10000800002 M201 (CUT1)
*N10000800003 P198=8
*N10000800004 P125=23.898, P126=131.73
N10000800005 Q899996
N10000800006 G41 D100
N10000800007 G1 X20.91 Y131.469 F=P101
*N10000800008 IF P108 = 0 GO 10000800010
N10000800009 G4 F=P108
N10000800010 G3 X26.868 Y126.469 I5.479 J0.479 F=P102 U20 O=P215
N10000800011 G2 X83.198 Y116.537 I9.631 J-110.079
N10000800012 G3 X87.847 Y126.506 I2.324 J4.984
N10000800013 G3 X25.909 Y137.427 I-51.348 J-110.116
N10000800014 G3 X20.91 Y131.469 I0.48 J-5.479 M380
*N10000800015 P123=20.91, P124=131.469
*N10000800016 P125=211.797, P126=139.135
*N10000800017 P150=191.041
N10000800018 G40
N10000800019 Q899997
*N10000900001 P189=0, P190=0, P191=209.633, P192=142.619
N10000900002 M201 (CUT1)
*N10000900003 P198=9
*N10000900004 P125=211.797, P126=139.135
N10000900005 Q899996
N10000900006 G41 D100
N10000900007 G1 X209.34 Y137.414 F=P101
*N10000900008 IF P108 = 0 GO 10000900010
N10000900009 G4 F=P108
N10000900010 G1 X219.091 Y123.489 F=P102 U20 O=P215
N10000900011 G2 X218.354 Y119.31 I-2.458 J-1.721
N10000900012 G1 X171.486 Y86.493
N10000900013 G3 X164.181 Y63.223 I11.471 J-16.383
N10000900014 G2 X172.498 Y15.97 I-127.682 J-46.833
N10000900015 G2 X169.76 Y12.991 I-3 J0.01
N10000900016 G1 X135.702 Y10.011
N10000900017 G2 X132.442 Y13.103 I-0.261 J2.989
N10000900018 G3 X16.601 Y110.305 I-95.943 J3.287
N10000900019 G2 X12.991 Y112.978 I-0.621 J2.935
N10000900020 G1 X10.011 Y147.036
N10000900021 G2 X12.47 Y150.25 I2.989 J0.261
N10000900022 G2 X124.177 Y120.353 I24.029 J-133.86
N10000900023 G3 X148.543 Y119.259 I12.894 J15.289
N10000900024 G1 X195.411 Y152.076
N10000900025 G2 X199.589 Y151.34 I1.721 J-2.457
N10000900026 G1 X209.34 Y137.414 M380
*N10000900027 P123=209.34, P124=137.414
*N10000900028 P125=0.000, P126=0.000
*N10000900029 P150=0
N10000900030 G40
N10000900031 Q899997
N10000900032 (END OF PARTS)
*N10000900033 P193=0,P194=0,P197=0,P198=0,P199=0
N10000900034 Q899994
N10000900035 G53 (MACHINE COORDINATE SYSTEM ON)
N10000900036 M30
%
