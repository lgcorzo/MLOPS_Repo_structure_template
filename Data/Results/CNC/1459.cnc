
%
P1 (LANTEK)
N10 (Nukon Fiber Laser PA8000)
N11 (SHEET SIZE:[mm]: 229.633 x 162.619)
N12 (MATERIAL: Al99)
N13 (THICKNESS:[mm]: 0.8)
N14 (STRATEGY: Air/150)
N15 (DATE: 22/06/2022 12:34:44)
N16 (RUNTIME:[hour min sec]: 00:00:12)
N17 (SHEET WEIGHT:[kg]: 0.0806603)
N18 (FOCAL LENGTH: 0.00)
N19 (NOZZLE DIAMETER: 0.00)
*N20 P188=163 / (SHEET SIZE Y)
*N21 P187=230 / (SHEET SIZE X)
*N22 P193=1/(TOTAL NR OF PARTS IN JOB)
N23 G71 M271
*N24 P160=0
*N25 P166=0
*N26 P167=0
*N27 P171=0
*N28 P173=10/ (JOB BOX X MIN)
*N29 P174=10/ (JOB BOX Y MIN)
*N30 P175=152.619/ (JOB BOX X MAX)
*N31 P176=219.634/ (JOB BOX Y MAX)
*N32 P177=142.619/ (JOB WIDTH)
*N33 P183=209.634/ (JOB HEIGHT)
*N34 P150=154.833 /(LENGHT OF FIRST RAPID MOVEMENT)
*N35 P151=9 /(NUMBER OF CONTOURS)
*N36 P152=659.814 /(TOTAL RAPID MOVEMENT LENGHT)
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
*N10000100005 P189=0, P190=0, P191=142.619, P192=209.633
N10000100006 Q899993
*N10000100007 P189=0, P190=0, P191=9, P192=9
N10000100008 M205 (SMALL HOLE)
*N10000100009 P198 = 1
N10000100010 G0 X56.963 Y143.974
N10000100011 Q899996
N10000100012 G41 D100
N10000100013 G1 X54.842 Y141.853 F=P101
*N10000100014 IF P108 = 0 GO 10000100016
N10000100015 G4 F=P108
*N10000100016 P123=54.842, P124=141.853
*N10000100017 P125=73.346, P126=155.446
*N10000100018 P150=22.96
N10000100019 G40
N10000100020 Q899997
*N10000200001 P189=0, P190=0, P191=9, P192=9
N10000200002 M205 (SMALL HOLE)
*N10000200003 P198=2
*N10000200004 P125=73.346, P126=155.446
N10000200005 Q899996
N10000200006 G41 D100
N10000200007 G1 X71.225 Y153.324 F=P102 U20 O=P215
*N10000200008 IF P108 = 0 GO 10000200010
N10000200009 G4 F=P108
*N10000200010 P123=71.225, P124=153.324
*N10000200011 P125=61.301, P126=172.648
*N10000200012 P150=21.723
N10000200013 G40
N10000200014 Q899997
*N10000300001 P189=0, P190=0, P191=9, P192=9
N10000300002 M205 (SMALL HOLE)
*N10000300003 P198=3
*N10000300004 P125=61.301, P126=172.648
N10000300005 Q899996
N10000300006 G41 D100
N10000300007 G1 X59.18 Y170.526 F=P101
*N10000300008 IF P108 = 0 GO 10000300010
N10000300009 G4 F=P108
*N10000300010 P123=59.18, P124=170.526
*N10000300011 P125=44.918, P126=161.176
*N10000300012 P150=17.054
N10000300013 G40
N10000300014 Q899997
*N10000400001 P189=0, P190=0, P191=9, P192=9
N10000400002 M205 (SMALL HOLE)
*N10000400003 P198=4
*N10000400004 P125=44.918, P126=161.176
N10000400005 Q899996
N10000400006 G41 D100
N10000400007 G1 X42.797 Y159.055 F=P102 U20 O=P215
*N10000400008 IF P108 = 0 GO 10000400010
N10000400009 G4 F=P108
*N10000400010 P123=42.797, P124=159.055
*N10000400011 P125=24.556, P126=190.256
*N10000400012 P150=36.142
N10000400013 G40
N10000400014 Q899997
*N10000500001 P189=0, P190=0, P191=9, P192=9
N10000500002 M205 (SMALL HOLE)
*N10000500003 P198=5
*N10000500004 P125=24.556, P126=190.256
N10000500005 Q899996
N10000500006 G41 D100
N10000500007 G1 X22.435 Y188.135 F=P101
*N10000500008 IF P108 = 0 GO 10000500010
N10000500009 G4 F=P108
*N10000500010 P123=22.435, P124=188.135
*N10000500011 P125=40.939, P126=201.728
*N10000500012 P150=22.96
N10000500013 G40
N10000500014 Q899997
*N10000600001 P189=0, P190=0, P191=9, P192=9
N10000600002 M205 (SMALL HOLE)
*N10000600003 P198=6
*N10000600004 P125=40.939, P126=201.728
N10000600005 Q899996
N10000600006 G41 D100
N10000600007 G1 X38.818 Y199.606 F=P102 U20 O=P215
*N10000600008 IF P108 = 0 GO 10000600010
N10000600009 G4 F=P108
*N10000600010 P123=38.818, P124=199.606
*N10000600011 P125=77.647, P126=130.086
*N10000600012 P150=79.629
N10000600013 G40
N10000600014 Q899997
*N10000700001 P189=0, P190=0, P191=67.425, P192=31.537
N10000700002 M201 (CUT1)
*N10000700003 P198=7
*N10000700004 P125=77.647, P126=130.086
N10000700005 Q899996
N10000700006 G41 D100
N10000700007 G1 X75.189 Y128.366 F=P101
*N10000700008 IF P108 = 0 GO 10000700010
N10000700009 G4 F=P108
N10000700010 G3 X82.849 Y127.015 I4.506 J3.154 F=P102 U20 O=P215
N10000700011 G2 X136.599 Y146.578 I63.38 J-90.516
N10000700012 G3 X135.64 Y157.536 I-0.48 J5.479
N10000700013 G3 X76.54 Y136.026 I10.589 J-121.037
N10000700014 G3 X75.189 Y128.366 I3.155 J-4.506 M380
*N10000700015 P123=75.189, P124=128.366
*N10000700016 P125=30.889, P126=23.898
*N10000700017 P150=113.473
N10000700018 G40
N10000700019 Q899997
*N10000800001 P189=0, P190=0, P191=21.868, P192=70.134
N10000800002 M201 (CUT1)
*N10000800003 P198=8
*N10000800004 P125=30.889, P126=23.898
N10000800005 Q899996
N10000800006 G41 D100
N10000800007 G1 X31.15 Y20.91 F=P101
*N10000800008 IF P108 = 0 GO 10000800010
N10000800009 G4 F=P108
N10000800010 G3 X36.15 Y26.868 I-0.479 J5.479 F=P102 U20 O=P215
N10000800011 G2 X46.082 Y83.198 I110.079 J9.631
N10000800012 G3 X36.113 Y87.847 I-4.984 J2.324
N10000800013 G3 X25.192 Y25.909 I110.116 J-51.348
N10000800014 G3 X31.15 Y20.91 I5.479 J0.48 M380
*N10000800015 P123=31.15, P124=20.91
*N10000800016 P125=23.484, P126=211.797
*N10000800017 P150=191.041
N10000800018 G40
N10000800019 Q899997
*N10000900001 P189=0, P190=0, P191=142.619, P192=209.633
N10000900002 M201 (CUT1)
*N10000900003 P198=9
*N10000900004 P125=23.484, P126=211.797
N10000900005 Q899996
N10000900006 G41 D100
N10000900007 G1 X25.205 Y209.34 F=P101
*N10000900008 IF P108 = 0 GO 10000900010
N10000900009 G4 F=P108
N10000900010 G1 X39.13 Y219.091 F=P102 U20 O=P215
N10000900011 G2 X43.309 Y218.354 I1.721 J-2.458
N10000900012 G1 X76.126 Y171.486
N10000900013 G3 X99.396 Y164.181 I16.383 J11.471
N10000900014 G2 X146.649 Y172.498 I46.833 J-127.682
N10000900015 G2 X149.628 Y169.76 I-0.01 J-3
N10000900016 G1 X152.608 Y135.702
N10000900017 G2 X149.516 Y132.442 I-2.989 J-0.261
N10000900018 G3 X52.314 Y16.601 I-3.287 J-95.943
N10000900019 G2 X49.641 Y12.991 I-2.935 J-0.621
N10000900020 G1 X15.583 Y10.011
N10000900021 G2 X12.369 Y12.47 I-0.261 J2.989
N10000900022 G2 X42.266 Y124.177 I133.86 J24.029
N10000900023 G3 X43.36 Y148.543 I-15.289 J12.894
N10000900024 G1 X10.543 Y195.411
N10000900025 G2 X11.279 Y199.589 I2.457 J1.721
N10000900026 G1 X25.205 Y209.34 M380
*N10000900027 P123=25.205, P124=209.34
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