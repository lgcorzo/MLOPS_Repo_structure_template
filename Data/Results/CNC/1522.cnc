
%
P1 (LANTEK)
N10 (Nukon Fiber Laser PA8000)
N11 (SHEET SIZE:[mm]: 313 x 182)
N12 (MATERIAL: Al99)
N13 (THICKNESS:[mm]: 0.8)
N14 (STRATEGY: Air/150)
N15 (DATE: 22/06/2022 12:35:41)
N16 (RUNTIME:[hour min sec]: 00:00:26)
N17 (SHEET WEIGHT:[kg]: 0.123047)
N18 (FOCAL LENGTH: 0.00)
N19 (NOZZLE DIAMETER: 0.00)
*N20 P188=182 / (SHEET SIZE Y)
*N21 P187=313 / (SHEET SIZE X)
*N22 P193=1/(TOTAL NR OF PARTS IN JOB)
N23 G71 M271
*N24 P160=0
*N25 P166=0
*N26 P167=0
*N27 P171=0
*N28 P173=10/ (JOB BOX X MIN)
*N29 P174=10/ (JOB BOX Y MIN)
*N30 P175=172/ (JOB BOX X MAX)
*N31 P176=303/ (JOB BOX Y MAX)
*N32 P177=162/ (JOB WIDTH)
*N33 P183=293/ (JOB HEIGHT)
*N34 P150=126.916 /(LENGHT OF FIRST RAPID MOVEMENT)
*N35 P151=22 /(NUMBER OF CONTOURS)
*N36 P152=975.341 /(TOTAL RAPID MOVEMENT LENGHT)
*N37 P153=898.856 /(CUT1 LENGHT)
*N38 P154=0 /(CUT2 LENGHT)
*N39 P155=0 /(CUT3 LENGHT)
*N40 P156=0 /(FILM BURNING LENGHT)
*N41 P157=442.076 /(HOLE LENGHT)
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

N10000100001 (PART NAME:DXFPUNC0)
*N10000100002 P199=1
*N10000100003 P198=1
*N10000100004 P178=10, P179=10, P194=22
*N10000100005 P189=0, P190=0, P191=162, P192=293
N10000100006 Q899993
*N10000100007 P189=0, P190=0, P191=6.2, P192=6.2
N10000100008 M205 (SMALL HOLE)
*N10000100009 P198 = 1
N10000100010 G0 X121.929 Y35.229
N10000100011 Q899996
N10000100012 G41 D100
N10000100013 G1 X119.808 Y33.108 F=P101
*N10000100014 IF P108 = 0 GO 10000100016
N10000100015 G4 F=P108
*N10000100016 P123=119.808, P124=33.108
*N10000100017 P125=133, P126=52.6
*N10000100018 P150=23.537
N10000100019 G40
N10000100020 Q899997
*N10000200001 P189=0, P190=0, P191=4.5, P192=4.5
N10000200002 M205 (SMALL HOLE)
*N10000200003 P198=2
*N10000200004 P125=133, P126=52.6
N10000200005 Q899996
N10000200006 G41 D100
N10000200007 G1 X131.409 Y51.009 F=P102 U20 O=P215
*N10000200008 IF P108 = 0 GO 10000200010
N10000200009 G4 F=P108
*N10000200010 P123=131.409, P124=51.009
*N10000200011 P125=112.429, P126=58.972
*N10000200012 P150=20.583
N10000200013 G40
N10000200014 Q899997
*N10000300001 P189=0, P190=0, P191=4.5, P192=4.5
N10000300002 M205 (SMALL HOLE)
*N10000300003 P198=3
*N10000300004 P125=112.429, P126=58.972
N10000300005 Q899996
N10000300006 G41 D100
N10000300007 G1 X110.838 Y57.381 F=P101
*N10000300008 IF P108 = 0 GO 10000300010
N10000300009 G4 F=P108
*N10000300010 P123=110.838, P124=57.381
*N10000300011 P125=118.964, P126=69.036
*N10000300012 P150=14.208
N10000300013 G40
N10000300014 Q899997
*N10000400001 P189=0, P190=0, P191=3.2, P192=3.2
N10000400002 M205 (SMALL HOLE)
*N10000400003 P198=4
*N10000400004 P125=118.964, P126=69.036
N10000400005 Q899996
N10000400006 G41 D100
N10000400007 G1 X117.833 Y67.905 F=P102 U20 O=P215
*N10000400008 IF P108 = 0 GO 10000400010
N10000400009 G4 F=P108
*N10000400010 P123=117.833, P124=67.905
*N10000400011 P125=125.5, P126=79.1
*N10000400012 P150=13.569
N10000400013 G40
N10000400014 Q899997
*N10000500001 P189=0, P190=0, P191=4.5, P192=4.5
N10000500002 M205 (SMALL HOLE)
*N10000500003 P198=5
*N10000500004 P125=125.5, P126=79.1
N10000500005 Q899996
N10000500006 G41 D100
N10000500007 G1 X123.909 Y77.509 F=P101
*N10000500008 IF P108 = 0 GO 10000500010
N10000500009 G4 F=P108
*N10000500010 P123=123.909, P124=77.509
*N10000500011 P125=98.096, P126=60.771
*N10000500012 P150=30.765
N10000500013 G40
N10000500014 Q899997
*N10000600001 P189=0, P190=0, P191=27.701, P192=37.257
N10000600002 M205 (SMALL HOLE)
*N10000600003 P198=6
*N10000600004 P125=98.096, P126=60.771
N10000600005 Q899996
N10000600006 G41 D100
N10000600007 G1 X96.462 Y58.255 F=P102 U20 O=P215
*N10000600008 IF P108 = 0 GO 10000600010
N10000600009 G4 F=P108
N10000600010 G3 X103.379 Y59.725 I2.724 J4.193
N10000600011 G1 X121.08 Y86.982
N10000600012 G3 X112.693 Y92.428 I-4.194 J2.723
N10000600013 G1 X94.992 Y65.171
N10000600014 G3 X96.462 Y58.255 I4.194 J-2.723 M380
*N10000600015 P123=96.462, P124=58.255
*N10000600016 P125=133, P126=99.6
*N10000600017 P150=55.176
N10000600018 G40
N10000600019 Q899997
*N10000700001 P189=0, P190=0, P191=4.5, P192=4.5
N10000700002 M205 (SMALL HOLE)
*N10000700003 P198=7
*N10000700004 P125=133, P126=99.6
N10000700005 Q899996
N10000700006 G41 D100
N10000700007 G1 X131.409 Y98.009 F=P101
*N10000700008 IF P108 = 0 GO 10000700010
N10000700009 G4 F=P108
*N10000700010 P123=131.409, P124=98.009
*N10000700011 P125=90.7, P126=105
*N10000700012 P150=41.305
N10000700013 G40
N10000700014 Q899997
*N10000800001 P189=0, P190=0, P191=10, P192=6.5
N10000800002 M205 (SMALL HOLE)
*N10000800003 P198=8
*N10000800004 P125=90.7, P126=105
N10000800005 Q899996
N10000800006 G41 D100
N10000800007 G1 X87.7 Y105 F=P102 U20 O=P215
*N10000800008 IF P108 = 0 GO 10000800010
N10000800009 G4 F=P108
N10000800010 G1 X87.7 Y101.75
N10000800011 G1 X97.7 Y101.75
N10000800012 G1 X97.7 Y108.25
N10000800013 G1 X87.7 Y108.25
N10000800014 G1 X87.7 Y105 M380
*N10000800015 P123=87.7, P124=105
*N10000800016 P125=56.212, P126=76.42
*N10000800017 P150=42.524
N10000800018 G40
N10000800019 Q899997
*N10000900001 P189=0, P190=0, P191=5.538, P192=5.538
N10000900002 M205 (SMALL HOLE)
*N10000900003 P198=9
*N10000900004 P125=56.212, P126=76.42
N10000900005 Q899996
N10000900006 G41 D100
N10000900007 G1 X55.929 Y76.137 F=P101
*N10000900008 IF P108 = 0 GO 10000900010
N10000900009 G4 F=P108
N10000900010 G1 X58.298 Y73.769 F=P102 U20 O=P215
N10000900011 G3 X58.863 Y74.334 I0.283 J0.282
N10000900012 G1 X54.126 Y79.072
N10000900013 G3 X53.56 Y78.506 I-0.283 J-0.283
N10000900014 G1 X55.929 Y76.137 M380
*N10000900015 P123=55.929, P124=76.137
*N10000900016 P125=94.429, P126=42.429
*N10000900017 P150=51.171
N10000900018 G40
N10000900019 Q899997
*N10001000001 P189=0, P190=0, P191=6.2, P192=6.2
N10001000002 M205 (SMALL HOLE)
*N10001000003 P198=10
*N10001000004 P125=94.429, P126=42.429
N10001000005 Q899996
N10001000006 G41 D100
N10001000007 G1 X92.308 Y40.308 F=P101
*N10001000008 IF P108 = 0 GO 10001000010
N10001000009 G4 F=P108
*N10001000010 P123=92.308, P124=40.308
*N10001000011 P125=24, P126=116.5
*N10001000012 P150=102.329
N10001000013 G40
N10001000014 Q899997
*N10001100001 P189=0, P190=0, P191=3.5, P192=3.5
N10001100002 M205 (SMALL HOLE)
*N10001100003 P198=11
*N10001100004 P125=24, P126=116.5
N10001100005 Q899996
N10001100006 G41 D100
N10001100007 G1 X22.763 Y115.263 F=P102 U20 O=P215
*N10001100008 IF P108 = 0 GO 10001100010
N10001100009 G4 F=P108
*N10001100010 P123=22.763, P124=115.263
*N10001100011 P125=23.042, P126=156.5
*N10001100012 P150=41.238
N10001100013 G40
N10001100014 Q899997
*N10001200001 P189=0, P190=0, P191=0.8, P192=7.5
N10001200002 M205 (SMALL HOLE)
*N10001200003 P198=12
*N10001200004 P125=23.042, P126=156.5
N10001200005 Q899996
N10001200006 G41 D100
N10001200007 G1 X22.642 Y156.5 F=P101
*N10001200008 IF P108 = 0 GO 10001200010
N10001200009 G4 F=P108
N10001200010 G1 X22.642 Y153.15 F=P102 U20 O=P215
N10001200011 G3 X23.442 Y153.15 I0.4 J0
N10001200012 G1 X23.442 Y159.85
N10001200013 G3 X22.642 Y159.85 I-0.4 J0
N10001200014 G1 X22.642 Y156.5 M380
*N10001200015 P123=22.642, P124=156.5
*N10001200016 P125=24, P126=196.5
*N10001200017 P150=40.023
N10001200018 G40
N10001200019 Q899997
*N10001300001 P189=0, P190=0, P191=3.5, P192=3.5
N10001300002 M205 (SMALL HOLE)
*N10001300003 P198=13
*N10001300004 P125=24, P126=196.5
N10001300005 Q899996
N10001300006 G41 D100
N10001300007 G1 X22.763 Y195.263 F=P101
*N10001300008 IF P108 = 0 GO 10001300010
N10001300009 G4 F=P108
*N10001300010 P123=22.763, P124=195.263
*N10001300011 P125=56.212, P126=236.58
*N10001300012 P150=53.159
N10001300013 G40
N10001300014 Q899997
*N10001400001 P189=0, P190=0, P191=5.538, P192=5.538
N10001400002 M205 (SMALL HOLE)
*N10001400003 P198=14
*N10001400004 P125=56.212, P126=236.58
N10001400005 Q899996
N10001400006 G41 D100
N10001400007 G1 X55.929 Y236.863 F=P102 U20 O=P215
*N10001400008 IF P108 = 0 GO 10001400010
N10001400009 G4 F=P108
N10001400010 G1 X53.56 Y234.494
N10001400011 G3 X54.126 Y233.928 I0.283 J-0.283
N10001400012 G1 X58.863 Y238.666
N10001400013 G3 X58.298 Y239.231 I-0.282 J0.283
N10001400014 G1 X55.929 Y236.863 M380
*N10001400015 P123=55.929, P124=236.863
*N10001400016 P125=90.7, P126=208
*N10001400017 P150=45.19
N10001400018 G40
N10001400019 Q899997
*N10001500001 P189=0, P190=0, P191=10, P192=6.5
N10001500002 M205 (SMALL HOLE)
*N10001500003 P198=15
*N10001500004 P125=90.7, P126=208
N10001500005 Q899996
N10001500006 G41 D100
N10001500007 G1 X87.7 Y208 F=P101
*N10001500008 IF P108 = 0 GO 10001500010
N10001500009 G4 F=P108
N10001500010 G1 X87.7 Y204.75 F=P102 U20 O=P215
N10001500011 G1 X97.7 Y204.75
N10001500012 G1 X97.7 Y211.25
N10001500013 G1 X87.7 Y211.25
N10001500014 G1 X87.7 Y208 M380
*N10001500015 P123=87.7, P124=208
*N10001500016 P125=94.429, P126=270.429
*N10001500017 P150=62.791
N10001500018 G40
N10001500019 Q899997
*N10001600001 P189=0, P190=0, P191=6.2, P192=6.2
N10001600002 M205 (SMALL HOLE)
*N10001600003 P198=16
*N10001600004 P125=94.429, P126=270.429
N10001600005 Q899996
N10001600006 G41 D100
N10001600007 G1 X92.308 Y268.308 F=P101
*N10001600008 IF P108 = 0 GO 10001600010
N10001600009 G4 F=P108
*N10001600010 P123=92.308, P124=268.308
*N10001600011 P125=121.929, P126=277.629
*N10001600012 P150=31.053
N10001600013 G40
N10001600014 Q899997
*N10001700001 P189=0, P190=0, P191=6.2, P192=6.2
N10001700002 M205 (SMALL HOLE)
*N10001700003 P198=17
*N10001700004 P125=121.929, P126=277.629
N10001700005 Q899996
N10001700006 G41 D100
N10001700007 G1 X119.808 Y275.508 F=P102 U20 O=P215
*N10001700008 IF P108 = 0 GO 10001700010
N10001700009 G4 F=P108
*N10001700010 P123=119.808, P124=275.508
*N10001700011 P125=162, P126=224.6
*N10001700012 P150=66.12
N10001700013 G40
N10001700014 Q899997
*N10001800001 P189=0, P190=0, P191=2, P192=2
N10001800002 M205 (SMALL HOLE)
*N10001800003 P198=18
*N10001800004 P125=162, P126=224.6
N10001800005 Q899996
N10001800006 G41 D100
N10001800007 G1 X161.293 Y223.893 F=P101
*N10001800008 IF P108 = 0 GO 10001800010
N10001800009 G4 F=P108
*N10001800010 P123=161.293, P124=223.893
*N10001800011 P125=162, P126=221.2
*N10001800012 P150=2.784
N10001800013 G40
N10001800014 Q899997
*N10001900001 P189=0, P190=0, P191=2, P192=2
N10001900002 M205 (SMALL HOLE)
*N10001900003 P198=19
*N10001900004 P125=162, P126=221.2
N10001900005 Q899996
N10001900006 G41 D100
N10001900007 G1 X161.293 Y220.493 F=P102 U20 O=P215
*N10001900008 IF P108 = 0 GO 10001900010
N10001900009 G4 F=P108
*N10001900010 P123=161.293, P124=220.493
*N10001900011 P125=162, P126=217.8
*N10001900012 P150=2.784
N10001900013 G40
N10001900014 Q899997
*N10002000001 P189=0, P190=0, P191=2, P192=2
N10002000002 M205 (SMALL HOLE)
*N10002000003 P198=20
*N10002000004 P125=162, P126=217.8
N10002000005 Q899996
N10002000006 G41 D100
N10002000007 G1 X161.293 Y217.093 F=P101
*N10002000008 IF P108 = 0 GO 10002000010
N10002000009 G4 F=P108
*N10002000010 P123=161.293, P124=217.093
*N10002000011 P125=162, P126=214.4
*N10002000012 P150=2.784
N10002000013 G40
N10002000014 Q899997
*N10002100001 P189=0, P190=0, P191=2, P192=2
N10002100002 M205 (SMALL HOLE)
*N10002100003 P198=21
*N10002100004 P125=162, P126=214.4
N10002100005 Q899996
N10002100006 G41 D100
N10002100007 G1 X161.293 Y213.693 F=P102 U20 O=P215
*N10002100008 IF P108 = 0 GO 10002100010
N10002100009 G4 F=P108
*N10002100010 P123=161.293, P124=213.693
*N10002100011 P125=76, P126=275.5
*N10002100012 P150=105.333
N10002100013 G40
N10002100014 Q899997
*N10002200001 P189=0, P190=0, P191=162, P192=293
N10002200002 M201 (CUT1)
*N10002200003 P198=22
*N10002200004 P125=76, P126=275.5
N10002200005 Q899996
N10002200006 G41 D100
N10002200007 G1 X79 Y275.5 F=P101
*N10002200008 IF P108 = 0 GO 10002200010
N10002200009 G4 F=P108
N10002200010 G1 X108.3 Y275.5 F=P102 U20 O=P215
N10002200011 G1 X115.5 Y282.7
N10002200012 G1 X127.5 Y282.7
N10002200013 G1 X127.5 Y292.3
N10002200014 G1 X140 Y292.3
N10002200015 G1 X140 Y303
N10002200016 G1 X156.5 Y303
N10002200017 G1 X156.5 Y293.5
N10002200018 G1 X169.5 Y293.5
N10002200019 G1 X169.5 Y282.7
N10002200020 G1 X172 Y282.7
N10002200021 G1 X172 Y232.5
N10002200022 G1 X162 Y232.5
N10002200023 G1 X162 Y226
N10002200024 G1 X172 Y226
N10002200025 G1 X172 Y213
N10002200026 G1 X162 Y213
N10002200027 G1 X162 Y206.5
N10002200028 G1 X172 Y206.5
N10002200029 G1 X172 Y106
N10002200030 G1 X160 Y106
N10002200031 G1 X160 Y46
N10002200032 G1 X172 Y46
N10002200033 G1 X172 Y30.3
N10002200034 G1 X169.5 Y30.3
N10002200035 G1 X169.5 Y19.5
N10002200036 G1 X156.5 Y19.5
N10002200037 G1 X156.5 Y10
N10002200038 G1 X140 Y10
N10002200039 G1 X140 Y20.7
N10002200040 G1 X127.5 Y20.7
N10002200041 G1 X127.5 Y30.3
N10002200042 G1 X115.5 Y30.3
N10002200043 G1 X108.3 Y37.5
N10002200044 G1 X79 Y37.5
N10002200045 G1 X19.5 Y97
N10002200046 G1 X19.5 Y121
N10002200047 G1 X10 Y121
N10002200048 G1 X10 Y192
N10002200049 G1 X19.5 Y192
N10002200050 G1 X19.5 Y216
N10002200051 G1 X79 Y275.5 M380
*N10002200052 P123=79, P124=275.5
*N10002200053 P125=0.000, P126=0.000
*N10002200054 P150=0
N10002200055 G40
N10002200056 Q899997
N10002200057 (END OF PARTS)
*N10002200058 P193=0,P194=0,P197=0,P198=0,P199=0
N10002200059 Q899994
N10002200060 G53 (MACHINE COORDINATE SYSTEM ON)
N10002200061 M30
%
