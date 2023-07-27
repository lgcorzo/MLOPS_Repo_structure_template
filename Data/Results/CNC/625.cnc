
%
N10 (DATE: 06/22/22)
N11 (MATERIAL: Al99)
N12 (THICKNESS: 0.8)
N13 (SHEET SIZE [mm]:209.6332 x 142.619)
*N10 P187=210
*N11 P188=143
*N12 P193=0
*N13 P194=0
N14 G71 M271
*N15 P173=-0.13
*N16 P174=-0.13
*N17 P175=209.763
*N18 P176=142.749
*N19 P177=209.893
*N20 P183=142.879
*N21 P150=143.158 /(LENGHT OF FIRST RAPID MOVEMENT)
*N22 P151=9 /(NUMBER OF CONTOURS)
*N23 P152=667.805 /(TOTAL RAPID MOVEMENT LENGHT)
*N24 P153=0 /(CUT1 LENGHT)
*N25 P154=0 /(CUT2 LENGHT)
*N26 P155=1158.163 /(CUT3 LENGHT)
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

N100010001 (PART NAME:DXFPAR0105)
*N100010002 P199=1
*N100010003 P178=142.143
*N100010004 P179=17.009
*N100010005 P189=0
*N100010006 P190=0
*N100010007 P191=209.633
*N100010008 P192=142.619
*N100010009 P194=9
*N100010010 P198=1
*N100010011 P193=1
N100010012 Q899993
N100010013 G0 X142.143 Y17.009
N100010014 M203 (NORMAL INSIDE CUTTING)
*N100010015 P198 = 1
N100010016 Q899996
N100010017 G41 D100
N100010018 G1 X142.655 Y11.163 F=P101
*N100010019 IF P108 = 0 GO 100010021
N100010020 G4 F=P108
N100010021 G3 X147.407 Y16.967 I-0.597 J5.337 F=P102 U20
N100010022 G3 X125.919 Y76.005 I-120.909 J-10.578
N100010023 G3 X117.121 Y69.844 I-4.398 J-3.081
N100010024 G2 X136.708 Y16.031 I-90.623 J-63.454
N100010025 G3 X142.525 Y11.15 I5.349 J0.468
*N100010026 P123=142.525, P124=11.15
*N100010027 P125=146.599, P126=78.304
*N100010028 P150=67.278
N100010029 G40
N100010030 Q899997
*N100020001 P125=146.599, P126=78.304
N100020002 M203 (NORMAL INSIDE CUTTING)
*N100020003 P198 = P198+1
N100020004 Q899996
N100020005 G41 D100
N100020006 G1 X149.687 Y75.215 F=P101
*N100020007 IF P108 = 0 GO 100020009
N100020008 G4 F=P108
N100020009 G3 X143.417 Y81.302 I-3.181 J2.996 F=P102 U20
N100020010 G3 X149.596 Y75.122 I3.09 J-3.091
*N100020011 P123=149.596, P124=75.122
*N100020012 P125=163.801, P126=90.349
*N100020013 P150=20.824
N100020014 G40
N100020015 Q899997
*N100030001 P125=163.801, P126=90.349
N100030002 M203 (NORMAL INSIDE CUTTING)
*N100030003 P198 = P198+1
N100030004 Q899996
N100030005 G41 D100
N100030006 G1 X166.889 Y87.26 F=P101
*N100030007 IF P108 = 0 GO 100030009
N100030008 G4 F=P108
N100030009 G3 X160.618 Y93.347 I-3.181 J2.997 F=P102 U20
N100030010 G3 X166.799 Y87.167 I3.09 J-3.09
*N100030011 P123=166.799, P124=87.167
*N100030012 P125=135.127, P126=94.687
*N100030013 P150=32.552
N100030014 G40
N100030015 Q899997
*N100040001 P125=135.127, P126=94.687
N100040002 M203 (NORMAL INSIDE CUTTING)
*N100040003 P198 = P198+1
N100040004 Q899996
N100040005 G41 D100
N100040006 G1 X138.216 Y91.598 F=P101
*N100040007 IF P108 = 0 GO 100040009
N100040008 G4 F=P108
N100040009 G3 X131.945 Y97.685 I-3.181 J2.996 F=P102 U20
N100040010 G3 X138.125 Y91.505 I3.09 J-3.091
*N100040011 P123=138.125, P124=91.505
*N100040012 P125=152.329, P126=106.732
*N100040013 P150=20.823
N100040014 G40
N100040015 Q899997
*N100050001 P125=152.329, P126=106.732
N100050002 M203 (NORMAL INSIDE CUTTING)
*N100050003 P198 = P198+1
N100050004 Q899996
N100050005 G41 D100
N100050006 G1 X155.418 Y103.643 F=P101
*N100050007 IF P108 = 0 GO 100050009
N100050008 G4 F=P108
N100050009 G3 X149.147 Y109.73 I-3.181 J2.997 F=P102 U20
N100050010 G3 X155.327 Y103.55 I3.09 J-3.09
*N100050011 P123=155.327, P124=103.55
*N100050012 P125=192.88, P126=110.71
*N100050013 P150=38.23
N100050014 G40
N100050015 Q899997
*N100060001 P125=192.88, P126=110.71
N100060002 M203 (NORMAL INSIDE CUTTING)
*N100060003 P198 = P198+1
N100060004 Q899996
N100060005 G41 D100
N100060006 G1 X195.969 Y107.622 F=P101
*N100060007 IF P108 = 0 GO 100060009
N100060008 G4 F=P108
N100060009 G3 X189.698 Y113.709 I-3.181 J2.997 F=P102 U20
N100060010 G3 X195.879 Y107.529 I3.09 J-3.09
*N100060011 P123=195.879, P124=107.529
*N100060012 P125=75.124, P126=111.85
*N100060013 P150=120.832
N100060014 G40
N100060015 Q899997
*N100070001 P125=75.124, P126=111.85
N100070002 M203 (NORMAL INSIDE CUTTING)
*N100070003 P198 = P198+1
N100070004 Q899996
N100070005 G41 D100
N100070006 G1 X80.443 Y109.37 F=P101
*N100070007 IF P108 = 0 GO 100070009
N100070008 G4 F=P108
N100070009 G3 X77.792 Y116.388 I-4.92 J2.151 F=P102 U20
N100070010 G3 X15.921 Y127.298 I-51.294 J-109.998
N100070011 G3 X16.856 Y116.598 I0.468 J-5.349
N100070012 G2 X73.253 Y106.655 I9.642 J-110.209
N100070013 G3 X80.389 Y109.251 I2.27 J4.867
*N100070014 P123=80.389, P124=109.251
*N100070015 P125=181.409, P126=127.093
*N100070016 P150=102.583
N100070017 G40
N100070018 Q899997
*N100080001 P125=181.409, P126=127.093
N100080002 M203 (NORMAL INSIDE CUTTING)
*N100080003 P198 = P198+1
N100080004 Q899996
N100080005 G41 D100
N100080006 G1 X184.497 Y124.005 F=P101
*N100080007 IF P108 = 0 GO 100080009
N100080008 G4 F=P108
N100080009 G3 X178.227 Y130.092 I-3.181 J2.997 F=P102 U20
N100080010 G3 X184.407 Y123.912 I3.09 J-3.09
*N100080011 P123=184.407, P124=123.912
*N100080012 P125=114.683, P126=24.379
*N100080013 P150=121.525
N100080014 G40
N100080015 Q899997
*N100090001 P125=114.683, P126=24.379
N100090002 M203 (NORMAL OUTSIDE CUTTING)
*N100090003 P198 = P198+1
N100090004 Q899996
N100090005 G41 D101
N100090006 G1 X120.436 Y25.543 F=P101
*N100090007 IF P108 = 0 GO 100090009
N100090008 G4 F=P108
N100090009 G3 X6.628 Y100.178 I-93.937 J-19.154 F=P102 U20
N100090010 G2 X2.861 Y102.967 I-0.649 J3.062
N100090011 G1 X-0.118 Y137.025
N100090012 G2 X2.447 Y140.378 I3.118 J0.272
N100090013 G2 X114.261 Y110.452 I24.052 J-133.988
N100090014 G3 X138.469 Y109.365 I12.81 J15.19
N100090015 G1 X185.337 Y142.183
N100090016 G2 X189.695 Y141.414 I1.795 J-2.564
N100090017 G1 X209.197 Y113.563
N100090018 G2 X208.428 Y109.204 I-2.564 J-1.795
N100090019 G1 X161.56 Y76.387
N100090020 G3 X154.303 Y53.268 I11.397 J-16.277
N100090021 G2 X162.628 Y5.97 I-127.805 J-46.878
N100090022 G2 X159.771 Y2.861 I-3.13 J0.009
N100090023 G1 X125.714 Y-0.118
N100090024 G2 X122.312 Y3.107 I-0.273 J3.118
N100090025 G3 X120.462 Y25.416 I-95.814 J3.283
*N100090026 P150=0
N100090027 G40
N100090028 Q899997
N100090029 (END OF PARTS)
N100090030 G0 X0.000 Y0.000
N100090031 Q899994
*N100090032 P193=0,P194=0,P198=0,P199=0,P160=0
N100090033 G53 (MACHINE COORDINATE SYSTEM ON)
N100090034 M30
