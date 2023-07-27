N000000001:( ******* LASER CUTTING MACHINE    ****** )
N000000002 ( *************************************** )
N000000003 ( DATE      : 22/06/2022  )
N000000004 ( MATERIAL    : Al99  )
N000000005 ( THICKNESS   : 0.8  )
N000000006 ( SHEET SIZE  : 114 x 114  )
N000000007 ( TOTAL TIME   : 00:00:05  )
N000000008 ( *************************************** )
N000000009 V.E.Par[55] = 114
N000000010 V.E.Par[56] = 114
N000000011 V.E.Par[46] = 0
N000000012 V.E.Par[47] = 0.8
N000000013 V.E.Par[48] = 1
N000000014 V.E.Par[49] = 1
N000000015 V.E.Par[50] = 3
N000000016 M40=0
N000000017 G810
N000000018 G808
N000000019 G54
N000000020 $IF V.E.Kparams[0] == 1
N000000021 M121
N000000022 $GOTO V.E.LineNumber
N000000023 $ENDIF
N000000024 ( ************** START ***************** )
N000000025 ;( PART NAME 1    :Pieza_prueba_Maxwell)
N000000026 [100002] V.E.Kparams[2] = 1 V.E.Kparams[3] = 2
N000000027 V.E.Par[80] = 72.63 V.E.Par[81] = 41.51
N000000028 V.E.Par[82] = 74.68 V.E.Par[83] = 39.46
N000000029 M111 M101
N000000030 G812
N000000031 $IF V.E.Kparams[15] == 1 OR V.E.Kparams[15] == 2
N000000032 V.E.Par[17] = 72.63 V.E.Par[18] = 41.51
N000000033 V.E.Par[15] = 39.32 V.E.Par[16] = 74.68
N000000034 M111 M101
N000000035 G81
N000000036 $GOTO [100002]
N000000037 $ENDIF
N000000038 $IF V.E.Par[78] == 0
N000000039 D1 G41
N000000040 G01 X74.68 Y39.46 FP60
N000000041 $IF V.E.Par[84] == 1
N000000042 M43
N000000043 $ENDIF
N000000044 $ENDIF
N000000045 D1 G41
N000000046 G03 X39.39 Y74.61 I-17.68 J17.54 FP70
N000000047 G03 X74.61 Y39.39 I17.61 J-17.61
N000000048 G804
N000000049 G40 G01 X74.61 Y39.39
N000000050 V.E.Par[17] = 74.61 V.E.Par[18] = 39.39
N000000051 V.E.Par[15] = 39.32 V.E.Par[16] = 74.68
N000000052 V.E.Par[82] = 39.32 V.E.Par[83] = 74.68
N000000053 M111 M101
N000000054 G81
N000000055 [100003] V.E.Kparams[2] = 1 V.E.Kparams[3] = 3
N000000056 V.E.Par[80] = 39.32 V.E.Par[81] = 74.68
N000000057 V.E.Par[82] = 39.32 V.E.Par[83] = 74.68
N000000058 M111 M101
N000000059 G812
N000000060 $IF V.E.Kparams[15] == 1 OR V.E.Kparams[15] == 2
N000000061 V.E.Par[17] = 39.32 V.E.Par[18] = 74.68
N000000062 V.E.Par[15] = 112 V.E.Par[16] = 6.9
N000000063 M111 M101
N000000064 G81
N000000065 $GOTO [100003]
N000000066 $ENDIF
N000000067 $IF V.E.Par[78] == 0
N000000068 G01 X7 Y107 FP70
N000000069 $IF V.E.Par[84] == 1
N000000070 M43
N000000071 $ENDIF
N000000072 $ENDIF
N000000073 G804
N000000074 G40 G01 X7 Y107
N000000075 V.E.Par[17] = 7 V.E.Par[18] = 107
N000000076 V.E.Par[15] = 112 V.E.Par[16] = 6.9
N000000077 V.E.Par[82] = 107 V.E.Par[83] = 6.9
N000000078 M111 M101
N000000079 G81
N000000080 [100004] V.E.Kparams[2] = 1 V.E.Kparams[3] = 4
N000000081 V.E.Par[80] = 112 V.E.Par[81] = 6.9
N000000082 V.E.Par[82] = 107 V.E.Par[83] = 6.9
N000000083 M111 M101
N000000084 G812
N000000085 $IF V.E.Kparams[15] == 1 OR V.E.Kparams[15] == 2
N000000086 V.E.Par[17] = 112 V.E.Par[18] = 6.9
N000000087 V.E.Par[15] = 72.63 V.E.Par[16] = 41.51
N000000088 M111 M101
N000000089 G81
N000000090 G818
N000000091 $GOTO [100001]
N000000092 $ENDIF
N000000093 $IF V.E.Par[78] == 0
N000000094 D2 G41
N000000095 G01 X107 Y6.9 FP60
N000000096 $IF V.E.Par[84] == 1
N000000097 M43
N000000098 $ENDIF
N000000099 $ENDIF
N000000100 D2 G41
N000000101 G01 X17 Y6.9 FP70
N000000102 G02 X6.9 Y17 I0 J10.1
N000000103 G01 X6.9 Y107.1
N000000104 G01 X97 Y107.1
N000000105 G02 X107.1 Y97 I0 J-10.1
N000000106 G01 X107.1 Y7
N000000107 G804
N000000108 G40 D0
N000000109 G809
N000000110 G0 X0 Y0
N000000111 G53
N000000112 $IF V.E.Kparams[10] == 1
N000000113 G819
N000000114 $GOTO N000000001
N000000115 $ENDIF
N000000116 M30
