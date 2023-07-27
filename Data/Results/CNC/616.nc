N000000001:( ******* LASER CUTTING MACHINE    ****** )
N000000002 ( *************************************** )
N000000003 ( DATE      : 22/06/2022  )
N000000004 ( MATERIAL    : Al99  )
N000000005 ( THICKNESS   : 0.8  )
N000000006 ( SHEET SIZE  : 108 x 108  )
N000000007 ( TOTAL TIME   : 00:00:07  )
N000000008 ( *************************************** )
N000000009 V.E.Par[55] = 108
N000000010 V.E.Par[56] = 108
N000000011 V.E.Par[46] = 0
N000000012 V.E.Par[47] = 0.8
N000000013 V.E.Par[48] = 0
N000000014 V.E.Par[49] = 1
N000000015 V.E.Par[50] = 3
N000000016 V.E.Par[130] = 0
N000000017 V.E.Par[131] = 0
N000000018 M40=0
N000000019 G810
N000000020 G808
N000000021 G54
N000000022 $IF V.E.Kparams[0] == 1
N000000023 M121
N000000024 $GOTO V.E.LineNumber
N000000025 $ENDIF
N000000026 ( ************** START ***************** )
N000000027 ;( PART NAME 1    :Pieza_prueba_Maxwell)
N000000028 [100002] V.E.Kparams[2] = 1 V.E.Kparams[3] = 2
N000000029 V.E.Par[80] = 68.94 V.E.Par[81] = 39.24
N000000030 V.E.Par[82] = 71.68 V.E.Par[83] = 36.51
N000000031 M111 M101
N000000032 G812
N000000033 $IF V.E.Kparams[15] == 1 OR V.E.Kparams[15] == 2
N000000034 V.E.Par[17] = 68.94 V.E.Par[18] = 39.24
N000000035 V.E.Par[15] = 36.32 V.E.Par[16] = 71.68
N000000036 M111 M101
N000000037 G81
N000000038 $GOTO [100002]
N000000039 $ENDIF
N000000040 $IF V.E.Par[78] == 0
N000000041 D1 G41
N000000042 G01 X71.68 Y36.51 FP60
N000000043 $IF V.E.Par[84] == 1
N000000044 M43
N000000045 $ENDIF
N000000046 $ENDIF
N000000047 D1 G41
N000000048 G03 X36.41 Y71.59 I-17.68 J17.49 FP70
N000000049 G03 X71.59 Y36.41 I17.59 J-17.59
N000000050 G804
N000000051 G40 G01 X71.59 Y36.41
N000000052 V.E.Par[17] = 71.59 V.E.Par[18] = 36.41
N000000053 V.E.Par[15] = 36.32 V.E.Par[16] = 71.68
N000000054 V.E.Par[82] = 36.32 V.E.Par[83] = 71.68
N000000055 M111 M101
N000000056 G81
N000000057 [100003] V.E.Kparams[2] = 1 V.E.Kparams[3] = 3
N000000058 V.E.Par[80] = 36.32 V.E.Par[81] = 71.68
N000000059 V.E.Par[82] = 36.32 V.E.Par[83] = 71.68
N000000060 M111 M101
N000000061 G812
N000000062 $IF V.E.Kparams[15] == 1 OR V.E.Kparams[15] == 2
N000000063 V.E.Par[17] = 36.32 V.E.Par[18] = 71.68
N000000064 V.E.Par[15] = 108 V.E.Par[16] = 3.87
N000000065 M111 M101
N000000066 G81
N000000067 $GOTO [100003]
N000000068 $ENDIF
N000000069 $IF V.E.Par[78] == 0
N000000070 G01 X4 Y104 FP70
N000000071 $IF V.E.Par[84] == 1
N000000072 M43
N000000073 $ENDIF
N000000074 $ENDIF
N000000075 G804
N000000076 G40 G01 X4 Y104
N000000077 V.E.Par[17] = 4 V.E.Par[18] = 104
N000000078 V.E.Par[15] = 108 V.E.Par[16] = 3.87
N000000079 V.E.Par[82] = 104 V.E.Par[83] = 3.87
N000000080 M111 M101
N000000081 G81
N000000082 [100004] V.E.Kparams[2] = 1 V.E.Kparams[3] = 4
N000000083 V.E.Par[80] = 108 V.E.Par[81] = 3.87
N000000084 V.E.Par[82] = 104 V.E.Par[83] = 3.87
N000000085 M111 M101
N000000086 G812
N000000087 $IF V.E.Kparams[15] == 1 OR V.E.Kparams[15] == 2
N000000088 V.E.Par[17] = 108 V.E.Par[18] = 3.87
N000000089 V.E.Par[15] = 68.94 V.E.Par[16] = 39.24
N000000090 M111 M101
N000000091 G81
N000000092 G818
N000000093 $GOTO [100001]
N000000094 $ENDIF
N000000095 $IF V.E.Par[78] == 0
N000000096 D2 G41
N000000097 G01 X104 Y3.87 FP60
N000000098 $IF V.E.Par[84] == 1
N000000099 M43
N000000100 $ENDIF
N000000101 $ENDIF
N000000102 D2 G41
N000000103 G01 X14 Y3.87 FP70
N000000104 G02 X3.87 Y14 I0 J10.13
N000000105 G01 X3.87 Y104.13
N000000106 G01 X94 Y104.13
N000000107 G02 X104.13 Y94 I0 J-10.13
N000000108 G01 X104.13 Y4
N000000109 G01 X104.13 Y2
N000000110 G804
N000000111 G40 D0
N000000112 G809
N000000113 G0 X0 Y0
N000000114 G53
N000000115 $IF V.E.Kparams[10] == 1
N000000116 G819
N000000117 $GOTO N000000001
N000000118 $ENDIF
N000000119 M30