// RUN: circt-opt -pass-pipeline="builtin.module(lower-firrtl-to-hw)" -verify-diagnostics %s | FileCheck %s
// RUN: circt-opt -pass-pipeline="builtin.module(lower-firrtl-to-hw{disable-mem-randomization})" -verify-diagnostics %s | FileCheck %s --check-prefix DISABLE_RANDOM --implicit-check-not RANDOMIZE_MEM
// RUN: circt-opt -pass-pipeline="builtin.module(lower-firrtl-to-hw{disable-reg-randomization})" -verify-diagnostics %s | FileCheck %s --check-prefix DISABLE_RANDOM --implicit-check-not RANDOMIZE_REG
// RUN: circt-opt -pass-pipeline="builtin.module(lower-firrtl-to-hw{disable-mem-randomization disable-reg-randomization})" -verify-diagnostics %s | FileCheck %s --check-prefix DISABLE_RANDOM --implicit-check-not RANDOMIZE_MEM --implicit-check-not RANDOMIZE_REG

// DISABLE_RANDOM-LABEL: module @Simple
firrtl.circuit "Simple"   attributes {annotations = [{class =
"sifive.enterprise.firrtl.ExtractAssumptionsAnnotation", directory = "dir1",  filename = "./dir1/filename1" }, {class =
"sifive.enterprise.firrtl.ExtractCoverageAnnotation", directory = "dir2",  filename = "./dir2/filename2" }, {class =
"sifive.enterprise.firrtl.ExtractAssertionsAnnotation", directory = "dir3",  filename = "./dir3/filename3" }]}
{
  // Headers
  // CHECK:      sv.ifdef  "PRINTF_COND_" {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   sv.ifdef  "PRINTF_COND" {
  // CHECK-NEXT:     sv.macro.def @PRINTF_COND_ "(`PRINTF_COND)"
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     sv.macro.def @PRINTF_COND_ "1"
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK:      sv.ifdef  "ASSERT_VERBOSE_COND_" {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   sv.ifdef  "ASSERT_VERBOSE_COND" {
  // CHECK-NEXT:     sv.macro.def @ASSERT_VERBOSE_COND_ "(`ASSERT_VERBOSE_COND)"
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     sv.macro.def @ASSERT_VERBOSE_COND_ "1"
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK:      sv.ifdef  "STOP_COND_" {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   sv.ifdef  "STOP_COND" {
  // CHECK-NEXT:     sv.macro.def @STOP_COND_ "(`STOP_COND)"
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     sv.macro.def @STOP_COND_ "1"
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK:      sv.ifdef  "INIT_RANDOM_PROLOG_" {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   sv.ifdef  "RANDOMIZE" {
  // CHECK-NEXT:     sv.ifdef  "VERILATOR" {
  // CHECK-NEXT:       sv.macro.def @INIT_RANDOM_PROLOG_ "`INIT_RANDOM"
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       sv.macro.def @INIT_RANDOM_PROLOG_ "`INIT_RANDOM #`RANDOMIZE_DELAY begin end"
  // CHECK-NEXT:     }
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     sv.macro.def @INIT_RANDOM_PROLOG_ ""
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  //These come from MemSimple, IncompleteRead, and MemDepth1
  // CHECK-LABEL: hw.generator.schema @FIRRTLMem, "FIRRTL_Memory", ["depth", "numReadPorts", "numWritePorts", "numReadWritePorts", "readLatency", "writeLatency", "width", "maskGran", "readUnderWrite", "writeUnderWrite", "writeClockIDs", "initFilename", "initIsBinary", "initIsInline"]
  // CHECK: hw.module.generated @aa_combMem, @FIRRTLMem(%W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i8, %W1_addr: i4, %W1_en: i1, %W1_clk: i1, %W1_data: i8) attributes {depth = 16 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 8 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 2 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : i32, width = 8 : ui32, writeClockIDs = [0 : i32, 0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  // CHECK: hw.module.generated @ab_combMem, @FIRRTLMem(%W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i8, %W1_addr: i4, %W1_en: i1, %W1_clk: i1, %W1_data: i8) attributes {depth = 16 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 8 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 2 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : i32, width = 8 : ui32, writeClockIDs = [0 : i32, 1 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  // CHECK: hw.module.generated @mem0_combMem, @FIRRTLMem(%R0_addr: i1, %R0_en: i1, %R0_clk: i1) -> (R0_data: i32) attributes {depth = 1 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 32 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 0 : ui32, readUnderWrite = 1 : i32, width = 32 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  // CHECK: hw.module.generated @_M_combMem, @FIRRTLMem(%R0_addr: i4, %R0_en: i1, %R0_clk: i1) -> (R0_data: i42) attributes {depth = 12 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 42 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : i32, width = 42 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  // CHECK: hw.module.generated @tbMemoryKind1_combMem, @FIRRTLMem(%R0_addr: i4, %R0_en: i1, %R0_clk: i1, %W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i8) -> (R0_data: i8) attributes {depth = 16 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 8 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : i32, width = 8 : ui32, writeClockIDs = [0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  // CHECK: hw.module.generated @_M_mask_combMem, @FIRRTLMem(%R0_addr: i10, %R0_en: i1, %R0_clk: i1, %RW0_addr: i10, %RW0_en: i1, %RW0_clk: i1, %RW0_wmode: i1, %RW0_wdata: i40, %RW0_wmask: i4, %W0_addr: i10, %W0_en: i1, %W0_clk: i1, %W0_data: i40, %W0_mask: i4) -> (R0_data: i40, RW0_rdata: i40) attributes {depth = 1022 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 10 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : i32, width = 40 : ui32, writeClockIDs = [0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
  // CHECK: hw.module.generated @_M_combMem_0, @FIRRTLMem(%R0_addr: i4, %R0_en: i1, %R0_clk: i1, %RW0_addr: i4, %RW0_en: i1, %RW0_clk: i1, %RW0_wmode: i1, %RW0_wdata: i42, %W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i42) -> (R0_data: i42, RW0_rdata: i42) attributes {depth = 12 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 42 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 1 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : i32, width = 42 : ui32, writeClockIDs = [0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}

  // CHECK-LABEL: hw.module @Simple
  module @Simple(in %in1: !firrtl.uint<4>,
                        in %in2: !firrtl.uint<2>,
                        in %in3: !firrtl.sint<8>,
                        in %in4: !firrtl.uint<0>,
                        in %in5: !firrtl.sint<0>,
                        out %out1: !firrtl.sint<1>,
                        out %out2: !firrtl.sint<1>  ) {
    // Issue #364: https://github.com/llvm/circt/issues/364
    // CHECK: = hw.constant -1175 : i12
    // CHECK-DAG: hw.constant -4 : i4
    %c12_ui4 = constant 12 : !firrtl.uint<4>

    // CHECK-DAG: hw.constant 2 : i3
    %c2_si3 = constant 2 : !firrtl.sint<3>


    // CHECK: %out4 = hw.wire [[OUT4_VAL:%.+]] sym @__Simple__out4 : i4
    %out4 = wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<4>
    // CHECK: hw.wire {{%.+}} sym @__Simple{{.*}}
    // CHECK: hw.wire {{%.+}} sym @__Simple{{.*}}
    %500 = wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<4>
    %501 = wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<5>

    // CHECK: %dntnode = hw.wire %in1 sym @__Simple__dntnode
    %dntnode = node %in1 {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<4>

    // CHECK: %clockWire = hw.wire %false : i1
    %c0_clock = specialconstant 0 : !firrtl.clock
    %clockWire = wire : !firrtl.clock
    connect %clockWire, %c0_clock : !firrtl.clock, !firrtl.clock

    // CHECK: %out5 = hw.wire %c0_i4 sym @__Simple__out5 : i4
    %out5 = wire sym @__Simple__out5 : !firrtl.uint<4>
    %tmp1 = invalidvalue : !firrtl.uint<4>
    connect %out5, %tmp1 : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK: [[ZEXT:%.+]] = comb.concat %false, %in1 : i1, i4
    // CHECK: [[ADD:%.+]] = comb.add bin %c12_i5, [[ZEXT]] : i5
    %0 = add %c12_ui4, %in1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    %1 = asUInt %in1 : (!firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: [[ZEXT1:%.+]] = comb.concat %false, [[ADD]] : i1, i5
    // CHECK: [[ZEXT2:%.+]] = comb.concat %c0_i2, %in1 : i2, i4
    // CHECK-NEXT: [[SUB:%.+]] = comb.sub bin [[ZEXT1]], [[ZEXT2]] : i6
    %2 = sub %0, %1 : (!firrtl.uint<5>, !firrtl.uint<4>) -> !firrtl.uint<6>

    %in2s = asSInt %in2 : (!firrtl.uint<2>) -> !firrtl.sint<2>

    // CHECK: [[PADRES_SIGN:%.+]] = comb.extract %in2 from 1 : (i2) -> i1
    // CHECK: [[PADRES:%.+]] = comb.concat [[PADRES_SIGN]], %in2 : i1, i2
    %3 = pad %in2s, 3 : (!firrtl.sint<2>) -> !firrtl.sint<3>

    // CHECK: [[PADRES2:%.+]] = comb.concat %c0_i2, %in2 : i2, i2
    %4 = pad %in2, 4 : (!firrtl.uint<2>) -> !firrtl.uint<4>

    // CHECK: [[IN2EXT:%.+]] = comb.concat %c0_i2, %in2 : i2, i2
    // CHECK: [[XOR:%.+]] = comb.xor bin [[IN2EXT]], [[PADRES2]] : i4
    %5 = xor %in2, %4 : (!firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: comb.and bin [[XOR]]
    %and = and %5, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: comb.or bin [[XOR]]
    %or = or %5, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: [[CONCAT1:%.+]] = comb.concat [[PADRES2]], [[XOR]] : i4, i4
    %6 = cat %4, %5 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<8>

    // CHECK: comb.concat %in1, %in2
    %7 = cat %in1, %in2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<6>

    // CHECK: %out6 = hw.wire [[PADRES2]] sym @__Simple__out6 : i4
    %out6 = wire sym @__Simple__out6 : !firrtl.uint<4>
    connect %out6, %4 : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK: %out7 = hw.wire [[XOR]] sym @__Simple__out7 : i4
    %out7 = wire sym @__Simple__out7 : !firrtl.uint<4>
    connect %out7, %5 : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK: %out8 = hw.wire [[ZEXT:%.+]] sym @__Simple__out8 : i4
    // CHECK-NEXT: [[ZEXT]] = comb.concat %c0_i2, %in2 : i2, i2
    %out8 = wire sym @__Simple__out8 : !firrtl.uint<4>
    connect %out8, %in2 : !firrtl.uint<4>, !firrtl.uint<2>

    // CHECK: %test-name = hw.wire {{%.+}} sym @"__Simple__test-name" : i4
    wire {name = "test-name", annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<4>

    // CHECK: = hw.wire {{%.+}} : i2
    %_t_1 = wire droppable_name : !firrtl.uint<2>

    // CHECK: = hw.wire {{%.+}} : !hw.array<13xi1>
    %_t_2 = wire droppable_name : !firrtl.vector<uint<1>, 13>

    // CHECK: = hw.wire {{%.+}} : !hw.array<13xi2>
    %_t_3 = wire droppable_name : !firrtl.vector<uint<2>, 13>

    // CHECK-NEXT: = comb.extract [[CONCAT1]] from 3 : (i8) -> i5
    %8 = bits %6 7 to 3 : (!firrtl.uint<8>) -> !firrtl.uint<5>

    // CHECK-NEXT: = comb.extract [[CONCAT1]] from 5 : (i8) -> i3
    %9 = head %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<3>

    // CHECK-NEXT: = comb.extract [[CONCAT1]] from 0 : (i8) -> i5
    %10 = tail %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<5>

    // CHECK-NEXT: = comb.extract [[CONCAT1]] from 3 : (i8) -> i5
    %11 = shr %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<5>

    %12 = shr %6, 8 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = comb.extract %in3 from 7 : (i8) -> i1
    %13 = shr %in3, 8 : (!firrtl.sint<8>) -> !firrtl.sint<1>

    // CHECK-NEXT: = comb.concat [[CONCAT1]], %c0_i3 : i8, i3
    %14 = shl %6, 3 : (!firrtl.uint<8>) -> !firrtl.uint<11>

    // CHECK-NEXT: = comb.parity [[CONCAT1]] : i8
    %15 = xorr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = comb.icmp bin eq  {{.*}}, %c-1_i8 : i8
    %16 = andr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: = comb.icmp bin ne {{.*}}, %c0_i8 : i8
    %17 = orr %6 : (!firrtl.uint<8>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[ZEXTC1:%.+]] = comb.concat %c0_i6, [[CONCAT1]] : i6, i8
    // CHECK-NEXT: [[ZEXT2:%.+]] = comb.concat %c0_i8, [[SUB]] : i8, i6
    // CHECK-NEXT: [[VAL18:%.+]] = comb.mul bin [[ZEXTC1]], [[ZEXT2]] : i14
    %18 = mul %6, %2 : (!firrtl.uint<8>, !firrtl.uint<6>) -> !firrtl.uint<14>

    // CHECK: [[IN3SEXT:%.+]] = comb.concat {{.*}}, %in3 : i1, i8
    // CHECK: [[PADRESSEXT:%.+]] = comb.concat {{.*}}, [[PADRES]] : i6, i3
    // CHECK-NEXT: = comb.divs bin [[IN3SEXT]], [[PADRESSEXT]] : i9
    %19 = div %in3, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.sint<9>

    // CHECK: [[IN3EX:%.+]] = comb.concat {{.*}}, [[PADRES]] : i5, i3
    // CHECK-NEXT: [[MOD1:%.+]] = comb.mods bin %in3, [[IN3EX]] : i8
    // CHECK-NEXT: = comb.extract [[MOD1]] from 0 : (i8) -> i3
    %20 = rem %in3, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.sint<3>

    // CHECK: [[IN4EX:%.+]] = comb.concat {{.*}}, [[PADRES]] : i5, i3
    // CHECK-NEXT: [[MOD2:%.+]] = comb.mods bin [[IN4EX]], %in3 : i8
    // CHECK-NEXT: = comb.extract [[MOD2]] from 0 : (i8) -> i3
    %21 = rem %3, %in3 : (!firrtl.sint<3>, !firrtl.sint<8>) -> !firrtl.sint<3>

    // Nodes with names become wires.
    // CHECK-NEXT: %n1 = hw.wire %in2
    // CHECK-NEXT: %n2 = hw.wire %in2 sym @__Simple__n2 : i2
    %n1 = node interesting_name %in2 {name = "n1"} : !firrtl.uint<2>
    %n2 = node interesting_name %in2  {name = "n2", annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<2>

    // Nodes with no names are just dropped.
    %22 = node droppable_name %in2 {name = ""} : !firrtl.uint<2>

    // CHECK-NEXT: %n3 = hw.wire %in2 sym @nodeSym : i2
    %n3 = node sym @nodeSym %in2 : !firrtl.uint<2>

    // CHECK-NEXT: [[CVT:%.+]] = comb.concat %false, %in2 : i1, i2
    %23 = cvt %22 : (!firrtl.uint<2>) -> !firrtl.sint<3>

    // Will be dropped, here because this triggered a crash
    %s23 = cvt %in3 : (!firrtl.sint<8>) -> !firrtl.sint<8>

    // CHECK-NEXT: [[XOR:%.+]] = comb.xor bin [[CVT]], %c-1_i3 : i3
    %24 = not %23 : (!firrtl.sint<3>) -> !firrtl.uint<3>

    %s24 = asSInt %24 : (!firrtl.uint<3>) -> !firrtl.sint<3>

    // CHECK: [[SEXT:%.+]] = comb.concat {{.*}}, [[XOR]] : i1, i3
    // CHECK-NEXT: [[SUB:%.+]] = comb.sub bin %c0_i4, [[SEXT]] : i4
    %25 = neg %s24 : (!firrtl.sint<3>) -> !firrtl.sint<4>

    // CHECK: [[CVT4:%.+]] = comb.concat {{.*}}, [[CVT]] : i1, i3
    // CHECK-NEXT: comb.mux bin {{.*}}, [[CVT4]], [[SUB]] : i4
    %26 = mux(%17, %23, %25) : (!firrtl.uint<1>, !firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.sint<4>

    // CHECK-NEXT: = comb.icmp bin eq {{.*}}, %c-1_i14 : i14
    %28 = andr %18 : (!firrtl.uint<14>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[XOREXT:%.+]] = comb.concat %c0_i11, [[XOR]]
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shru bin [[XOREXT]], [[VAL18]] : i14
    // CHECK-NEXT: [[DSHR:%.+]] = comb.extract [[SHIFT]] from 0 : (i14) -> i3
    %29 = dshr %24, %18 : (!firrtl.uint<3>, !firrtl.uint<14>) -> !firrtl.uint<3>

    // CHECK-NEXT: = comb.concat %c0_i5, {{.*}} : i5, i3
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shrs bin %in3, {{.*}} : i8
    %a29 = dshr %in3, %9 : (!firrtl.sint<8>, !firrtl.uint<3>) -> !firrtl.sint<8>

    // CHECK: = comb.concat {{.*}}, %in3 : i7, i8
    // CHECK-NEXT: = comb.concat %c0_i12, [[DSHR]]
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shl bin {{.*}}, {{.*}} : i15
    %30 = dshl %in3, %29 : (!firrtl.sint<8>, !firrtl.uint<3>) -> !firrtl.sint<15>

    // CHECK-NEXT: = comb.shl bin [[DSHR]], [[DSHR]] : i3
    %dshlw = dshlw %29, %29 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>

    // Issue #367: https://github.com/llvm/circt/issues/367
    // CHECK: = comb.concat {{.*}} : i10, i4
    // CHECK-NEXT: [[SHIFT:%.+]] = comb.shrs bin {{.*}}, {{.*}} : i14
    // CHECK-NEXT: = comb.extract [[SHIFT]] from 0 : (i14) -> i4
    %31 = dshr %25, %18 : (!firrtl.sint<4>, !firrtl.uint<14>) -> !firrtl.sint<4>

    // CHECK-NEXT: comb.icmp bin ule {{.*}}, {{.*}} : i4
    %41 = leq %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp bin ult {{.*}}, {{.*}} : i4
    %42 = lt %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp bin uge {{.*}}, {{.*}} : i4
    %43 = geq %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp bin ugt {{.*}}, {{.*}} : i4
    %44 = gt %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp bin eq {{.*}}, {{.*}} : i4
    %45 = eq %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT: comb.icmp bin ne {{.*}}, {{.*}} : i4
    %46 = neq %in1, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>

    // Noop
    %47 = asClock %44 : (!firrtl.uint<1>) -> !firrtl.clock
    %48 = asAsyncReset %44 : (!firrtl.uint<1>) -> !firrtl.asyncreset

    // CHECK: [[VERB1:%.+]] = sv.verbatim.expr "MAGIC_CONSTANT" : () -> i42
    // CHECK: [[VERB2:%.+]] = sv.verbatim.expr "$bits({{[{][{]0[}][}]}}, {{[{][{]1[}][}]}})"([[VERB1]]) : (i42) -> i32 {symbols = [@Simple]}
    // CHECK: [[VERB3:%.+]] = sv.verbatim.expr.se "$size({{[{][{]0[}][}]}}, {{[{][{]1[}][}]}})"([[VERB1]]) : (i42) -> !hw.inout<i32> {symbols = [@Simple]}
    // CHECK: [[VERB3READ:%.+]] = sv.read_inout [[VERB3]]
    // CHECK: [[VERB1EXT:%.+]] = comb.concat {{%.+}}, [[VERB1]] : i1, i42
    // CHECK: [[VERB2EXT:%.+]] = comb.concat {{%.+}}, [[VERB2]] : i11, i32
    // CHECK: [[ADD:%.+]] = comb.add bin [[VERB1EXT]], [[VERB2EXT]] : i43
    // CHECK: [[VERB3EXT:%.+]] = comb.concat {{%.+}}, [[VERB3READ]] : i12, i32
    // CHECK: [[ADDEXT:%.+]] = comb.concat {{%.+}}, [[ADD]] : i1, i43
    // CHECK: = comb.add bin [[VERB3EXT]], [[ADDEXT]] : i44
    %56 = verbatim.expr "MAGIC_CONSTANT" : () -> !firrtl.uint<42>
    %57 = verbatim.expr "$bits({{0}}, {{1}})"(%56) : (!firrtl.uint<42>) -> !firrtl.uint<32> {symbols = [@Simple]}
    %58 = verbatim.wire "$size({{0}}, {{1}})"(%56) : (!firrtl.uint<42>) -> !firrtl.uint<32> {symbols = [@Simple]}
    %59 = add %56, %57 : (!firrtl.uint<42>, !firrtl.uint<32>) -> !firrtl.uint<43>
    %60 = add %58, %59 : (!firrtl.uint<32>, !firrtl.uint<43>) -> !firrtl.uint<44>

    // Issue #353
    // CHECK: [[PADRES_EXT:%.+]] = comb.concat {{.*}}, [[PADRES]] : i5, i3
    // CHECK: = comb.and bin %in3, [[PADRES_EXT]] : i8
    %49 = and %in3, %3 : (!firrtl.sint<8>, !firrtl.sint<3>) -> !firrtl.uint<8>

    // Issue #355: https://github.com/llvm/circt/issues/355
    // CHECK: [[IN1:%.+]] = comb.concat %c0_i6, %in1 : i6, i4
    // CHECK: [[DIV:%.+]] = comb.divu bin [[IN1]], %c306_i10 : i10
    // CHECK: = comb.extract [[DIV]] from 0 : (i10) -> i4
    %c306_ui10 = constant 306 : !firrtl.uint<10>
    %50 = div %in1, %c306_ui10 : (!firrtl.uint<4>, !firrtl.uint<10>) -> !firrtl.uint<4>

    %c1175_ui11 = constant 1175 : !firrtl.uint<11>
    %51 = neg %c1175_ui11 : (!firrtl.uint<11>) -> !firrtl.sint<12>
    // https://github.com/llvm/circt/issues/821
    // CHECK: [[CONCAT:%.+]] = comb.concat %false, %in1 : i1, i4
    // CHECK:  = comb.sub bin %c0_i5, [[CONCAT]] : i5
    %52 = neg %in1 : (!firrtl.uint<4>) -> !firrtl.sint<5>
    %53 = neg %in4 : (!firrtl.uint<0>) -> !firrtl.sint<1>
    // CHECK: [[SEXT:%.+]] = comb.concat {{.*}}, %in3 : i1, i8
    // CHECK: = comb.sub bin %c0_i9, [[SEXT]] : i9
    %54 = neg %in3 : (!firrtl.sint<8>) -> !firrtl.sint<9>
    connect %out1, %53 : !firrtl.sint<1>, !firrtl.sint<1>
    %55 = neg %in5 : (!firrtl.sint<0>) -> !firrtl.sint<1>

    %61 = multibit_mux %17, %55, %55, %55 : !firrtl.uint<1>, !firrtl.sint<1>
    // CHECK:      %[[ZEXT_INDEX:.+]] = comb.concat %false, {{.*}} : i1, i1
    // CHECK-NEXT: %[[ARRAY:.+]] = hw.array_create %false, %false, %false
    // CHECK-NEXT: %[[GET0:.+]] = hw.array_get %[[ARRAY]][%c0_i2]
    // CHECK-NEXT: %[[FILLER:.+]] = hw.array_create %[[GET0]] : i1
    // CHECK-NEXT: %[[EXT:.+]] = hw.array_concat %[[FILLER]], %[[ARRAY]]
    // CHECK-NEXT: %[[ARRAY_GET:.+]] = hw.array_get %[[EXT]][%[[ZEXT_INDEX]]]
    // CHECK: hw.output %false, %[[ARRAY_GET]] : i1, i1
    connect %out2, %61 : !firrtl.sint<1>, !firrtl.sint<1>
  }

//   module Print :
//    input clock: Clock
//    input reset: UInt<1>
//    input a: UInt<4>
//    input b: UInt<4>
//    printf(clock, reset, "No operands!\n")
//    printf(clock, reset, "Hi %x %x\n", add(a, a), b)

  // CHECK-LABEL: hw.module private @Print
  module private @Print(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                       in %a: !firrtl.uint<4>, in %b: !firrtl.uint<4>) {
    // CHECK: [[ADD:%.+]] = comb.add

    // CHECK:      sv.ifdef "SYNTHESIS" {
    // CHECK-NEXT: } else  {
    // CHECK-NEXT:   sv.always posedge %clock {
    // CHECK-NEXT:     %PRINTF_COND_ = sv.macro.ref @PRINTF_COND_() : () -> i1
    // CHECK-NEXT:     [[AND:%.+]] = comb.and bin %PRINTF_COND_, %reset
    // CHECK-NEXT:     sv.if [[AND]] {
    // CHECK-NEXT:       [[FD:%.+]] = hw.constant -2147483646 : i32
    // CHECK-NEXT:       sv.fwrite [[FD]], "No operands!\0A"
    // CHECK-NEXT:     }
    // CHECK-NEXT:     %PRINTF_COND__0 = sv.macro.ref @PRINTF_COND_() : () -> i1
    // CHECK-NEXT:     [[AND:%.+]] = comb.and bin %PRINTF_COND__0, %reset : i1
    // CHECK-NEXT:     sv.if [[AND]] {
    // CHECK-NEXT:       [[FD:%.+]] = hw.constant -2147483646 : i32
    // CHECK-NEXT:       sv.fwrite [[FD]], "Hi %x %x\0A"(%2, %b) : i5, i4
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
   printf %clock, %reset, "No operands!\0A" : !firrtl.clock, !firrtl.uint<1>

    %0 = add %a, %a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    printf %clock, %reset, "Hi %x %x\0A"(%0, %b) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<4>

    skip

    // CHECK: hw.output
   }



// module Stop3 :
//    input clock1: Clock
//    input clock2: Clock
//    input reset: UInt<1>
//    stop(clock1, reset, 42)
//    stop(clock2, reset, 0)

  // CHECK-LABEL: hw.module private @Stop
  module private @Stop(in %clock1: !firrtl.clock, in %clock2: !firrtl.clock, in %reset: !firrtl.uint<1>) {

    // CHECK-NEXT: sv.ifdef "SYNTHESIS" {
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.always posedge %clock1 {
    // CHECK-NEXT:     %STOP_COND_ = sv.macro.ref @STOP_COND_
    // CHECK-NEXT:     %0 = comb.and bin %STOP_COND_, %reset : i1
    // CHECK-NEXT:     sv.if %0 {
    // CHECK-NEXT:       sv.fatal
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    stop %clock1, %reset, 42 : !firrtl.clock, !firrtl.uint<1>

    // CHECK-NEXT:   sv.always posedge %clock2 {
    // CHECK-NEXT:     %STOP_COND_ = sv.macro.ref @STOP_COND_
    // CHECK-NEXT:     %0 = comb.and bin %STOP_COND_, %reset : i1
    // CHECK-NEXT:     sv.if %0 {
    // CHECK-NEXT:       sv.finish
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    stop %clock2, %reset, 0 : !firrtl.clock, !firrtl.uint<1>
  }

  // circuit Verification:
  //   module Verification:
  //     input clock: Clock
  //     input aCond: UInt<8>
  //     input aEn: UInt<8>
  //     input bCond: UInt<1>
  //     input bEn: UInt<1>
  //     input cCond: UInt<1>
  //     input cEn: UInt<1>
  //     assert(clock, bCond, bEn, "assert0")
  //     assert(clock, bCond, bEn, "assert0") : assert_0
  //     assume(clock, aCond, aEn, "assume0")
  //     assume(clock, aCond, aEn, "assume0") : assume_0
  //     cover(clock,  cCond, cEn, "cover0)"
  //     cover(clock,  cCond, cEn, "cover0)" : cover_0

  // CHECK-LABEL: hw.module private @Verification
  module private @Verification(in %clock: !firrtl.clock, in %aCond: !firrtl.uint<1>,
    in %aEn: !firrtl.uint<1>, in %bCond: !firrtl.uint<1>, in %bEn: !firrtl.uint<1>,
    in %cCond: !firrtl.uint<1>, in %cEn: !firrtl.uint<1>, in %value: !firrtl.uint<42>) {

    assert %clock, %aCond, %aEn, "assert0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {isConcurrent = true}
    assert %clock, %aCond, %aEn, "assert0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {isConcurrent = true, name = "assert_0"}
    assert %clock, %aCond, %aEn, "assert0"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42> {isConcurrent = true}
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %aEn, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or bin [[TMP1]], %aCond
    // CHECK-NEXT: sv.assert.concurrent posedge %clock, [[TMP2]] message "assert0"
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP3:%.+]] = comb.xor %aEn, [[TRUE]]
    // CHECK-NEXT: [[TMP4:%.+]] = comb.or bin [[TMP3]], %aCond
    // CHECK-NEXT: sv.assert.concurrent posedge %clock, [[TMP4]] label "assert__assert_0" message "assert0"
    // CHECK-NEXT: [[SAMPLED:%.+]] =  sv.system.sampled %value : i42
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP5:%.+]] = comb.xor %aEn, [[TRUE]]
    // CHECK-NEXT: [[TMP6:%.+]] = comb.or bin [[TMP5]], %aCond
    // CHECK-NEXT: sv.assert.concurrent posedge %clock, [[TMP6]] message "assert0"([[SAMPLED]]) : i42
    // CHECK-NEXT: sv.ifdef "USE_PROPERTY_AS_CONSTRAINT" {
    // CHECK-NEXT:   sv.assume.concurrent posedge %clock, [[TMP2]]
    // CHECK-NEXT:   sv.assume.concurrent posedge %clock, [[TMP4]] label "assume__assert_0"
    // CHECK-NEXT:   sv.assume.concurrent posedge %clock, [[TMP6]]
    // CHECK-NEXT: }
    assume %clock, %bCond, %bEn, "assume0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {isConcurrent = true}
    assume %clock, %bCond, %bEn, "assume0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {isConcurrent = true, name = "assume_0"}
    assume %clock, %bCond, %bEn, "assume0"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42> {isConcurrent = true}
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %bEn, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or bin [[TMP1]], %bCond
    // CHECK-NEXT: sv.assume.concurrent posedge %clock, [[TMP2]] message "assume0"
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %bEn, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or bin [[TMP1]], %bCond
    // CHECK-NEXT: sv.assume.concurrent posedge %clock, [[TMP2]] label "assume__assume_0" message "assume0"
    // CHECK-NEXT: [[SAMPLED:%.+]] = sv.system.sampled %value
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %bEn, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or bin [[TMP1]], %bCond
    // CHECK-NEXT: sv.assume.concurrent posedge %clock, [[TMP2]] message "assume0"([[SAMPLED]]) : i42
    cover %clock, %cCond, %cEn, "cover0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {isConcurrent = true}
    cover %clock, %cCond, %cEn, "cover0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {isConcurrent = true, name = "cover_0"}
    cover %clock, %cCond, %cEn, "cover0"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42> {isConcurrent = true}
    // CHECK-NEXT: [[TMP:%.+]] = comb.and bin %cEn, %cCond
    // CHECK-NEXT: sv.cover.concurrent posedge %clock, [[TMP]]
    // CHECK-NEXT: [[TMP:%.+]] = comb.and bin %cEn, %cCond
    // CHECK-NEXT: sv.cover.concurrent posedge %clock, [[TMP]] label "cover__cover_0"
    // CHECK-NEXT: [[TMP:%.+]] = comb.and bin %cEn, %cCond
    // CHECK-NEXT: sv.cover.concurrent posedge %clock, [[TMP]]
    cover %clock, %cCond, %cEn, "cover1" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 1 : i32, isConcurrent = true, name = "cover_1"}
    cover %clock, %cCond, %cEn, "cover2" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 2 : i32, isConcurrent = true, name = "cover_2"}
    // CHECK: sv.cover.concurrent negedge %clock, {{%.+}} label "cover__cover_1"
    // CHECK: sv.cover.concurrent edge %clock, {{%.+}} label "cover__cover_2"

    // CHECK-NEXT: sv.always posedge %clock {
    // CHECK-NEXT:   sv.if %aEn {
    // CHECK-NEXT:     sv.assert %aCond, immediate message "assert0"
    // CHECK-NEXT:     sv.assert %aCond, immediate label "assert__assert_0" message "assert0"
    // CHECK-NEXT:     sv.assert %aCond, immediate message "assert0"(%value) : i42
    // CHECK-NEXT:   }
    // CHECK-NEXT:   sv.if %bEn {
    // CHECK-NEXT:     sv.assume %bCond, immediate message "assume0"
    // CHECK-NEXT:     sv.assume %bCond, immediate label "assume__assume_0" message "assume0"
    // CHECK-NEXT:     sv.assume %bCond, immediate message "assume0"(%value) : i42
    // CHECK-NEXT:   }
    // CHECK-NEXT:   sv.if %cEn {
    // CHECK-NEXT:     sv.cover %cCond, immediate
    // CHECK-NOT:        label
    // CHECK-NEXT:     sv.cover %cCond, immediate label "cover__cover_0"
    // CHECK-NEXT:     sv.cover %cCond, immediate
    // CHECK-NOT:        label
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    assert %clock, %aCond, %aEn, "assert0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    assert %clock, %aCond, %aEn, "assert0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {name = "assert_0"}
    assert %clock, %aCond, %aEn, "assert0"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    assume %clock, %bCond, %bEn, "assume0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    assume %clock, %bCond, %bEn, "assume0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {name = "assume_0"}
    assume %clock, %bCond, %bEn, "assume0"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    cover %clock, %cCond, %cEn, "cover0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    cover %clock, %cCond, %cEn, "cover0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {name = "cover_0"}
    cover %clock, %cCond, %cEn, "cover0"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    // CHECK-NEXT: hw.output
  }

  // CHECK-LABEL: hw.module private @VerificationGuards
  module private @VerificationGuards(
    in %clock: !firrtl.clock,
    in %cond: !firrtl.uint<1>,
    in %enable: !firrtl.uint<1>
  ) {
    assert %clock, %cond, %enable, "assert0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {isConcurrent = true, guards = ["HELLO", "WORLD"]} 
    assume %clock, %cond, %enable, "assume0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {isConcurrent = true, guards = ["HELLO", "WORLD"]}
    cover %clock, %cond, %enable, "cover0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {isConcurrent = true, guards = ["HELLO", "WORLD"]}

    // CHECK-NEXT: sv.ifdef "HELLO" {
    // CHECK-NEXT:   sv.ifdef "WORLD" {
    // CHECK-NEXT:     [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT:     [[TMP1:%.+]] = comb.xor %enable, [[TRUE]]
    // CHECK-NEXT:     [[TMP2:%.+]] = comb.or bin [[TMP1]], %cond
    // CHECK-NEXT:     sv.assert.concurrent posedge %clock, [[TMP2]] message "assert0"
    // CHECK-NEXT:     sv.ifdef "USE_PROPERTY_AS_CONSTRAINT" {
    // CHECK-NEXT:       sv.assume.concurrent posedge %clock, [[TMP2]]
    // CHECK-NEXT:     }
    // CHECK-NEXT:     [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT:     [[TMP1:%.+]] = comb.xor %enable, [[TRUE]]
    // CHECK-NEXT:     [[TMP2:%.+]] = comb.or bin [[TMP1]], %cond
    // CHECK-NEXT:     sv.assume.concurrent posedge %clock, [[TMP2]] message "assume0"
    // CHECK-NEXT:     [[TMP:%.+]] = comb.and bin %enable, %cond
    // CHECK-NEXT:     sv.cover.concurrent posedge %clock, [[TMP]]
    // CHECK-NOT:      label
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
  }

  // CHECK-LABEL: hw.module private @VerificationAssertFormat
  module private @VerificationAssertFormat(
    in %clock: !firrtl.clock,
    in %cond: !firrtl.uint<1>,
    in %enable: !firrtl.uint<1>,
    in %value: !firrtl.uint<42>,
    in %i0: !firrtl.uint<0>
  ) {
    assert %clock, %cond, %enable, "assert0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {isConcurrent = true, format = "sva"}
    // CHECK-NEXT: [[FALSE:%.+]] = hw.constant false
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %enable, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.or bin [[TMP1]], %cond
    // CHECK-NEXT: sv.assert.concurrent posedge %clock, [[TMP2]] message "assert0"
    // CHECK-NEXT: sv.ifdef "USE_PROPERTY_AS_CONSTRAINT" {
    // CHECK-NEXT:   sv.assume.concurrent posedge %clock, [[TMP2]]
    // CHECK-NEXT: }
    assert %clock, %cond, %enable, "assert1 %d, %d"(%value, %i0) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>, !firrtl.uint<0> {isConcurrent = true, format = "ifElseFatal"}
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1:%.+]] = comb.xor %cond, [[TRUE]]
    // CHECK-NEXT: [[TMP2:%.+]] = comb.and bin %enable, [[TMP1]]
    // CHECK-NEXT: sv.ifdef "SYNTHESIS" {
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.always posedge %clock {
    // CHECK-NEXT:     sv.if [[TMP2]] {
    // CHECK-NEXT:       [[ASSERT_VERBOSE_COND:%.+]] = sv.macro.ref @ASSERT_VERBOSE_COND_
    // CHECK-NEXT:       sv.if [[ASSERT_VERBOSE_COND]] {
    // CHECK-NEXT:         sv.error "assert1 %d, %d"(%value, %false) : i42, i1
    // CHECK-NEXT:       }
    // CHECK-NEXT:       [[STOP_COND:%.+]] = sv.macro.ref @STOP_COND_
    // CHECK-NEXT:       sv.if [[STOP_COND]] {
    // CHECK-NEXT:         sv.fatal
    // CHECK-NEXT:       }
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
  }

  module private @bar(in %io_cpu_flush: !firrtl.uint<1>) {
    // CHECK: hw.probe @baz, %io_cpu_flush, %io_cpu_flush : i1, i1
    probe @baz, %io_cpu_flush, %io_cpu_flush  : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module private @foo
  module private @foo() {
    // CHECK:      %io_cpu_flush.wire = hw.wire %z_i1 sym @__foo__io_cpu_flush.wire : i1
    %io_cpu_flush.wire = wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    // CHECK-NEXT: hw.instance "fetch" @bar(io_cpu_flush: %io_cpu_flush.wire: i1)
    %i = instance fetch @bar(in io_cpu_flush: !firrtl.uint<1>)
    connect %i, %io_cpu_flush.wire : !firrtl.uint<1>, !firrtl.uint<1>

    %hits_1_7 = node %io_cpu_flush.wire {name = "hits_1_7", annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    // CHECK-NEXT:  %hits_1_7 = hw.wire %io_cpu_flush.wire sym @__foo__hits_1_7 : i1
    %1455 = builtin.unrealized_conversion_cast %hits_1_7 : !firrtl.uint<1> to !firrtl.uint<1>
  }

  // CHECK: sv.bind <@bindTest::@[[bazSymbol:.+]]>
  // CHECK-NOT: output_file
  // CHECK-NEXT: sv.bind <@bindTest::@[[quxSymbol:.+]]> {
  // CHECK-SAME: output_file = #hw.output_file<"bindings.sv", excludeFromFileList>
  // CHECK-NEXT: hw.module private @bindTest()
  module private @bindTest() {
    // CHECK: hw.instance "baz" sym @[[bazSymbol]] @bar
    %baz = instance baz {lowerToBind} @bar(in io_cpu_flush: !firrtl.uint<1>)
    // CHECK: hw.instance "qux" sym @[[quxSymbol]] @bar
    %qux = instance qux {lowerToBind, output_file = #hw.output_file<"bindings.sv", excludeFromFileList>} @bar(in io_cpu_flush: !firrtl.uint<1>)
  }


  // CHECK-LABEL: hw.module private @output_fileTest
  // CHECK-SAME: output_file = #hw.output_file<"output_fileTest.sv", excludeFromFileList>
  module private @output_fileTest() attributes {
      output_file = #hw.output_file<"output_fileTest.sv", excludeFromFileList >} {
  }

  // https://github.com/llvm/circt/issues/314
  // CHECK-LABEL: hw.module private @issue314
  module private @issue314(in %inp_2: !firrtl.uint<27>, in %inpi: !firrtl.uint<65>) {
    // CHECK: %c0_i38 = hw.constant 0 : i38
    // CHECK: %tmp48 = hw.wire %2 : i27
    %tmp48 = wire : !firrtl.uint<27>

    // CHECK-NEXT: %0 = comb.concat %c0_i38, %inp_2 : i38, i27
    // CHECK-NEXT: %1 = comb.divu bin %0, %inpi : i65
    %0 = div %inp_2, %inpi : (!firrtl.uint<27>, !firrtl.uint<65>) -> !firrtl.uint<27>
    // CHECK-NEXT: %2 = comb.extract %1 from 0 : (i65) -> i27
    connect %tmp48, %0 : !firrtl.uint<27>, !firrtl.uint<27>
  }

  // https://github.com/llvm/circt/issues/318
  // CHECK-LABEL: hw.module private @test_rem
  // CHECK-NEXT:     %0 = comb.modu
  // CHECK-NEXT:     hw.output %0
  module private @test_rem(in %tmp85: !firrtl.uint<1>, in %tmp79: !firrtl.uint<1>,
       out %out: !firrtl.uint<1>) {
    %2 = rem %tmp79, %tmp85 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %out, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module private @Analog(%a1: !hw.inout<i1>, %b1: !hw.inout<i1>,
  // CHECK:                          %c1: !hw.inout<i1>) -> (outClock: i1) {
  // CHECK-NEXT:   %0 = sv.read_inout %c1 : !hw.inout<i1>
  // CHECK-NEXT:   %1 = sv.read_inout %b1 : !hw.inout<i1>
  // CHECK-NEXT:   %2 = sv.read_inout %a1 : !hw.inout<i1>
  // CHECK-NEXT:   sv.ifdef "SYNTHESIS"  {
  // CHECK-NEXT:     sv.assign %a1, %1 : i1
  // CHECK-NEXT:     sv.assign %a1, %0 : i1
  // CHECK-NEXT:     sv.assign %b1, %2 : i1
  // CHECK-NEXT:     sv.assign %b1, %0 : i1
  // CHECK-NEXT:     sv.assign %c1, %2 : i1
  // CHECK-NEXT:     sv.assign %c1, %1 : i1
  // CHECK-NEXT:    } else {
  // CHECK-NEXT:     sv.ifdef "verilator" {
  // CHECK-NEXT:       sv.verbatim "`error \22Verilator does not support alias and thus cannot arbitrarily connect bidirectional wires and ports\22"
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       sv.alias %a1, %b1, %c1 : !hw.inout<i1>
  // CHECK-NEXT:     }
  // CHECK-NEXT:    }
  // CHECK-NEXT:    hw.output %2 : i1
  module private @Analog(in %a1: !firrtl.analog<1>, in %b1: !firrtl.analog<1>,
                        in %c1: !firrtl.analog<1>, out %outClock: !firrtl.clock) {
    attach %a1, %b1, %c1 : !firrtl.analog<1>, !firrtl.analog<1>, !firrtl.analog<1>

    %1 = asClock %a1 : (!firrtl.analog<1>) -> !firrtl.clock
    connect %outClock, %1 : !firrtl.clock, !firrtl.clock
  }

  //  module MemSimple :
  //     input clock1  : Clock
  //     input clock2  : Clock
  //     input inpred  : UInt<1>
  //     input indata  : SInt<42>
  //     output result : SInt<42>
  //     output result2 : SInt<42>
  //
  //     mem _M : @[Decoupled.scala 209:27]
  //           data-type => SInt<42>
  //           depth => 12
  //           read-latency => 0
  //           write-latency => 1
  //           reader => read
  //           writer => write
  //           readwriter => rw
  //           read-under-write => undefined
  //
  //     result <= _M.read.data
  //     result2 <= _M.rw.rdata
  //
  //     _M.read.addr <= UInt<1>("h0")
  //     _M.read.en <= UInt<1>("h1")
  //     _M.read.clk <= clock1
  //     _M.rw.addr <= UInt<1>("h0")
  //     _M.rw.en <= UInt<1>("h1")
  //     _M.rw.clk <= clock1
  //     _M.rw.wmask <= UInt<1>("h1")
  //     _M.rw.wmode <= UInt<1>("h1")
  //     _M.write.addr <= validif(inpred, UInt<3>("h0"))
  //     _M.write.en <= mux(inpred, UInt<1>("h1"), UInt<1>("h0"))
  //     _M.write.clk <= clock2
  //     _M.write.data <= validif(inpred, indata)
  //     _M.write.mask <= validif(inpred, UInt<1>("h1"))

  // CHECK-LABEL: hw.module private @MemSimple(
  module private @MemSimple(in %clock1: !firrtl.clock, in %clock2: !firrtl.clock,
                           in %inpred: !firrtl.uint<1>, in %indata: !firrtl.sint<42>,
                           out %result: !firrtl.sint<42>,
                           out %result2: !firrtl.sint<42>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %c0_ui3 = constant 0 : !firrtl.uint<3>
    %_M_read, %_M_rw, %_M_write = mem Undefined {depth = 12 : i64, name = "_M", portNames = ["read", "rw", "write"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>
  // CHECK: %[[v1:.+]] = comb.and bin %true, %inpred : i1
  // CHECK: %[[v2:.+]] = comb.and bin %inpred, %true : i1
  // CHECK: %_M_ext.R0_data, %_M_ext.RW0_rdata = hw.instance "_M_ext" @_M_combMem_0(R0_addr: %c0_i4: i4, R0_en: %true: i1, R0_clk: %clock1: i1, RW0_addr: %c0_i4_0: i4, RW0_en: %true: i1, RW0_clk: %clock1: i1, RW0_wmode: %[[v1]]: i1, RW0_wdata: %indata: i42, W0_addr: %c0_i4_1: i4, W0_en: %[[v2]]: i1, W0_clk: %clock2: i1, W0_data: %indata: i42) -> (R0_data: i42, RW0_rdata: i42)
  // CHECK: hw.output %_M_ext.R0_data, %_M_ext.RW0_rdata : i42, i42

      %0 = subfield %_M_read[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>
      connect %result, %0 : !firrtl.sint<42>, !firrtl.sint<42>
      %1 = subfield %_M_rw[rdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>
      connect %result2, %1 : !firrtl.sint<42>, !firrtl.sint<42>
      %2 = subfield %_M_read[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>
      connect %2, %c0_ui1 : !firrtl.uint<4>, !firrtl.uint<1>
      %3 = subfield %_M_read[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>
      connect %3, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %4 = subfield %_M_read[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>
      connect %4, %clock1 : !firrtl.clock, !firrtl.clock

      %5 = subfield %_M_rw[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>
      connect %5, %c0_ui1 : !firrtl.uint<4>, !firrtl.uint<1>
      %6 = subfield %_M_rw[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>
      connect %6, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %7 = subfield %_M_rw[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>
      connect %7, %clock1 : !firrtl.clock, !firrtl.clock
      %8 = subfield %_M_rw[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>
      connect %8, %inpred : !firrtl.uint<1>, !firrtl.uint<1>
      %9 = subfield %_M_rw[wmode] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>
      connect %9, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %10 = subfield %_M_rw[wdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>
      connect %10, %indata : !firrtl.sint<42>, !firrtl.sint<42>

      %11 = subfield %_M_write[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>
      connect %11, %c0_ui3 : !firrtl.uint<4>, !firrtl.uint<3>
      %12 = subfield %_M_write[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>
      connect %12, %inpred : !firrtl.uint<1>, !firrtl.uint<1>
      %13 = subfield %_M_write[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>
      connect %13, %clock2 : !firrtl.clock, !firrtl.clock
      %14 = subfield %_M_write[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>
      connect %14, %indata : !firrtl.sint<42>, !firrtl.sint<42>
      %15 = subfield %_M_write[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>
      connect %15, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module private @MemSimple_mask(
  module private @MemSimple_mask(in %clock1: !firrtl.clock, in %clock2: !firrtl.clock,
                           in %inpred: !firrtl.uint<1>, in %indata: !firrtl.sint<40>,
                           out %result: !firrtl.sint<40>,
                           out %result2: !firrtl.sint<40>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %c0_ui10 = constant 0 : !firrtl.uint<10>
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %c0_ui3 = constant 0 : !firrtl.uint<3>
    %c0_ui4 = constant 0 : !firrtl.uint<4>
    %c1_ui5 = constant 1 : !firrtl.uint<5>
    %_M_read, %_M_rw, %_M_write = mem Undefined {depth = 1022 : i64, name = "_M_mask", portNames = ["read", "rw", "write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: sint<40>>, !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>, !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>, mask: uint<4>>
    // CHECK: %_M_mask_ext.R0_data, %_M_mask_ext.RW0_rdata = hw.instance "_M_mask_ext" @_M_mask_combMem(R0_addr: %c0_i10: i10, R0_en: %true: i1, R0_clk: %clock1: i1, RW0_addr: %c0_i10: i10, RW0_en: %true: i1, RW0_clk: %clock1: i1, RW0_wmode: %true: i1, RW0_wdata: %indata: i40, RW0_wmask: %c0_i4: i4, W0_addr: %c0_i10: i10, W0_en: %inpred: i1, W0_clk: %clock2: i1, W0_data: %indata: i40, W0_mask: %c0_i4: i4) -> (R0_data: i40, RW0_rdata: i40)
    // CHECK: hw.output %_M_mask_ext.R0_data, %_M_mask_ext.RW0_rdata : i40, i40

      %0 = subfield %_M_read[data] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: sint<40>>
      connect %result, %0 : !firrtl.sint<40>, !firrtl.sint<40>
      %1 = subfield %_M_rw[rdata] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>
      connect %result2, %1 : !firrtl.sint<40>, !firrtl.sint<40>
      %2 = subfield %_M_read[addr] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: sint<40>>
      connect %2, %c0_ui10 : !firrtl.uint<10>, !firrtl.uint<10>
      %3 = subfield %_M_read[en] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: sint<40>>
      connect %3, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %4 = subfield %_M_read[clk] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: sint<40>>
      connect %4, %clock1 : !firrtl.clock, !firrtl.clock

      %5 = subfield %_M_rw[addr] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>,  wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>
      connect %5, %c0_ui10 : !firrtl.uint<10>, !firrtl.uint<10>
      %6 = subfield %_M_rw[en] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>
      connect %6, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %7 = subfield %_M_rw[clk] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>
      connect %7, %clock1 : !firrtl.clock, !firrtl.clock
      %8 = subfield %_M_rw[wmask] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>
      connect %8, %c0_ui4 : !firrtl.uint<4>, !firrtl.uint<4>
      %9 = subfield %_M_rw[wmode] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>
      connect %9, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %10 = subfield %_M_rw[wdata] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, rdata flip: sint<40>, wmode: uint<1>, wdata: sint<40>, wmask: uint<4>>
      connect %10, %indata : !firrtl.sint<40>, !firrtl.sint<40>

      %11 = subfield %_M_write[addr] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>, mask: uint<4>>
      connect %11, %c0_ui10 : !firrtl.uint<10>, !firrtl.uint<10>
      %12 = subfield %_M_write[en] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>, mask: uint<4>>
      connect %12, %inpred : !firrtl.uint<1>, !firrtl.uint<1>
      %13 = subfield %_M_write[clk] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>, mask: uint<4>>
      connect %13, %clock2 : !firrtl.clock, !firrtl.clock
      %14 = subfield %_M_write[data] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>, mask: uint<4>>
      connect %14, %indata : !firrtl.sint<40>, !firrtl.sint<40>
      %15 = subfield %_M_write[mask] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: sint<40>, mask: uint<4>>
      connect %15, %c0_ui4 : !firrtl.uint<4>, !firrtl.uint<4>
  }
  // CHECK-LABEL: hw.module private @IncompleteRead(
  // The read port has no use of the data field.
  module private @IncompleteRead(in %clock1: !firrtl.clock) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %c1_ui1 = constant 1 : !firrtl.uint<1>

    // CHECK:  %_M_ext.R0_data = hw.instance "_M_ext" @_M_combMem(R0_addr: %c0_i4: i4, R0_en: %true: i1, R0_clk: %clock1: i1) -> (R0_data: i42)
    %_M_read = mem Undefined {depth = 12 : i64, name = "_M", portNames = ["read"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>
    // Read port.
    %6 = subfield %_M_read[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>
    connect %6, %c0_ui1 : !firrtl.uint<4>, !firrtl.uint<1>
    %7 = subfield %_M_read[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>
    connect %7, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %8 = subfield %_M_read[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>
    connect %8, %clock1 : !firrtl.clock, !firrtl.clock
  }

  // CHECK-LABEL: hw.module private @top_modx() -> (tmp27: i23) {
  // CHECK-NEXT:    %c0_i23 = hw.constant 0 : i23
  // CHECK-NEXT:    %c42_i23 = hw.constant 42 : i23
  // CHECK-NEXT:    hw.output %c0_i23 : i23
  // CHECK-NEXT:  }
  module private @top_modx(out %tmp27: !firrtl.uint<23>) {
    %0 = wire : !firrtl.uint<0>
    %c42_ui23 = constant 42 : !firrtl.uint<23>
    %1 = tail %c42_ui23, 23 : (!firrtl.uint<23>) -> !firrtl.uint<0>
    connect %0, %1 : !firrtl.uint<0>, !firrtl.uint<0>
    %2 = head %c42_ui23, 0 : (!firrtl.uint<23>) -> !firrtl.uint<0>
    %3 = pad %2, 23 : (!firrtl.uint<0>) -> !firrtl.uint<23>
    connect %tmp27, %3 : !firrtl.uint<23>, !firrtl.uint<23>
  }

  // CHECK-LABEL: hw.module private @SimpleStruct(%source: !hw.struct<valid: i1, ready: i1, data: i64>) -> (fldout: i64) {
  // CHECK-NEXT:    %data = hw.struct_extract %source["data"] : !hw.struct<valid: i1, ready: i1, data: i64>
  // CHECK-NEXT:    hw.output %data : i64
  // CHECK-NEXT:  }
  module private @SimpleStruct(in %source: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>,
                              out %fldout: !firrtl.uint<64>) {
    %2 = subfield %source[data] : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
    connect %fldout, %2 : !firrtl.uint<64>, !firrtl.uint<64>
  }

  // CHECK-LABEL: hw.module private @SimpleEnum(%source: !hw.enum<valid, ready, data>) -> (sink: !hw.enum<valid, ready, data>) {
  // CHECK-NEXT:    %valid = hw.enum.constant valid : !hw.enum<valid, ready, data
  // CHECK-NEXT:    %0 = hw.enum.cmp %source, %valid : !hw.enum<valid, ready, data>, !hw.enum<valid, ready, data>
  // CHECK-NEXT:    hw.output %source : !hw.enum<valid, ready, data>
  // CHECK-NEXT:  }
  module private @SimpleEnum(in %source: !firrtl.enum<valid: uint<0>, ready: uint<0>, data: uint<0>>,
                              out %sink: !firrtl.enum<valid: uint<0>, ready: uint<0>, data: uint<0>>) {
    %0 = istag %source valid : !firrtl.enum<valid: uint<0>, ready: uint<0>, data: uint<0>>
    %1 = subtag %source[valid] : !firrtl.enum<valid: uint<0>, ready: uint<0>, data: uint<0>>
    strictconnect %sink, %source : !firrtl.enum<valid: uint<0>, ready: uint<0>, data: uint<0>>
  }

  // CHECK-LABEL: hw.module private @SimpleEnumCreate() -> (sink: !hw.enum<a, b, c>) { 
  // CHECK-NEXT:   %a = hw.enum.constant a : !hw.enum<a, b, c> 
  // CHECK-NEXT:   hw.output %a : !hw.enum<a, b, c> 
  // CHECK-NEXT: }
  module private @SimpleEnumCreate(in %input: !firrtl.uint<0>,
                                         out %sink: !firrtl.enum<a: uint<0>, b: uint<0>, c: uint<0>>) {
    %0 = enumcreate a(%input) : (!firrtl.uint<0>) -> !firrtl.enum<a: uint<0>, b: uint<0>, c: uint<0>>
    strictconnect %sink, %0 : !firrtl.enum<a: uint<0>, b: uint<0>, c: uint<0>>
  }

  // CHECK-LABEL:  hw.module private @DataEnum(%source: !hw.struct<tag: !hw.enum<a, b, c>, body: !hw.union<a: i2, b: i1, c: i32>>) -> (sink: !hw.struct<tag: !hw.enum<a, b, c>, body: !hw.union<a: i2, b: i1, c: i32>>) {
  // CHECK-NEXT:    %tag = hw.struct_extract %source["tag"] : !hw.struct<tag: !hw.enum<a, b, c>, body: !hw.union<a: i2, b: i1, c: i32>>
  // CHECK-NEXT:    %a = hw.enum.constant a : !hw.enum<a, b, c> 
  // CHECK-NEXT:    %0 = hw.enum.cmp %tag, %a : !hw.enum<a, b, c>, !hw.enum<a, b, c>
  // CHECK-NEXT:    %body = hw.struct_extract %source["body"] : !hw.struct<tag: !hw.enum<a, b, c>, body: !hw.union<a: i2, b: i1, c: i32>>
  // CHECK-NEXT:    %1 = hw.union_extract %body["a"] : !hw.union<a: i2, b: i1, c: i32>
  // CHECK-NEXT:    hw.output %source : !hw.struct<tag: !hw.enum<a, b, c>, body: !hw.union<a: i2, b: i1, c: i32>>
  // CHECK-NEXT:  }
  module private @DataEnum(in %source: !firrtl.enum<a: uint<2>, b: uint<1>, c: uint<32>>,
                              out %sink: !firrtl.enum<a: uint<2>, b: uint<1>, c: uint<32>>) {
    %0 = istag %source a : !firrtl.enum<a: uint<2>, b: uint<1>, c: uint<32>>
    %1 = subtag %source[a] : !firrtl.enum<a: uint<2>, b: uint<1>, c: uint<32>>
    strictconnect %sink, %source : !firrtl.enum<a: uint<2>, b: uint<1>, c: uint<32>>
  }

  // CHECK-LABEL: hw.module private @DataEnumCreate(%input: i2) -> (sink: !hw.struct<tag: !hw.enum<a, b, c>, body: !hw.union<a: i2, b: i1, c: i32>>) { 
  // CHECK-NEXT:   %a = hw.enum.constant a : !hw.enum<a, b, c> 
  // CHECK-NEXT:   %0 = hw.union_create "a", %input : !hw.union<a: i2, b: i1, c: i32> 
  // CHECK-NEXT:   %1 = hw.struct_create (%a, %0) : !hw.struct<tag: !hw.enum<a, b, c>, body: !hw.union<a: i2, b: i1, c: i32>> 
  // CHECK-NEXT:   hw.output %1 : !hw.struct<tag: !hw.enum<a, b, c>, body: !hw.union<a: i2, b: i1, c: i32>> 
  // CHECK-NEXT: } 
  module private @DataEnumCreate(in %input: !firrtl.uint<2>,
                                       out %sink: !firrtl.enum<a: uint<2>, b: uint<1>, c: uint<32>>) {
    %0 = enumcreate a (%input) : (!firrtl.uint<2>) -> !firrtl.enum<a: uint<2>, b: uint<1>, c: uint<32>>
    strictconnect %sink, %0 : !firrtl.enum<a: uint<2>, b: uint<1>, c: uint<32>>
  }

  // CHECK-LABEL: IsInvalidIssue572
  // https://github.com/llvm/circt/issues/572
  module private @IsInvalidIssue572(in %a: !firrtl.analog<1>) {
    // CHECK-NEXT: %0 = sv.read_inout %a : !hw.inout<i1>

    // CHECK-NEXT: %.invalid_analog = sv.wire : !hw.inout<i1>
    // CHECK-NEXT: %1 = sv.read_inout %.invalid_analog : !hw.inout<i1>
    %0 = invalidvalue : !firrtl.analog<1>

    // CHECK-NEXT: sv.ifdef "SYNTHESIS"  {
    // CHECK-NEXT:   sv.assign %a, %1 : i1
    // CHECK-NEXT:   sv.assign %.invalid_analog, %0 : i1
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.ifdef "verilator" {
    // CHECK-NEXT:     sv.verbatim "`error \22Verilator does not support alias and thus cannot arbitrarily connect bidirectional wires and ports\22"
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:     sv.alias %a, %.invalid_analog : !hw.inout<i1>, !hw.inout<i1>
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    attach %a, %0 : !firrtl.analog<1>, !firrtl.analog<1>
  }

  // CHECK-LABEL: IsInvalidIssue654
  // https://github.com/llvm/circt/issues/654
  module private @IsInvalidIssue654() {
    %w = wire : !firrtl.uint<0>
    %0 = invalidvalue : !firrtl.uint<0>
    connect %w, %0 : !firrtl.uint<0>, !firrtl.uint<0>
  }

  // CHECK-LABEL: ASQ
  // https://github.com/llvm/circt/issues/699
  module private @ASQ(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %widx_widx_bin = regreset %clock, %reset, %c0_ui1 {name = "widx_widx_bin"} : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<4>
  }

  // CHECK-LABEL: hw.module private @Struct0bits(%source: !hw.struct<valid: i1, ready: i1, data: i0>) {
  // CHECK-NEXT:    hw.output
  // CHECK-NEXT:  }
  module private @Struct0bits(in %source: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<0>>) {
    %2 = subfield %source[data] : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<0>>
  }

  // CHECK-LABEL: hw.module private @MemDepth1
  module private @MemDepth1(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>,
                           in %addr: !firrtl.uint<1>, out %data: !firrtl.uint<32>) {
    // CHECK: %mem0_ext.R0_data = hw.instance "mem0_ext" @mem0_combMem(R0_addr: %addr: i1, R0_en: %en: i1, R0_clk: %clock: i1) -> (R0_data: i32)
    // CHECK: hw.output %mem0_ext.R0_data : i32
    %mem0_load0 = mem Old {depth = 1 : i64, name = "mem0", portNames = ["load0"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    %0 = subfield %mem0_load0[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    connect %0, %clock : !firrtl.clock, !firrtl.clock
    %1 = subfield %mem0_load0[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    connect %1, %addr : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = subfield %mem0_load0[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    connect %data, %2 : !firrtl.uint<32>, !firrtl.uint<32>
    %3 = subfield %mem0_load0[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    connect %3, %en : !firrtl.uint<1>, !firrtl.uint<1>
}

  // https://github.com/llvm/circt/issues/1115
  // CHECK-LABEL: hw.module private @issue1115
  module private @issue1115(in %a: !firrtl.sint<20>, out %tmp59: !firrtl.sint<2>) {
    %0 = shr %a, 21 : (!firrtl.sint<20>) -> !firrtl.sint<1>
    connect %tmp59, %0 : !firrtl.sint<2>, !firrtl.sint<1>
  }

  // CHECK-LABEL: issue1303
  module private @issue1303(out %out: !firrtl.reset) {
    %c1_ui = constant 1 : !firrtl.uint<1>
    connect %out, %c1_ui : !firrtl.reset, !firrtl.uint<1>
    // CHECK-NEXT: %true = hw.constant true
    // CHECK-NEXT: hw.output %true
  }

  // CHECK-LABEL: hw.module private @Force
  module private @Force(in %in: !firrtl.uint<42>) {
    // CHECK: %foo = sv.verbatim.expr.se "foo" : () -> !hw.inout<i42>
    // CHECK: sv.initial {
    // CHECK:   sv.force %foo, %in : i42
    // CHECK: }
    %foo = verbatim.wire "foo" : () -> !firrtl.uint<42>
    force %foo, %in : !firrtl.uint<42>, !firrtl.uint<42>
  }

  extmodule @chkcoverAnno(in clock: !firrtl.clock) attributes {annotations = [{class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}
  // chckcoverAnno is extracted because it is instantiated inside the DUT.
  // CHECK-LABEL: hw.module.extern @chkcoverAnno(%clock: i1)
  // CHECK-SAME: attributes {firrtl.extract.cover.extra}

  extmodule @chkcoverAnno2(in clock: !firrtl.clock) attributes {annotations = [{class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}
  // checkcoverAnno2 is NOT extracted because it is not instantiated under the
  // DUT.
  // CHECK-LABEL: hw.module.extern @chkcoverAnno2(%clock: i1)
  // CHECK-NOT: attributes {firrtl.extract.cover.extra}

  // CHECK-LABEL: hw.module.extern @InnerNamesExt
  // CHECK-SAME:  (
  // CHECK-SAME:    clockIn: i1 {hw.exportPort = #hw<innerSym@extClockInSym>}
  // CHECK-SAME:  ) -> (
  // CHECK-SAME:    clockOut: i1 {hw.exportPort = #hw<innerSym@extClockOutSym>}
  // CHECK-SAME:  )
  extmodule @InnerNamesExt(
    in clockIn: !firrtl.clock sym @extClockInSym,
    out clockOut: !firrtl.clock sym @extClockOutSym
  )
  attributes {annotations = [{class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}

  // CHECK-LABEL: hw.module private @FooDUT
  module private @FooDUT() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %chckcoverAnno_clock = instance chkcoverAnno @chkcoverAnno(in clock: !firrtl.clock)
  }

  // CHECK-LABEL: hw.module private @MemoryWritePortBehavior
  module private @MemoryWritePortBehavior(in %clock1: !firrtl.clock, in %clock2: !firrtl.clock) {
    // This memory has both write ports driven by the same clock.  It should be
    // lowered to an "aa" memory. Even if the clock is passed via different wires,
    // we should identify the clocks to be same.
    //
    // CHECK: hw.instance "aa_ext" @aa_combMem
    %memory_aa_w0, %memory_aa_w1 = mem Undefined {depth = 16 : i64, name = "aa", portNames = ["w0", "w1"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %clk_aa_w0 = subfield %memory_aa_w0[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %clk_aa_w1 = subfield %memory_aa_w1[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %cwire1 = wire : !firrtl.clock
    %cwire2 = wire : !firrtl.clock
    connect %cwire1, %clock1 : !firrtl.clock, !firrtl.clock
    connect %cwire2, %clock1 : !firrtl.clock, !firrtl.clock
    connect %clk_aa_w0, %cwire1 : !firrtl.clock, !firrtl.clock
    connect %clk_aa_w1, %cwire2 : !firrtl.clock, !firrtl.clock

    // This memory has different clocks for each write port.  It should be
    // lowered to an "ab" memory.
    //
    // CHECK: hw.instance "ab_ext" @ab_combMem
    %memory_ab_w0, %memory_ab_w1 = mem Undefined {depth = 16 : i64, name = "ab", portNames = ["w0", "w1"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %clk_ab_w0 = subfield %memory_ab_w0[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %clk_ab_w1 = subfield %memory_ab_w1[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    connect %clk_ab_w0, %clock1 : !firrtl.clock, !firrtl.clock
    connect %clk_ab_w1, %clock2 : !firrtl.clock, !firrtl.clock

    // This memory is the same as the first memory, but a node is used to alias
    // the second write port clock (e.g., this could be due to a dont touch
    // annotation blocking this from being optimized away).  This should be
    // lowered to an "aa" since they are identical.
    //
    // CHECK: hw.instance "ab_node_ext" @aa_combMem
    %memory_ab_node_w0, %memory_ab_node_w1 = mem Undefined {depth = 16 : i64, name = "ab_node", portNames = ["w0", "w1"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %clk_ab_node_w0 = subfield %memory_ab_node_w0[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %clk_ab_node_w1 = subfield %memory_ab_node_w1[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    connect %clk_ab_node_w0, %clock1 : !firrtl.clock, !firrtl.clock
    %tmp = node %clock1 : !firrtl.clock
    connect %clk_ab_node_w1, %tmp : !firrtl.clock, !firrtl.clock
  }

  // CHECK-LABEL: hw.module private @AsyncResetBasic(
  module private @AsyncResetBasic(in %clock: !firrtl.clock, in %arst: !firrtl.asyncreset, in %srst: !firrtl.uint<1>) {
    %c9_ui42 = constant 9 : !firrtl.uint<42>
    %c-9_si42 = constant -9 : !firrtl.sint<42>
    // The following should not error because the reset values are constant.
    %r0 = regreset %clock, %arst, %c9_ui42 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<42>, !firrtl.uint<42>
    %r1 = regreset %clock, %srst, %c9_ui42 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<42>, !firrtl.uint<42>
    %r2 = regreset %clock, %arst, %c-9_si42 : !firrtl.clock, !firrtl.asyncreset, !firrtl.sint<42>, !firrtl.sint<42>
    %r3 = regreset %clock, %srst, %c-9_si42 : !firrtl.clock, !firrtl.uint<1>, !firrtl.sint<42>, !firrtl.sint<42>
  }

  // CHECK-LABEL: hw.module private @BitCast1
  module private @BitCast1() {
    %a = wire : !firrtl.vector<uint<2>, 13>
    %b = bitcast %a : (!firrtl.vector<uint<2>, 13>) -> (!firrtl.uint<26>)
    // CHECK: hw.bitcast %a : (!hw.array<13xi2>) -> i26
  }

  // CHECK-LABEL: hw.module private @BitCast2
  module private @BitCast2() {
    %a = wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>
    %b = bitcast %a : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>) -> (!firrtl.uint<3>)
    // CHECK: hw.bitcast %a : (!hw.struct<valid: i1, ready: i1, data: i1>) -> i3

  }
  // CHECK-LABEL: hw.module private @BitCast3
  module private @BitCast3() {
    %a = wire : !firrtl.uint<26>
    %b = bitcast %a : (!firrtl.uint<26>) -> (!firrtl.vector<uint<2>, 13>)
    // CHECK: hw.bitcast %a : (i26) -> !hw.array<13xi2>
  }

  // CHECK-LABEL: hw.module private @BitCast4
  module private @BitCast4() {
    %a = wire : !firrtl.uint<3>
    %b = bitcast %a : (!firrtl.uint<3>) -> (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>)
    // CHECK: hw.bitcast %a : (i3) -> !hw.struct<valid: i1, ready: i1, data: i1>

  }
  // CHECK-LABEL: hw.module private @BitCast5
  module private @BitCast5() {
    %a = wire : !firrtl.bundle<valid: uint<2>, ready: uint<1>, data: uint<3>>
    %b = bitcast %a : (!firrtl.bundle<valid: uint<2>, ready: uint<1>, data: uint<3>>) -> (!firrtl.vector<uint<2>, 3>)
    // CHECK: hw.bitcast %a : (!hw.struct<valid: i2, ready: i1, data: i3>) -> !hw.array<3xi2>
  }

  // CHECK-LABEL: hw.module private @InnerNames
  // CHECK-SAME:  (
  // CHECK-SAME:    %value: i42 {hw.exportPort = #hw<innerSym@portValueSym>}
  // CHECK-SAME:    %clock: i1 {hw.exportPort = #hw<innerSym@portClockSym>}
  // CHECK-SAME:    %reset: i1 {hw.exportPort = #hw<innerSym@portResetSym>}
  // CHECK-SAME:  ) -> (
  // CHECK-SAME:    out: i1 {hw.exportPort = #hw<innerSym@portOutSym>}
  // CHECK-SAME:  )
  module private @InnerNames(
    in %value: !firrtl.uint<42> sym @portValueSym,
    in %clock: !firrtl.clock sym @portClockSym,
    in %reset: !firrtl.uint<1> sym @portResetSym,
    out %out: !firrtl.uint<1> sym @portOutSym
  ) {
    instance instName sym @instSym @BitCast1()
    // CHECK: hw.instance "instName" sym @instSym @BitCast1
    %nodeName = node sym @nodeSym %value : !firrtl.uint<42>
    // CHECK: %nodeName = hw.wire %value sym @nodeSym : i42
    %wireName = wire sym @wireSym : !firrtl.uint<42>
    // CHECK: %wireName = hw.wire %z_i42 sym @wireSym : i42
    %regName = reg sym @regSym %clock : !firrtl.clock, !firrtl.uint<42>
    // CHECK: %regName = seq.firreg %regName clock %clock sym @regSym : i42
    %regResetName = regreset sym @regResetSym %clock, %reset, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<42>, !firrtl.uint<42>
    // CHECK: %regResetName = seq.firreg %regResetName clock %clock sym @regResetSym reset sync %reset, %value : i42
    %memName_port = mem sym @memSym Undefined {depth = 12 : i64, name = "memName", portNames = ["port"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    // CHECK: {{%.+}} = hw.instance "memName_ext" sym @memSym
    connect %out, %reset : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module private @connectNarrowUIntVector
  module private @connectNarrowUIntVector(in %clock: !firrtl.clock, in %a: !firrtl.vector<uint<1>, 1>, out %b: !firrtl.vector<uint<3>, 1>) {
    %r1 = reg %clock  : !firrtl.clock, !firrtl.vector<uint<2>, 1>
    connect %r1, %a : !firrtl.vector<uint<2>, 1>, !firrtl.vector<uint<1>, 1>
    connect %b, %r1 : !firrtl.vector<uint<3>, 1>, !firrtl.vector<uint<2>, 1>
    // CHECK:      [[OUT:%.+]] = hw.wire [[T6:%.+]] : !hw.array<1xi3>
    // CHECK-NEXT: %r1 = seq.firreg [[T3:%.+]] clock %clock : !hw.array<1xi2>
    // CHECK-NEXT: [[T1:%.+]] = hw.array_get %a[%false] : !hw.array<1xi1>
    // CHECK-NEXT: [[T2:%.+]] = comb.concat %false, [[T1]] : i1, i1
    // CHECK-NEXT: [[T3]] = hw.array_create [[T2]] : i2
    // CHECK-NEXT: [[T4:%.+]] = hw.array_get %r1[%false] : !hw.array<1xi2>
    // CHECK-NEXT: [[T5:%.+]] = comb.concat %false, [[T4]] : i1, i2
    // CHECK-NEXT: [[T6]] = hw.array_create [[T5]] : i3
    // CHECK-NEXT: hw.output [[OUT]] : !hw.array<1xi3>
  }

  // CHECK-LABEL: hw.module private @connectNarrowSIntVector
  module private @connectNarrowSIntVector(in %clock: !firrtl.clock, in %a: !firrtl.vector<sint<1>, 1>, out %b: !firrtl.vector<sint<3>, 1>) {
    %r1 = reg %clock  : !firrtl.clock, !firrtl.vector<sint<2>, 1>
    connect %r1, %a : !firrtl.vector<sint<2>, 1>, !firrtl.vector<sint<1>, 1>
    connect %b, %r1 : !firrtl.vector<sint<3>, 1>, !firrtl.vector<sint<2>, 1>
    // CHECK:      [[OUT:%.+]] = hw.wire [[T7:%.+]] : !hw.array<1xi3>
    // CHECK-NEXT: %r1 = seq.firreg [[T3:%.+]] clock %clock : !hw.array<1xi2>
    // CHECK-NEXT: [[T1:%.+]] = hw.array_get %a[%false] : !hw.array<1xi1>
    // CHECK-NEXT: [[T2:%.+]] = comb.concat [[T1]], [[T1]] : i1, i1
    // CHECK-NEXT: [[T3]] = hw.array_create [[T2]] : i2
    // CHECK-NEXT: [[T4:%.+]] = hw.array_get %r1[%false] : !hw.array<1xi2>
    // CHECK-NEXT: [[T5:%.+]] = comb.extract [[T4]] from 1 : (i2) -> i1
    // CHECK-NEXT: [[T6:%.+]] = comb.concat [[T5]], [[T4]] : i1, i2
    // CHECK-NEXT: [[T7]] = hw.array_create [[T6]] : i3
    // CHECK-NEXT: hw.output [[OUT]] : !hw.array<1xi3>
  }

  // CHECK-LABEL: hw.module private @SubIndex
  module private @SubIndex(in %a: !firrtl.vector<vector<uint<1>, 1>, 1>, in %clock: !firrtl.clock, out %o1: !firrtl.uint<1>, out %o2: !firrtl.vector<uint<1>, 1>) {
    %r1 = reg %clock  : !firrtl.clock, !firrtl.uint<1>
    %r2 = reg %clock  : !firrtl.clock, !firrtl.vector<uint<1>, 1>
    %0 = subindex %a[0] : !firrtl.vector<vector<uint<1>, 1>, 1>
    %1 = subindex %0[0] : !firrtl.vector<uint<1>, 1>
    connect %r1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %r2, %0 : !firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>
    connect %o1, %r1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %o2, %r2 : !firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>
    // CHECK:      %r1 = seq.firreg %1 clock %clock : i1
    // CHECK-NEXT: %r2 = seq.firreg %0 clock %clock : !hw.array<1xi1>
    // CHECK-NEXT: %0 = hw.array_get %a[%false] : !hw.array<1xarray<1xi1>>
    // CHECK-NEXT: %1 = hw.array_get %0[%false] : !hw.array<1xi1>
    // CHECK-NEXT: hw.output %r1, %r2 : i1, !hw.array<1xi1>
  }

  // CHECK-LABEL: hw.module private @SubAccess
  module private @SubAccess(in %x: !firrtl.uint<1>, in %y: !firrtl.uint<1>, in %a: !firrtl.vector<vector<uint<1>, 1>, 1>, in %clock: !firrtl.clock, out %o1: !firrtl.uint<1>, out %o2: !firrtl.vector<uint<1>, 1>) {
    %r1 = reg %clock  : !firrtl.clock, !firrtl.uint<1>
    %r2 = reg %clock  : !firrtl.clock, !firrtl.vector<uint<1>, 1>
    %0 = subaccess %a[%x] : !firrtl.vector<vector<uint<1>, 1>, 1>, !firrtl.uint<1>
    %1 = subaccess %0[%y] : !firrtl.vector<uint<1>, 1>, !firrtl.uint<1>
    connect %r1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %r2, %0 : !firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>
    connect %o1, %r1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %o2, %r2 : !firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>
    // CHECK:      %r1 = seq.firreg %1 clock %clock : i1
    // CHECK-NEXT: %r2 = seq.firreg %0 clock %clock : !hw.array<1xi1>
    // CHECK-NEXT: %0 = hw.array_get %a[%x] : !hw.array<1xarray<1xi1>>, i1
    // CHECK-NEXT: %1 = hw.array_get %0[%y] : !hw.array<1xi1>, i1
    // CHECK-NEXT: hw.output %r1, %r2 : i1, !hw.array<1xi1>
  }

  // CHECK-LABEL: hw.module private @zero_width_constant()
  // https://github.com/llvm/circt/issues/2269
  module private @zero_width_constant(out %a: !firrtl.uint<0>) {
    // CHECK-NEXT: hw.output
    %c0_ui0 = constant 0 : !firrtl.uint<0>
    connect %a, %c0_ui0 : !firrtl.uint<0>, !firrtl.uint<0>
  }

  // CHECK-LABEL: hw.module private @RegResetStructWiden
  module private @RegResetStructWiden(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %init: !firrtl.bundle<a: uint<2>>) {
    // CHECK:      [[FALSE:%.*]] = hw.constant false
    // CHECK-NEXT: [[A:%.*]] = hw.struct_extract %init["a"] : !hw.struct<a: i2>
    // CHECK-NEXT: [[PADDED:%.*]] = comb.concat [[FALSE]], [[A]] : i1, i2
    // CHECK-NEXT: [[STRUCT:%.*]] = hw.struct_create ([[PADDED]]) : !hw.struct<a: i3>
    // CHECK-NEXT: %reg = seq.firreg %reg clock %clock reset sync %reset, [[STRUCT]] : !hw.struct<a: i3>
    %reg = regreset %clock, %reset, %init  : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: uint<2>>, !firrtl.bundle<a: uint<3>>
  }

  // CHECK-LABEL: hw.module private @AggregateInvalidValue
  module private @AggregateInvalidValue(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %invalid = invalidvalue : !firrtl.bundle<a: uint<1>, b: vector<uint<10>, 10>>
    %reg = regreset %clock, %reset, %invalid : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: uint<1>, b: vector<uint<10>, 10>>, !firrtl.bundle<a: uint<1>, b: vector<uint<10>, 10>>
    // CHECK:      %c0_i101 = hw.constant 0 : i101
    // CHECK-NEXT: %0 = hw.bitcast %c0_i101 : (i101) -> !hw.struct<a: i1, b: !hw.array<10xi10>>
    // CHECK-NEXT: %reg = seq.firreg %reg clock %clock reset sync %reset, %0 : !hw.struct<a: i1, b: !hw.array<10xi10>>
  }

  // CHECK-LABEL: hw.module private @ForceNameSubmodule
  hw.hierpath private @nla_1 [@ForceNameTop::@sym_foo, @ForceNameSubmodule]
  hw.hierpath private @nla_2 [@ForceNameTop::@sym_bar, @ForceNameSubmodule]
  hw.hierpath private @nla_3 [@ForceNameTop::@sym_baz, @ForceNameSubextmodule]
  module private @ForceNameSubmodule() attributes {annotations = [
    {circt.nonlocal = @nla_2,
     class = "chisel3.util.experimental.ForceNameAnnotation", name = "Bar"},
    {circt.nonlocal = @nla_1,
     class = "chisel3.util.experimental.ForceNameAnnotation", name = "Foo"}]} {}
  extmodule private @ForceNameSubextmodule() attributes {annotations = [
    {circt.nonlocal = @nla_3,
     class = "chisel3.util.experimental.ForceNameAnnotation", name = "Baz"}]}
  // CHECK: hw.module private @ForceNameTop
  module private @ForceNameTop() {
    instance foo sym @sym_foo
      {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}]}
      @ForceNameSubmodule()
    instance bar sym @sym_bar
      {annotations = [{circt.nonlocal = @nla_2, class = "circt.nonlocal"}]}
      @ForceNameSubmodule()
    instance baz sym @sym_baz
      {annotations = [{circt.nonlocal = @nla_3, class = "circt.nonlocal"}]}
      @ForceNameSubextmodule()
    // CHECK:      hw.instance "foo" sym @sym_foo {{.+}} {hw.verilogName = "Foo"}
    // CHECK-NEXT: hw.instance "bar" sym @sym_bar {{.+}} {hw.verilogName = "Bar"}
    // CHECK-NEXT: hw.instance "baz" sym @sym_baz {{.+}} {hw.verilogName = "Baz"}
  }

  // CHECK-LABEL: hw.module private @PreserveName
  module private @PreserveName(in %a : !firrtl.uint<1>, in %b : !firrtl.uint<1>, out %c : !firrtl.uint<1>) {
    //CHECK comb.or %a, %b {sv.namehint = "myname"}
    %foo = or %a, %b {name = "myname"} : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %c, %foo : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK: comb.shl bin {{.*}} {sv.namehint = "anothername"}
    %bar = dshl %a, %b {name = "anothername"} : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  }

  // CHECK-LABEL: hw.module private @MultibitMux(%source_0: i1, %source_1: i1, %source_2: i1, %index: i2) -> (sink: i1) {
  module private @MultibitMux(in %source_0: !firrtl.uint<1>, in %source_1: !firrtl.uint<1>, in %source_2: !firrtl.uint<1>, out %sink: !firrtl.uint<1>, in %index: !firrtl.uint<2>) {
    %0 = multibit_mux %index, %source_2, %source_1, %source_0 : !firrtl.uint<2>, !firrtl.uint<1>
    connect %sink, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:      %c0_i2 = hw.constant 0 : i2
    // CHECK:      %0 = hw.array_create %source_2, %source_1, %source_0 : i1
    // CHECK-NEXT: %1 = hw.array_get %0[%c0_i2]
    // CHECK-NEXT: %2 = hw.array_create %1 : i1
    // CHECK-NEXT: %3 = hw.array_concat %2, %0
    // CHECK-NEXT: %4 = hw.array_get %3[%index]
    // CHECK-NEXT: hw.output %4 : i1
  }

  module private @inferUnmaskedMemory(in %clock: !firrtl.clock, in %rAddr: !firrtl.uint<4>, in %rEn: !firrtl.uint<1>, out %rData: !firrtl.uint<8>, in %wMask: !firrtl.uint<1>, in %wData: !firrtl.uint<8>) {
    %tbMemoryKind1_r, %tbMemoryKind1_w = mem Undefined  {depth = 16 : i64, modName = "tbMemoryKind1_ext", name = "tbMemoryKind1", portNames = ["r", "w"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %0 = subfield %tbMemoryKind1_w[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %1 = subfield %tbMemoryKind1_w[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %2 = subfield %tbMemoryKind1_w[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %3 = subfield %tbMemoryKind1_w[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %4 = subfield %tbMemoryKind1_w[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %5 = subfield %tbMemoryKind1_r[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
    %6 = subfield %tbMemoryKind1_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
    %7 = subfield %tbMemoryKind1_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
    %8 = subfield %tbMemoryKind1_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
    connect %8, %clock : !firrtl.clock, !firrtl.clock
    connect %7, %rEn : !firrtl.uint<1>, !firrtl.uint<1>
    connect %6, %rAddr : !firrtl.uint<4>, !firrtl.uint<4>
    connect %rData, %5 : !firrtl.uint<8>, !firrtl.uint<8>
    connect %4, %clock : !firrtl.clock, !firrtl.clock
    connect %3, %rEn : !firrtl.uint<1>, !firrtl.uint<1>
    connect %2, %rAddr : !firrtl.uint<4>, !firrtl.uint<4>
    connect %1, %wMask : !firrtl.uint<1>, !firrtl.uint<1>
    connect %0, %wData : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: hw.module private @inferUnmaskedMemory
  // CHECK-NEXT:   %[[v0:.+]] = comb.and bin %rEn, %wMask : i1
  // CHECK-NEXT:   %tbMemoryKind1_ext.R0_data = hw.instance "tbMemoryKind1_ext" @tbMemoryKind1_combMem(R0_addr: %rAddr: i4, R0_en: %rEn: i1, R0_clk: %clock: i1, W0_addr: %rAddr: i4, W0_en: %[[v0]]: i1, W0_clk: %clock: i1, W0_data: %wData: i8) -> (R0_data: i8)
  // CHECK-NEXT:   hw.output %tbMemoryKind1_ext.R0_data : i8

  // CHECK-LABEL: hw.module private @eliminateSingleOutputConnects
  // CHECK-NOT:     [[WIRE:%.+]] = sv.wire
  // CHECK-NEXT:    hw.output %a : i1
  module private @eliminateSingleOutputConnects(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    strictconnect %b, %a : !firrtl.uint<1>
  }

  // Check that modules with comments are lowered.
  // CHECK-LABEL: hw.module private @Commented() attributes {
  // CHECK-SAME:      comment = "this module is commented"
  // CHECK-SAME:  }
  module private @Commented() attributes {
      comment = "this module is commented"
  } {}

  // CHECK-LABEL: hw.module @preLoweredOps
  module @preLoweredOps() {
    // CHECK-NEXT: %0 = builtin.unrealized_conversion_cast to f32
    // CHECK-NEXT: %1 = arith.addf %0, %0 : f32
    // CHECK-NEXT: builtin.unrealized_conversion_cast %1 : f32 to index
    %0 = builtin.unrealized_conversion_cast to f32
    %1 = arith.addf %0, %0 : f32
    builtin.unrealized_conversion_cast %1 : f32 to index
  }

  // Used for testing.
  extmodule private @Blackbox(in inst: !firrtl.uint<1>)

  // Check that the following doesn't crash, when we have a no-op cast which
  // uses an input port.
  // CHECK-LABEL: hw.module private @BackedgesAndNoopCasts
  // CHECK-NEXT:    hw.instance "blackbox" @Blackbox(inst: %clock: i1) -> ()
  // CHECK-NEXT:    hw.output %clock : i1
  module private @BackedgesAndNoopCasts(in %clock: !firrtl.uint<1>, out %out : !firrtl.clock) {
    // Following comments describe why this used to crash.
    // Blackbox input port creates a backedge.
    %inst = instance blackbox @Blackbox(in inst: !firrtl.uint<1>)
    // No-op cast is removed, %cast lowered to point directly to the backedge.
    %cast = asClock %inst : (!firrtl.uint<1>) -> !firrtl.clock
    // Finalize the backedge, replacing all uses with %clock.
    strictconnect %inst, %clock : !firrtl.uint<1>
    // %cast accidentally still points to the back edge in the lowering table.
    strictconnect %out, %cast : !firrtl.clock
  }

  // Check that when inputs are connected to other inputs, the backedges are
  // properly resolved to the final real value.
  // CHECK-LABEL: hw.module @ChainedBackedges
  // CHECK-NEXT:    hw.instance "a" @Blackbox
  // CHECK-NEXT:    hw.instance "b" @Blackbox
  // CHECK-NEXT:    hw.output %in : i1
  module @ChainedBackedges(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %a_inst = instance a @Blackbox(in inst: !firrtl.uint<1>)
    %b_inst = instance b @Blackbox(in inst: !firrtl.uint<1>)
    strictconnect %a_inst, %in : !firrtl.uint<1>
    strictconnect %b_inst, %a_inst : !firrtl.uint<1>
    strictconnect %out, %b_inst : !firrtl.uint<1>
  }

  // Check that combinational cycles with no outside driver are lowered to
  // be driven from a wire.
  // CHECK-LABEL: hw.module @UndrivenInputPort()
  // CHECK-NEXT:    %undriven = sv.wire : !hw.inout<i1>
  // CHECK-NEXT:    %0 = sv.read_inout %undriven : !hw.inout<i1>
  // CHECK-NEXT:    hw.instance "blackbox" @Blackbox(inst: %0: i1) -> ()
  // CHECK-NEXT:    hw.instance "blackbox" @Blackbox(inst: %0: i1) -> ()
  module @UndrivenInputPort() {
    %0 = instance blackbox @Blackbox(in inst : !firrtl.uint<1>)
    %1 = instance blackbox @Blackbox(in inst : !firrtl.uint<1>)
    strictconnect %0, %1 : !firrtl.uint<1>
    strictconnect %1, %0 : !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module @LowerToFirReg(%clock: i1, %reset: i1, %value: i2)
  module @LowerToFirReg(
    in %clock: !firrtl.clock,
    in %reset: !firrtl.uint<1>,
    in %value: !firrtl.uint<2>
  ) {
    %regA = reg %clock: !firrtl.clock, !firrtl.uint<2>
    %regB = regreset %clock, %reset, %value: !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>
    strictconnect %regA, %value : !firrtl.uint<2>
    strictconnect %regB, %value : !firrtl.uint<2>
    // CHECK-NEXT: %regA = seq.firreg %value clock %clock : i2
    // CHECK-NEXT: %regB = seq.firreg %value clock %clock reset sync %reset, %value : i2
  }

  // CHECK-LABEL: hw.module @SyncReset(%clock: i1, %reset: i1, %value: i2) -> (result: i2)
  module @SyncReset(in %clock: !firrtl.clock,
                           in %reset: !firrtl.uint<1>,
                           in %value: !firrtl.uint<2>,
                           out %result: !firrtl.uint<2>) {
    %count = regreset %clock, %reset, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>

    // CHECK: %count = seq.firreg %count clock %clock reset sync %reset, %value : i2
    // CHECK: hw.output %count : i2

    strictconnect %result, %count : !firrtl.uint<2>
  }

  // CHECK-LABEL: hw.module @AsyncReset(%clock: i1, %reset: i1, %value: i2) -> (result: i2)
  module @AsyncReset(in %clock: !firrtl.clock,
                           in %reset: !firrtl.asyncreset,
                           in %value: !firrtl.uint<2>,
                           out %result: !firrtl.uint<2>) {
    %count = regreset %clock, %reset, %value : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<2>, !firrtl.uint<2>

    // CHECK: %count = seq.firreg %value clock %clock reset async %reset, %value : i2
    // CHECK: hw.output %count : i2

    strictconnect %count, %value : !firrtl.uint<2>
    strictconnect %result, %count : !firrtl.uint<2>
  }

  // CHECK-LABEL: hw.module @NoConnect(%clock: i1, %reset: i1) -> (result: i2)
  module @NoConnect(in %clock: !firrtl.clock,
                     in %reset: !firrtl.uint<1>,
                     out %result: !firrtl.uint<2>) {
    %count = reg %clock: !firrtl.clock, !firrtl.uint<2>
    // CHECK: %count = seq.firreg %count clock %clock : i2

    strictconnect %result, %count : !firrtl.uint<2>

    // CHECK: hw.output %count : i2
  }
  // CHECK-LABEL: hw.module @passThroughForeignTypes
  // CHECK-SAME:      (%inOpaque: index) -> (outOpaque: index) {
  // CHECK-NEXT:    %sub2.bar = hw.instance "sub2" @moreForeignTypes(foo: %sub1.bar: index) -> (bar: index)
  // CHECK-NEXT:    %sub1.bar = hw.instance "sub1" @moreForeignTypes(foo: %inOpaque: index) -> (bar: index)
  // CHECK-NEXT:    hw.output %sub2.bar : index
  // CHECK-NEXT:  }
  // CHECK-LABEL: hw.module @moreForeignTypes
  // CHECK-SAME:      (%foo: index) -> (bar: index) {
  // CHECK-NEXT:    hw.output %foo : index
  // CHECK-NEXT:  }
  module @passThroughForeignTypes(in %inOpaque: index, out %outOpaque: index) {
    // Declaration order intentionally reversed to enforce use-before-def in HW
    %sub2_foo, %sub2_bar = instance sub2 @moreForeignTypes(in foo: index, out bar: index)
    %sub1_foo, %sub1_bar = instance sub1 @moreForeignTypes(in foo: index, out bar: index)
    strictconnect %sub1_foo, %inOpaque : index
    strictconnect %sub2_foo, %sub1_bar : index
    strictconnect %outOpaque, %sub2_bar : index
  }
  module @moreForeignTypes(in %foo: index, out %bar: index) {
    strictconnect %bar, %foo : index
  }

  // CHECK-LABEL: hw.module @foreignOpsOnForeignTypes
  // CHECK-SAME:      (%x: f32) -> (y: f32) {
  // CHECK-NEXT:    [[TMP:%.+]] = arith.addf %x, %x : f32
  // CHECK-NEXT:    hw.output [[TMP]] : f32
  // CHECK-NEXT:  }
  module @foreignOpsOnForeignTypes(in %x: f32, out %y: f32) {
    %0 = arith.addf %x, %x : f32
    strictconnect %y, %0 : f32
  }

  // CHECK-LABEL: hw.module @wiresWithForeignTypes
  // CHECK-SAME:      (%in: f32) -> (out: f32) {
  // CHECK-NEXT:    [[ADD1:%.+]] = arith.addf [[ADD2:%.+]], [[ADD2]] : f32
  // CHECK-NEXT:    [[ADD2]] = arith.addf %in, [[ADD2]] : f32
  // CHECK-NEXT:    hw.output [[ADD1]] : f32
  // CHECK-NEXT:  }
  module @wiresWithForeignTypes(in %in: f32, out %out: f32) {
    %w1 = wire : f32
    %w2 = wire : f32
    strictconnect %out, %w2 : f32
    %0 = arith.addf %w1, %w1 : f32
    strictconnect %w2, %0 : f32
    %1 = arith.addf %in, %w1 : f32
    strictconnect %w1, %1 : f32
  }

  // CHECK-LABEL: LowerReadArrayInoutIntoArrayGet
  module @LowerReadArrayInoutIntoArrayGet(in %a: !firrtl.vector<uint<10>, 1>, out %b: !firrtl.uint<10>) {
    %r = wire : !firrtl.vector<uint<10>, 1>
    %0 = subindex %r[0] : !firrtl.vector<uint<10>, 1>
    // CHECK:      %r = hw.wire %a : !hw.array<1xi10>
    // CHECK-NEXT: [[RET:%.+]] = hw.array_get %r[%false] : !hw.array<1xi10>, i1
    // CHECK-NEXT: hw.output [[RET]]
    strictconnect %r, %a : !firrtl.vector<uint<10>, 1>
    strictconnect %b, %0 : !firrtl.uint<10>
  }

  // CHECK-LABEL: hw.module @MergeBundle
  module @MergeBundle(out %o: !firrtl.bundle<valid: uint<1>, ready: uint<1>>, in %i: !firrtl.uint<1>) {
    %a = wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
    strictconnect %o, %a : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
    %0 = bundlecreate %i, %i : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.bundle<valid: uint<1>, ready: uint<1>>
    strictconnect %a, %0 : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
    // CHECK:      %a = hw.wire [[BUNDLE:%.+]] : !hw.struct<valid: i1, ready: i1>
    // CHECK-NEXT: [[BUNDLE]] = hw.struct_create (%i, %i) : !hw.struct<valid: i1, ready: i1>
    // CHECK-NEXT: hw.output %a : !hw.struct<valid: i1, ready: i1>
  }

  // CHECK-LABEL: hw.module @MergeVector
  module @MergeVector(out %o: !firrtl.vector<uint<1>, 3>, in %i: !firrtl.uint<1>, in %j: !firrtl.uint<1>) {
    %a = wire : !firrtl.vector<uint<1>, 3>
    strictconnect %o, %a : !firrtl.vector<uint<1>, 3>
    %0 = vectorcreate %i, %i, %j : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 3>
    strictconnect %a, %0 : !firrtl.vector<uint<1>, 3>
    // CHECK:      %a = hw.wire [[VECTOR:%.+]] : !hw.array<3xi1>
    // CHECK-NEXT: [[VECTOR]] = hw.array_create %j, %i, %i : i1
    // CHECK-NEXT: hw.output %a : !hw.array<3xi1>
  }

  // CHECK-LABEL: hw.module @aggregateconstant
  module @aggregateconstant(out %out : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 2>, b: vector<vector<uint<8>, 2>, 2>>) {
    %0 = aggregateconstant [[[0 : ui8, 1: ui8], [2 : ui8, 3: ui8]], [[4: ui8, 5: ui8], [6: ui8, 7:ui8]]] :
      !firrtl.bundle<a: vector<vector<uint<8>, 2>, 2>, b: vector<vector<uint<8>, 2>, 2>>
    strictconnect %out, %0 : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 2>, b: vector<vector<uint<8>, 2>, 2>>
    // CHECK{LITERAL}:   %0 = hw.aggregate_constant [[[3 : i8, 2 : i8], [1 : i8, 0 : i8]], [[7 : i8, 6 : i8], [5 : i8, 4 : i8]]]
    // CHECK-SAME: !hw.struct<a: !hw.array<2xarray<2xi8>>, b: !hw.array<2xarray<2xi8>>>
    // CHECK: hw.output %0
  }

  // An internal-only analog connection between two instances should be implemented with a wire
  extmodule @AnalogInModA(in a: !firrtl.analog<8>)
  extmodule @AnalogInModB(in a: !firrtl.analog<8>)
  extmodule @AnalogOutModA(out a: !firrtl.analog<8>)
  module @AnalogMergeTwo() {
    %result_iIn = instance iIn @AnalogInModA(in a: !firrtl.analog<8>)
    %result_iOut = instance iOut @AnalogOutModA(out a: !firrtl.analog<8>)
    attach %result_iIn, %result_iOut : !firrtl.analog<8>, !firrtl.analog<8>
  }
  // CHECK-LABEL: hw.module @AnalogMergeTwo() {
  // CHECK:         %.a.wire = sv.wire : !hw.inout<i8>
  // CHECK:         hw.instance "iIn" @AnalogInModA(a: %.a.wire: !hw.inout<i8>) -> ()
  // CHECK:         hw.instance "iOut" @AnalogOutModA(a: %.a.wire: !hw.inout<i8>) -> ()
  // CHECK-NEXT:    hw.output
  // CHECK-NEXT:  }

  // An internal-only analog connection between three instances should be implemented with a wire
  module @AnalogMergeThree() {
    %result_iInA = instance iInA @AnalogInModA(in a: !firrtl.analog<8>)
    %result_iInB = instance iInB @AnalogInModB(in a: !firrtl.analog<8>)
    %result_iOut = instance iOut @AnalogOutModA(out a: !firrtl.analog<8>)
    attach %result_iInA, %result_iInB, %result_iOut : !firrtl.analog<8>, !firrtl.analog<8>, !firrtl.analog<8>
  }
  // CHECK-LABEL: hw.module @AnalogMergeThree() {
  // CHECK:         %.a.wire = sv.wire : !hw.inout<i8>
  // CHECK:         hw.instance "iInA" @AnalogInModA(a: %.a.wire: !hw.inout<i8>) -> ()
  // CHECK:         hw.instance "iInB" @AnalogInModB(a: %.a.wire: !hw.inout<i8>) -> ()
  // CHECK:         hw.instance "iOut" @AnalogOutModA(a: %.a.wire: !hw.inout<i8>) -> ()
  // CHECK-NEXT:    hw.output
  // CHECK-NEXT:  }

  // An analog connection between two instances and a module port should be implemented with a wire
  module @AnalogMergeTwoWithPort(out %a: !firrtl.analog<8>) {
    %result_iIn = instance iIn @AnalogInModA(in a: !firrtl.analog<8>)
    %result_iOut = instance iOut @AnalogOutModA(out a: !firrtl.analog<8>)
    attach %a, %result_iIn, %result_iOut : !firrtl.analog<8>, !firrtl.analog<8>, !firrtl.analog<8>
  }
  // CHECK-LABEL: hw.module @AnalogMergeTwoWithPort(%a: !hw.inout<i8>) {
  // CHECK-NEXT:    hw.instance "iIn" @AnalogInModA(a: %a: !hw.inout<i8>) -> ()
  // CHECK-NEXT:    hw.instance "iOut" @AnalogOutModA(a: %a: !hw.inout<i8>) -> ()
  // CHECK-NEXT:    hw.output
  // CHECK-NEXT:  }

  // Check forceable declarations are kept alive with symbols.
  // CHECK-LABEL: hw.module private @ForceableToSym(
  module private @ForceableToSym(in %in: !firrtl.uint<4>, in %clk: !firrtl.clock, out %out: !firrtl.uint<4>) {
    // CHECK-NEXT: %n = hw.wire %in sym @__ForceableToSym__n : i4
    // CHECK-NEXT: %w = hw.wire %n sym @__ForceableToSym__w : i4
    // CHECK-NEXT: %r = seq.firreg %w clock %clk sym @r : i4
    %n, %n_ref = node %in forceable : !firrtl.uint<4>
    %w, %w_ref = wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    %r, %r_ref = reg %clk forceable : !firrtl.clock, !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>

    strictconnect %w, %n : !firrtl.uint<4>
    strictconnect %r, %w : !firrtl.uint<4>
    strictconnect %out, %r : !firrtl.uint<4>
  }

  // Check lowering force and release operations.
  hw.hierpath private @xmrPath [@ForceRelease::@xmr_sym, @RefMe::@xmr_sym]
  module private @RefMe() {
    %x, %x_ref = wire sym @xmr_sym forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
  }
  // CHECK-LABEL: hw.module @ForceRelease(
  module @ForceRelease(in %c: !firrtl.uint<1>, in %clock: !firrtl.clock, in %x: !firrtl.uint<4>) {
    instance r sym @xmr_sym @RefMe()
    %0 = sv.xmr.ref @xmrPath : !hw.inout<i4>
    %1 = builtin.unrealized_conversion_cast %0 : !hw.inout<i4> to !firrtl.rwprobe<uint<4>>
    ref.force %clock, %c, %1, %x : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>
    %2 = sv.xmr.ref @xmrPath : !hw.inout<i4>
    %3 = builtin.unrealized_conversion_cast %2 : !hw.inout<i4> to !firrtl.rwprobe<uint<4>>
    ref.force_initial %c, %3, %x : !firrtl.uint<1>, !firrtl.uint<4>
    %4 = sv.xmr.ref @xmrPath : !hw.inout<i4>
    %5 = builtin.unrealized_conversion_cast %4 : !hw.inout<i4> to !firrtl.rwprobe<uint<4>>
    ref.release %clock, %c, %5 : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
    %6 = sv.xmr.ref @xmrPath : !hw.inout<i4>
    %7 = builtin.unrealized_conversion_cast %6 : !hw.inout<i4> to !firrtl.rwprobe<uint<4>>
    ref.release_initial %c, %7 : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
  }
  // CHECK-NEXT:  hw.instance "r" sym @xmr_sym @RefMe() -> ()
  // CHECK-NEXT:  %[[XMR1:.+]] = sv.xmr.ref @xmrPath : !hw.inout<i4>
  // CHECK-NEXT:  %[[XMR2:.+]] = sv.xmr.ref @xmrPath : !hw.inout<i4>
  // CHECK-NEXT:  %[[XMR3:.+]] = sv.xmr.ref @xmrPath : !hw.inout<i4>
  // CHECK-NEXT:  %[[XMR4:.+]] = sv.xmr.ref @xmrPath : !hw.inout<i4>
  // CHECK-NEXT:  sv.ifdef  "SYNTHESIS" {
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    sv.always posedge %clock {
  // CHECK-NEXT:      sv.if %c {
  // CHECK-NEXT:        sv.force %[[XMR1]], %x : i4
  // CHECK-NEXT:        sv.release %[[XMR3]] : !hw.inout<i4>
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:    sv.initial {
  // CHECK-NEXT:      sv.if %c {
  // CHECK-NEXT:        sv.force %[[XMR2]], %x : i4
  // CHECK-NEXT:        sv.release %[[XMR4]] : !hw.inout<i4>
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }

  // CHECK-LABEL: @SVAttr
  // CHECK-SAME:  attributes {sv.attributes = [#sv.attribute<"keep_hierarchy = \22true\22">]}
  // CHECK-NEXT: %w = hw.wire %a {sv.attributes = [#sv.attribute<"mark_debug = \22yes\22">]}
  // CHECK-NEXT: %n = hw.wire %w {sv.attributes = [#sv.attribute<"mark_debug = \22yes\22">]}
  // CHECK-NEXT: %r = seq.firreg %a clock %clock {firrtl.random_init_start = 0 : ui64, sv.attributes = [#sv.attribute<"keep = \22true\22", emitAsComment>]}
  module @SVAttr(in %a: !firrtl.uint<1>, in %clock: !firrtl.clock, out %b1: !firrtl.uint<1>, out %b2: !firrtl.uint<1>) attributes {convention = #firrtl<convention scalarized>, sv.attributes = [#sv.attribute<"keep_hierarchy = \22true\22">]} {
    %w = wire {sv.attributes = [#sv.attribute<"mark_debug = \22yes\22">]} : !firrtl.uint<1>
    %n = node %w {sv.attributes = [#sv.attribute<"mark_debug = \22yes\22">]} : !firrtl.uint<1>
    %r = reg %clock {firrtl.random_init_start = 0 : ui64, sv.attributes = [#sv.attribute<"keep = \22true\22", emitAsComment>]} : !firrtl.clock, !firrtl.uint<1>
    strictconnect %w, %a : !firrtl.uint<1>
    strictconnect %b1, %n : !firrtl.uint<1>
    strictconnect %r, %a : !firrtl.uint<1>
    strictconnect %b2, %r : !firrtl.uint<1>
  }

  // CHECK-LABEL: Elementwise
  module @Elementwise(in %a: !firrtl.vector<uint<1>, 2>, in %b: !firrtl.vector<uint<1>, 2>, out %c_0: !firrtl.vector<uint<1>, 2>, out %c_1: !firrtl.vector<uint<1>, 2>, out %c_2: !firrtl.vector<uint<1>, 2>) {
    %0 = elementwise_or %a, %b : (!firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>) -> !firrtl.vector<uint<1>, 2>
    strictconnect %c_0, %0 : !firrtl.vector<uint<1>, 2>
    %1 = elementwise_and %a, %b : (!firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>) -> !firrtl.vector<uint<1>, 2>
    strictconnect %c_1, %1 : !firrtl.vector<uint<1>, 2>
    %2 = elementwise_xor %a, %b : (!firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>) -> !firrtl.vector<uint<1>, 2>
    strictconnect %c_2, %2 : !firrtl.vector<uint<1>, 2>

    // CHECK-NEXT: %0 = hw.bitcast %a : (!hw.array<2xi1>) -> i2
    // CHECK-NEXT: %1 = hw.bitcast %b : (!hw.array<2xi1>) -> i2
    // CHECK-NEXT: %2 = comb.or %0, %1 : i2
    // CHECK-NEXT: %[[OR:.+]] = hw.bitcast %2 : (i2) -> !hw.array<2xi1>

    // CHECK-NEXT: %4 = hw.bitcast %a : (!hw.array<2xi1>) -> i2
    // CHECK-NEXT: %5 = hw.bitcast %b : (!hw.array<2xi1>) -> i2
    // CHECK-NEXT: %6 = comb.and %4, %5 : i2
    // CHECK-NEXT: %[[AND:.+]] = hw.bitcast %6 : (i2) -> !hw.array<2xi1>

    // CHECK-NEXT: %8 = hw.bitcast %a : (!hw.array<2xi1>) -> i2
    // CHECK-NEXT: %9 = hw.bitcast %b : (!hw.array<2xi1>) -> i2
    // CHECK-NEXT: %10 = comb.xor %8, %9 : i2
    // CHECK-NEXT: %[[XOR:.+]] = hw.bitcast %10 : (i2) -> !hw.array<2xi1>

    // CHECK-NEXT: hw.output %[[OR]], %[[AND]], %[[XOR]] : !hw.array<2xi1>, !hw.array<2xi1>, !hw.array<2xi1>
  }
}
