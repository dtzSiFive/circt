// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types))' %s | FileCheck --check-prefixes=CHECK,COMMON %s
// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types{preserve-aggregate=all}))' %s | FileCheck --check-prefixes=AGGREGATE,COMMON %s


firrtl.circuit "TopLevel" {

  // COMMON-LABEL: module private @Simple
  // COMMON-SAME: in %[[SOURCE_VALID_NAME:source_valid]]: [[SOURCE_VALID_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: out %[[SOURCE_READY_NAME:source_ready]]: [[SOURCE_READY_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: in %[[SOURCE_DATA_NAME:source_data]]: [[SOURCE_DATA_TYPE:!firrtl.uint<64>]]
  // COMMON-SAME: out %[[SINK_VALID_NAME:sink_valid]]: [[SINK_VALID_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: in %[[SINK_READY_NAME:sink_ready]]: [[SINK_READY_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: out %[[SINK_DATA_NAME:sink_data]]: [[SINK_DATA_TYPE:!firrtl.uint<64>]]
  module private @Simple(in %source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
                        out %sink: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {

    // COMMON-NEXT: when %[[SOURCE_VALID_NAME]] : !firrtl.uint<1>
    // COMMON-NEXT:   connect %[[SINK_DATA_NAME]], %[[SOURCE_DATA_NAME]] : [[SINK_DATA_TYPE]], [[SOURCE_DATA_TYPE]]
    // COMMON-NEXT:   connect %[[SINK_VALID_NAME]], %[[SOURCE_VALID_NAME]] : [[SINK_VALID_TYPE]], [[SOURCE_VALID_TYPE]]
    // COMMON-NEXT:   connect %[[SOURCE_READY_NAME]], %[[SINK_READY_NAME]] : [[SOURCE_READY_TYPE]], [[SINK_READY_TYPE]]

    %0 = subfield %source[valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    %1 = subfield %source[ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    %2 = subfield %source[data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    %3 = subfield %sink[valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    %4 = subfield %sink[ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    %5 = subfield %sink[data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    when %0 : !firrtl.uint<1> {
      connect %5, %2 : !firrtl.uint<64>, !firrtl.uint<64>
      connect %3, %0 : !firrtl.uint<1>, !firrtl.uint<1>
      connect %1, %4 : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }

  // COMMON-LABEL: module @TopLevel
  // COMMON-SAME: in %source_valid: [[SOURCE_VALID_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: out %source_ready: [[SOURCE_READY_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: in %source_data: [[SOURCE_DATA_TYPE:!firrtl.uint<64>]]
  // COMMON-SAME: out %sink_valid: [[SINK_VALID_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: in %sink_ready: [[SINK_READY_TYPE:!firrtl.uint<1>]]
  // COMMON-SAME: out %sink_data: [[SINK_DATA_TYPE:!firrtl.uint<64>]]
  module @TopLevel(in %source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
                          out %sink: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {

    // COMMON-NEXT: %inst_source_valid, %inst_source_ready, %inst_source_data, %inst_sink_valid, %inst_sink_ready, %inst_sink_data
    // COMMON-SAME: = instance "" @Simple(
    // COMMON-SAME: in source_valid: !firrtl.uint<1>, out source_ready: !firrtl.uint<1>, in source_data: !firrtl.uint<64>, out sink_valid: !firrtl.uint<1>, in sink_ready: !firrtl.uint<1>, out sink_data: !firrtl.uint<64>
    %sourceV, %sinkV = instance "" @Simple(in source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
                        out sink: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)

    // COMMON-NEXT: strictconnect %inst_source_valid, %source_valid
    // COMMON-NEXT: strictconnect %source_ready, %inst_source_ready
    // COMMON-NEXT: strictconnect %inst_source_data, %source_data
    // COMMON-NEXT: strictconnect %sink_valid, %inst_sink_valid
    // COMMON-NEXT: strictconnect %inst_sink_ready, %sink_ready
    // COMMON-NEXT: strictconnect %sink_data, %inst_sink_data
    connect %sourceV, %source : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>

    connect %sink, %sinkV : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
  }

  // COMMON-LABEL: module private @Recursive
  // CHECK-SAME: in %[[FLAT_ARG_1_NAME:arg_foo_bar_baz]]: [[FLAT_ARG_1_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: in %[[FLAT_ARG_2_NAME:arg_foo_qux]]: [[FLAT_ARG_2_TYPE:!firrtl.sint<64>]]
  // CHECK-SAME: out %[[OUT_1_NAME:out1]]: [[OUT_1_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: out %[[OUT_2_NAME:out2]]: [[OUT_2_TYPE:!firrtl.sint<64>]]
  // AGGREGATE-SAME: in %[[ARG_NAME:arg]]: [[ARG_TYPE:!firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>]]
  // AGGREGATE-SAME: out %[[OUT_1_NAME:out1]]: [[OUT_1_TYPE:!firrtl.uint<1>]]
  // AGGREGATE-SAME: out %[[OUT_2_NAME:out2]]: [[OUT_2_TYPE:!firrtl.sint<64>]]
  module private @Recursive(in %arg: !firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>,
                           out %out1: !firrtl.uint<1>, out %out2: !firrtl.sint<64>) {

    // CHECK-NEXT: connect %[[OUT_1_NAME]], %[[FLAT_ARG_1_NAME]] : [[OUT_1_TYPE]], [[FLAT_ARG_1_TYPE]]
    // CHECK-NEXT: connect %[[OUT_2_NAME]], %[[FLAT_ARG_2_NAME]] : [[OUT_2_TYPE]], [[FLAT_ARG_2_TYPE]]
    // AGGREGATE-NEXT:  %0 = subfield %[[ARG_NAME]][foo]
    // AGGREGATE-NEXT:  %1 = subfield %0[bar]
    // AGGREGATE-NEXT:  %2 = subfield %1[baz]
    // AGGREGATE-NEXT:  %3 = subfield %0[qux]
    // AGGREGATE-NEXT:  connect %[[OUT_1_NAME]], %2
    // AGGREGATE-NEXT:  connect %[[OUT_2_NAME]], %3

    %0 = subfield %arg[foo] : !firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>
    %1 = subfield %0[bar] : !firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>
    %2 = subfield %1[baz] : !firrtl.bundle<baz: uint<1>>
    %3 = subfield %0[qux] : !firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>
    connect %out1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %out2, %3 : !firrtl.sint<64>, !firrtl.sint<64>
  }

  // CHECK-LABEL: module private @Uniquification
  // CHECK-SAME: in %[[FLATTENED_ARG:a_b]]: [[FLATTENED_TYPE:!firrtl.uint<1>]],
  // CHECK-NOT: %[[FLATTENED_ARG]]
  // CHECK-SAME: in %[[RENAMED_ARG:a_b.+]]: [[RENAMED_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: {portNames = ["a_b", "a_b"]}
  module private @Uniquification(in %a: !firrtl.bundle<b: uint<1>>, in %a_b: !firrtl.uint<1>) {
  }

  // CHECK-LABEL: module private @Top
  module private @Top(in %in : !firrtl.bundle<a: uint<1>, b: uint<1>>,
                     out %out : !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    // CHECK: strictconnect %out_a, %in_a : !firrtl.uint<1>
    // CHECK: strictconnect %out_b, %in_b : !firrtl.uint<1>
    connect %out, %in : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
  }

  // CHECK-LABEL: module private @Foo
  // CHECK-SAME: in %[[FLAT_ARG_INPUT_NAME:a_b_c]]: [[FLAT_ARG_INPUT_TYPE:!firrtl.uint<1>]]
  // CHECK-SAME: out %[[FLAT_ARG_OUTPUT_NAME:b_b_c]]: [[FLAT_ARG_OUTPUT_TYPE:!firrtl.uint<1>]]
  module private @Foo(in %a: !firrtl.bundle<b: bundle<c: uint<1>>>, out %b: !firrtl.bundle<b: bundle<c: uint<1>>>) {
    // CHECK: strictconnect %[[FLAT_ARG_OUTPUT_NAME]], %[[FLAT_ARG_INPUT_NAME]] : [[FLAT_ARG_OUTPUT_TYPE]]
    connect %b, %a : !firrtl.bundle<b: bundle<c: uint<1>>>, !firrtl.bundle<b: bundle<c: uint<1>>>
  }

// Test lower of a 1-read 1-write aggregate memory
//
// circuit Foo :
//   module Foo :
//     input clock: Clock
//     input rAddr: UInt<4>
//     input rEn: UInt<1>
//     output rData: {a: UInt<8>, b: UInt<8>}
//     input wAddr: UInt<4>
//     input wEn: UInt<1>
//     input wMask: {a: UInt<1>, b: UInt<1>}
//     input wData: {a: UInt<8>, b: UInt<8>}
//
//     mem memory:
//       data-type => {a: UInt<8>, b: UInt<8>}
//       depth => 16
//       reader => r
//       writer => w
//       read-latency => 0
//       write-latency => 1
//       read-under-write => undefined
//
//     memory.r.clk <= clock
//     memory.r.en <= rEn
//     memory.r.addr <= rAddr
//     rData <= memory.r.data
//
//     memory.w.clk <= clock
//     memory.w.en <= wEn
//     memory.w.addr <= wAddr
//     memory.w.mask <= wMask
//     memory.w.data <= wData

  // CHECK-LABEL: module private @Mem2
  module private @Mem2(in %clock: !firrtl.clock, in %rAddr: !firrtl.uint<4>, in %rEn: !firrtl.uint<1>, out %rData: !firrtl.bundle<a: uint<8>, b: uint<8>>, in %wAddr: !firrtl.uint<4>, in %wEn: !firrtl.uint<1>, in %wMask: !firrtl.bundle<a: uint<1>, b: uint<1>>, in %wData: !firrtl.bundle<a: uint<8>, b: uint<8>>) {
    %memory_r, %memory_w = mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    %0 = subfield %memory_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    connect %0, %clock : !firrtl.clock, !firrtl.clock
    %1 = subfield %memory_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    connect %1, %rEn : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = subfield %memory_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    connect %2, %rAddr : !firrtl.uint<4>, !firrtl.uint<4>
    %3 = subfield %memory_r[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    connect %rData, %3 : !firrtl.bundle<a: uint<8>, b: uint<8>>, !firrtl.bundle<a: uint<8>, b: uint<8>>
    %4 = subfield %memory_w[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    connect %4, %clock : !firrtl.clock, !firrtl.clock
    %5 = subfield %memory_w[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    connect %5, %wEn : !firrtl.uint<1>, !firrtl.uint<1>
    %6 = subfield %memory_w[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    connect %6, %wAddr : !firrtl.uint<4>, !firrtl.uint<4>
    %7 = subfield %memory_w[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    connect %7, %wMask : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
    %8 = subfield %memory_w[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    connect %8, %wData : !firrtl.bundle<a: uint<8>, b: uint<8>>, !firrtl.bundle<a: uint<8>, b: uint<8>>

    // ---------------------------------------------------------------------------------
    // Split memory "a" should exist
    // CHECK: %[[MEMORY_A_R:.+]], %[[MEMORY_A_W:.+]] = mem {{.+}} data: uint<8>, mask: uint<1>
    //
    // Split memory "b" should exist
    // CHECK-NEXT: %[[MEMORY_B_R:.+]], %[[MEMORY_B_W:.+]] = mem {{.+}} data: uint<8>, mask: uint<1>
    // ---------------------------------------------------------------------------------
    // Read ports
    // CHECK-NEXT: %[[MEMORY_A_R_ADDR:.+]] = subfield %[[MEMORY_A_R]][addr]
    // CHECK-NEXT: strictconnect %[[MEMORY_A_R_ADDR]], %[[MEMORY_R_ADDR:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_R_ADDR:.+]] = subfield %[[MEMORY_B_R]][addr]
    // CHECK-NEXT: strictconnect %[[MEMORY_B_R_ADDR]], %[[MEMORY_R_ADDR]]
    // CHECK-NEXT: %[[MEMORY_A_R_EN:.+]] = subfield %[[MEMORY_A_R]][en]
    // CHECK-NEXT: strictconnect %[[MEMORY_A_R_EN]], %[[MEMORY_R_EN:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_R_EN:.+]] = subfield %[[MEMORY_B_R]][en]
    // CHECK-NEXT: strictconnect %[[MEMORY_B_R_EN]], %[[MEMORY_R_EN]]
    // CHECK-NEXT: %[[MEMORY_A_R_CLK:.+]] = subfield %[[MEMORY_A_R]][clk]
    // CHECK-NEXT: strictconnect %[[MEMORY_A_R_CLK]], %[[MEMORY_R_CLK:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_R_CLK:.+]] = subfield %[[MEMORY_B_R]][clk]
    // CHECK-NEXT: strictconnect %[[MEMORY_B_R_CLK]], %[[MEMORY_R_CLK]]
    // CHECK-NEXT: %[[MEMORY_A_R_DATA:.+]] = subfield %[[MEMORY_A_R]][data]
    // CHECK-NEXT: strictconnect %[[WIRE_A_R_DATA:.+]], %[[MEMORY_A_R_DATA]] :
    // CHECK-NEXT: %[[MEMORY_B_R_DATA:.+]] = subfield %[[MEMORY_B_R]][data]
    // CHECK-NEXT: strictconnect %[[WIRE_B_R_DATA:.+]], %[[MEMORY_B_R_DATA]] :
    // ---------------------------------------------------------------------------------
    // Write Ports
    // CHECK-NEXT: %[[MEMORY_A_W_ADDR:.+]] = subfield %[[MEMORY_A_W]][addr]
    // CHECK-NEXT: strictconnect %[[MEMORY_A_W_ADDR]], %[[MEMORY_W_ADDR:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_W_ADDR:.+]] = subfield %[[MEMORY_B_W]][addr]
    // CHECK-NEXT: strictconnect %[[MEMORY_B_W_ADDR]], %[[MEMORY_W_ADDR]] :
    // CHECK-NEXT: %[[MEMORY_A_W_EN:.+]] = subfield %[[MEMORY_A_W]][en]
    // CHECK-NEXT: strictconnect %[[MEMORY_A_W_EN]], %[[MEMORY_W_EN:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_W_EN:.+]] = subfield %[[MEMORY_B_W]][en]
    // CHECK-NEXT: strictconnect %[[MEMORY_B_W_EN]], %[[MEMORY_W_EN]] :
    // CHECK-NEXT: %[[MEMORY_A_W_CLK:.+]] = subfield %[[MEMORY_A_W]][clk]
    // CHECK-NEXT: strictconnect %[[MEMORY_A_W_CLK]], %[[MEMORY_W_CLK:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_W_CLK:.+]] = subfield %[[MEMORY_B_W]][clk]
    // CHECK-NEXT: strictconnect %[[MEMORY_B_W_CLK]], %[[MEMORY_W_CLK]] :
    // CHECK-NEXT: %[[MEMORY_A_W_DATA:.+]] = subfield %[[MEMORY_A_W]][data]
    // CHECK-NEXT: strictconnect %[[MEMORY_A_W_DATA]], %[[WIRE_A_W_DATA:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_W_DATA:.+]] = subfield %[[MEMORY_B_W]][data]
    // CHECK-NEXT: strictconnect %[[MEMORY_B_W_DATA]], %[[WIRE_B_W_DATA:.+]] :
    // CHECK-NEXT: %[[MEMORY_A_W_MASK:.+]] = subfield %[[MEMORY_A_W]][mask]
    // CHECK-NEXT: strictconnect %[[MEMORY_A_W_MASK]], %[[WIRE_A_W_MASK:.+]] :
    // CHECK-NEXT: %[[MEMORY_B_W_MASK:.+]] = subfield %[[MEMORY_B_W]][mask]
    // CHECK-NEXT: strictconnect %[[MEMORY_B_W_MASK]], %[[WIRE_B_W_MASK:.+]] :
    //
    // Connections to module ports
    // CHECK-NEXT: connect %[[MEMORY_R_CLK]], %clock
    // CHECK-NEXT: connect %[[MEMORY_R_EN]], %rEn
    // CHECK-NEXT: connect %[[MEMORY_R_ADDR]], %rAddr
    // CHECK-NEXT: strictconnect %rData_a, %[[WIRE_A_R_DATA]]
    // CHECK-NEXT: strictconnect %rData_b, %[[WIRE_B_R_DATA]]
    // CHECK-NEXT: connect %[[MEMORY_W_CLK]], %clock
    // CHECK-NEXT: connect %[[MEMORY_W_EN]], %wEn
    // CHECK-NEXT: connect %[[MEMORY_W_ADDR]], %wAddr
    // CHECK-NEXT: strictconnect %[[WIRE_A_W_MASK]], %wMask_a
    // CHECK-NEXT: strictconnect %[[WIRE_B_W_MASK]], %wMask_b
    // CHECK-NEXT: strictconnect %[[WIRE_A_W_DATA]], %wData_a
    // CHECK-NEXT: strictconnect %[[WIRE_B_W_DATA]], %wData_b
  }


// https://github.com/llvm/circt/issues/593

    module private @mod_2(in %clock: !firrtl.clock, in %inp_a: !firrtl.bundle<inp_d: uint<14>>) {
    }
    module private @top_mod(in %clock: !firrtl.clock) {
      %U0_clock, %U0_inp_a = instance U0 @mod_2(in clock: !firrtl.clock, in inp_a: !firrtl.bundle<inp_d: uint<14>>)
      %0 = invalidvalue : !firrtl.clock
      connect %U0_clock, %0 : !firrtl.clock, !firrtl.clock
      %1 = invalidvalue : !firrtl.bundle<inp_d: uint<14>>
      connect %U0_inp_a, %1 : !firrtl.bundle<inp_d: uint<14>>, !firrtl.bundle<inp_d: uint<14>>
    }



//CHECK-LABEL:     module private @mod_2(in %clock: !firrtl.clock, in %inp_a_inp_d: !firrtl.uint<14>)
//CHECK:    module private @top_mod(in %clock: !firrtl.clock)
//CHECK-NEXT:      %U0_clock, %U0_inp_a_inp_d = instance U0 @mod_2(in clock: !firrtl.clock, in inp_a_inp_d: !firrtl.uint<14>)
//CHECK-NEXT:      %invalid_clock = invalidvalue : !firrtl.clock
//CHECK-NEXT:      connect %U0_clock, %invalid_clock : !firrtl.clock, !firrtl.clock
//CHECK-NEXT:      %invalid_ui14 = invalidvalue : !firrtl.uint<14>
//CHECK-NEXT:      strictconnect %U0_inp_a_inp_d, %invalid_ui14 : !firrtl.uint<14>

//AGGREGATE-LABEL: module private @mod_2(in %clock: !firrtl.clock, in %inp_a: !firrtl.bundle<inp_d: uint<14>>)
//AGGREGATE:    module private @top_mod(in %clock: !firrtl.clock)
//AGGREGATE-NEXT:  %U0_clock, %U0_inp_a = instance U0  @mod_2(in clock: !firrtl.clock, in inp_a: !firrtl.bundle<inp_d: uint<14>>)
//AGGREGATE-NEXT:  %invalid_clock = invalidvalue : !firrtl.clock
//AGGREGATE-NEXT:  connect %U0_clock, %invalid_clock : !firrtl.clock, !firrtl.clock
//AGGREGATE-NEXT:  %invalid = invalidvalue : !firrtl.bundle<inp_d: uint<14>>
//AGGREGATE-NEXT:  %0 = subfield %invalid[inp_d] : !firrtl.bundle<inp_d: uint<14>>
//AGGREGATE-NEXT:  %1 = subfield %U0_inp_a[inp_d] : !firrtl.bundle<inp_d: uint<14>>
//AGGREGATE-NEXT:  strictconnect %1, %0 : !firrtl.uint<14>
// https://github.com/llvm/circt/issues/661

// This test is just checking that the following doesn't error.
    // COMMON-LABEL: module private @Issue661
    module private @Issue661(in %clock: !firrtl.clock) {
      %head_MPORT_2, %head_MPORT_6 = mem Undefined {depth = 20 : i64, name = "head", portNames = ["MPORT_2", "MPORT_6"], readLatency = 0 : i32, writeLatency = 1 : i32}
      : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>,
        !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>
      %127 = subfield %head_MPORT_6[clk] : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>
    }

// Check that a non-bundled mux ops are untouched.
    // CHECK-LABEL: module private @Mux
    module private @Mux(in %p: !firrtl.uint<1>, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
      // CHECK-NEXT: %0 = mux(%p, %a, %b) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      // CHECK-NEXT: connect %c, %0 : !firrtl.uint<1>, !firrtl.uint<1>
      %0 = mux(%p, %a, %b) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      connect %c, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    }
    // CHECK-LABEL: module private @MuxBundle
    module private @MuxBundle(in %p: !firrtl.uint<1>, in %a: !firrtl.bundle<a: uint<1>>, in %b: !firrtl.bundle<a: uint<1>>, out %c: !firrtl.bundle<a: uint<1>>) {
      // CHECK-NEXT: %0 = mux(%p, %a_a, %b_a) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      // CHECK-NEXT: strictconnect %c_a, %0 : !firrtl.uint<1>
      %0 = mux(%p, %a, %b) : (!firrtl.uint<1>, !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>) -> !firrtl.bundle<a: uint<1>>
      connect %c, %0 : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    }

    // CHECK-LABEL: module private @NodeBundle
    module private @NodeBundle(in %a: !firrtl.bundle<a: uint<1>>, out %b: !firrtl.uint<1>) {
      // CHECK-NEXT: %n_a = node %a_a  : !firrtl.uint<1>
      // CHECK-NEXT: connect %b, %n_a : !firrtl.uint<1>, !firrtl.uint<1>
      %n = node %a : !firrtl.bundle<a: uint<1>>
      %n_a = subfield %n[a] : !firrtl.bundle<a: uint<1>>
      connect %b, %n_a : !firrtl.uint<1>, !firrtl.uint<1>
    }

    // CHECK-LABEL: module private @RegBundle(in %a_a: !firrtl.uint<1>, in %clk: !firrtl.clock, out %b_a: !firrtl.uint<1>)
    module private @RegBundle(in %a: !firrtl.bundle<a: uint<1>>, in %clk: !firrtl.clock, out %b: !firrtl.bundle<a: uint<1>>) {
      // CHECK-NEXT: %x_a = reg %clk : !firrtl.clock, !firrtl.uint<1>
      // CHECK-NEXT: connect %x_a, %a_a : !firrtl.uint<1>, !firrtl.uint<1>
      // CHECK-NEXT: connect %b_a, %x_a : !firrtl.uint<1>, !firrtl.uint<1>
      %x = reg %clk {name = "x"} : !firrtl.clock, !firrtl.bundle<a: uint<1>>
      %0 = subfield %x[a] : !firrtl.bundle<a: uint<1>>
      %1 = subfield %a[a] : !firrtl.bundle<a: uint<1>>
      connect %0, %1 : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = subfield %b[a] : !firrtl.bundle<a: uint<1>>
      %3 = subfield %x[a] : !firrtl.bundle<a: uint<1>>
      connect %2, %3 : !firrtl.uint<1>, !firrtl.uint<1>
    }

    // CHECK-LABEL: module private @RegBundleWithBulkConnect(in %a_a: !firrtl.uint<1>, in %clk: !firrtl.clock, out %b_a: !firrtl.uint<1>)
    module private @RegBundleWithBulkConnect(in %a: !firrtl.bundle<a: uint<1>>, in %clk: !firrtl.clock, out %b: !firrtl.bundle<a: uint<1>>) {
      // CHECK-NEXT: %x_a = reg %clk : !firrtl.clock, !firrtl.uint<1>
      // CHECK-NEXT: strictconnect %x_a, %a_a : !firrtl.uint<1>
      // CHECK-NEXT: strictconnect %b_a, %x_a : !firrtl.uint<1>
      %x = reg %clk {name = "x"} : !firrtl.clock, !firrtl.bundle<a: uint<1>>
      connect %x, %a : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
      connect %b, %x : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    }

    // CHECK-LABEL: module private @WireBundle(in %a_a: !firrtl.uint<1>,  out %b_a: !firrtl.uint<1>)
    module private @WireBundle(in %a: !firrtl.bundle<a: uint<1>>,  out %b: !firrtl.bundle<a: uint<1>>) {
      // CHECK-NEXT: %x_a = wire  : !firrtl.uint<1>
      // CHECK-NEXT: connect %x_a, %a_a : !firrtl.uint<1>, !firrtl.uint<1>
      // CHECK-NEXT: connect %b_a, %x_a : !firrtl.uint<1>, !firrtl.uint<1>
      %x = wire : !firrtl.bundle<a: uint<1>>
      %0 = subfield %x[a] : !firrtl.bundle<a: uint<1>>
      %1 = subfield %a[a] : !firrtl.bundle<a: uint<1>>
      connect %0, %1 : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = subfield %b[a] : !firrtl.bundle<a: uint<1>>
      %3 = subfield %x[a] : !firrtl.bundle<a: uint<1>>
      connect %2, %3 : !firrtl.uint<1>, !firrtl.uint<1>
    }

  // CHECK-LABEL: module private @WireBundlesWithBulkConnect
  module private @WireBundlesWithBulkConnect(in %source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
                             out %sink: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
    // CHECK: %w_valid = wire  : !firrtl.uint<1>
    // CHECK: %w_ready = wire  : !firrtl.uint<1>
    // CHECK: %w_data = wire  : !firrtl.uint<64>
    %w = wire : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    // CHECK: strictconnect %w_valid, %source_valid : !firrtl.uint<1>
    // CHECK: strictconnect %source_ready, %w_ready : !firrtl.uint<1>
    // CHECK: strictconnect %w_data, %source_data : !firrtl.uint<64>
    connect %w, %source : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
    // CHECK: strictconnect %sink_valid, %w_valid : !firrtl.uint<1>
    // CHECK: strictconnect %w_ready, %sink_ready : !firrtl.uint<1>
    // CHECK: strictconnect %sink_data, %w_data : !firrtl.uint<64>
    connect %sink, %w : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
  }

// Test vector lowering
  module private @LowerVectors(in %a: !firrtl.vector<uint<1>, 2>, out %b: !firrtl.vector<uint<1>, 2>) {
    connect %b, %a: !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
  // CHECK-LABEL: module private @LowerVectors(in %a_0: !firrtl.uint<1>, in %a_1: !firrtl.uint<1>, out %b_0: !firrtl.uint<1>, out %b_1: !firrtl.uint<1>)
  // CHECK: strictconnect %b_0, %a_0
  // CHECK: strictconnect %b_1, %a_1
  // AGGREGATE-LABEL: module private @LowerVectors(in %a: !firrtl.vector<uint<1>, 2>, out %b: !firrtl.vector<uint<1>, 2>)
  // AGGREGATE-NEXT: %0 = subindex %a[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT: %1 = subindex %b[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT: strictconnect %1, %0 : !firrtl.uint<1>
  // AGGREGATE-NEXT: %2 = subindex %a[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT: %3 = subindex %b[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT: strictconnect %3, %2 : !firrtl.uint<1>

// Test vector of bundles lowering
  // COMMON-LABEL: module private @LowerVectorsOfBundles(in %in_0_a: !firrtl.uint<1>, out %in_0_b: !firrtl.uint<1>, in %in_1_a: !firrtl.uint<1>, out %in_1_b: !firrtl.uint<1>, out %out_0_a: !firrtl.uint<1>, in %out_0_b: !firrtl.uint<1>, out %out_1_a: !firrtl.uint<1>, in %out_1_b: !firrtl.uint<1>)
  module private @LowerVectorsOfBundles(in %in: !firrtl.vector<bundle<a : uint<1>, b  flip: uint<1>>, 2>,
                                       out %out: !firrtl.vector<bundle<a : uint<1>, b  flip: uint<1>>, 2>) {
    // COMMON:      strictconnect %out_0_a, %in_0_a : !firrtl.uint<1>
    // COMMON-NEXT: strictconnect %in_0_b, %out_0_b : !firrtl.uint<1>
    // COMMON-NEXT: strictconnect %out_1_a, %in_1_a : !firrtl.uint<1>
    // COMMON-NEXT: strictconnect %in_1_b, %out_1_b : !firrtl.uint<1>
    connect %out, %in: !firrtl.vector<bundle<a : uint<1>, b flip: uint<1>>, 2>, !firrtl.vector<bundle<a : uint<1>, b flip: uint<1>>, 2>
  }

  // COMMON-LABEL: extmodule @ExternalModule(in source_valid: !firrtl.uint<1>, out source_ready: !firrtl.uint<1>, in source_data: !firrtl.uint<64>)
  extmodule @ExternalModule(in source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
  module private @Test() {
    // COMMON: %inst_source_valid, %inst_source_ready, %inst_source_data = instance "" @ExternalModule(in source_valid: !firrtl.uint<1>, out source_ready: !firrtl.uint<1>, in source_data: !firrtl.uint<64>)
    %inst_source = instance "" @ExternalModule(in source: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
  }

// Test RegResetOp lowering
  // CHECK-LABEL: module private @LowerRegResetOp
  module private @LowerRegResetOp(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a_d: !firrtl.vector<uint<1>, 2>, out %a_q: !firrtl.vector<uint<1>, 2>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %init = wire  : !firrtl.vector<uint<1>, 2>
    %0 = subindex %init[0] : !firrtl.vector<uint<1>, 2>
    connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = subindex %init[1] : !firrtl.vector<uint<1>, 2>
    connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %r = regreset %clock, %reset, %init {name = "r"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    connect %r, %a_d : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    connect %a_q, %r : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
  // CHECK:   %c0_ui1 = constant 0 : !firrtl.uint<1>
  // CHECK:   %init_0 = wire  : !firrtl.uint<1>
  // CHECK:   %init_1 = wire  : !firrtl.uint<1>
  // CHECK:   connect %init_0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   connect %init_1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   %r_0 = regreset %clock, %reset, %init_0 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   %r_1 = regreset %clock, %reset, %init_1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   strictconnect %r_0, %a_d_0 : !firrtl.uint<1>
  // CHECK:   strictconnect %r_1, %a_d_1 : !firrtl.uint<1>
  // CHECK:   strictconnect %a_q_0, %r_0 : !firrtl.uint<1>
  // CHECK:   strictconnect %a_q_1, %r_1 : !firrtl.uint<1>
  // AGGREGATE:       %c0_ui1 = constant 0 : !firrtl.uint<1>
  // AGGREGATE-NEXT:  %init = wire  : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %0 = subindex %init[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // AGGREGATE-NEXT:  %1 = subindex %init[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // AGGREGATE-NEXT:  %r = regreset %clock, %reset, %init  : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %2 = subindex %a_d[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %3 = subindex %r[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  strictconnect %3, %2 : !firrtl.uint<1>
  // AGGREGATE-NEXT:  %4 = subindex %a_d[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %5 = subindex %r[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  strictconnect %5, %4 : !firrtl.uint<1>
  // AGGREGATE-NEXT:  %6 = subindex %r[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %7 = subindex %a_q[0] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  strictconnect %7, %6 : !firrtl.uint<1>
  // AGGREGATE-NEXT:  %8 = subindex %r[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  %9 = subindex %a_q[1] : !firrtl.vector<uint<1>, 2>
  // AGGREGATE-NEXT:  strictconnect %9, %8 : !firrtl.uint<1>

// Test RegResetOp lowering without name attribute
// https://github.com/llvm/circt/issues/795
  // CHECK-LABEL: module private @LowerRegResetOpNoName
  module private @LowerRegResetOpNoName(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a_d: !firrtl.vector<uint<1>, 2>, out %a_q: !firrtl.vector<uint<1>, 2>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %init = wire  : !firrtl.vector<uint<1>, 2>
    %0 = subindex %init[0] : !firrtl.vector<uint<1>, 2>
    connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = subindex %init[1] : !firrtl.vector<uint<1>, 2>
    connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %r = regreset %clock, %reset, %init {name = ""} : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    connect %r, %a_d : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    connect %a_q, %r : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
  // CHECK:   %c0_ui1 = constant 0 : !firrtl.uint<1>
  // CHECK:   %init_0 = wire  : !firrtl.uint<1>
  // CHECK:   %init_1 = wire  : !firrtl.uint<1>
  // CHECK:   connect %init_0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   connect %init_1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   %0 = regreset %clock, %reset, %init_0 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   %1 = regreset %clock, %reset, %init_1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:   strictconnect %0, %a_d_0 : !firrtl.uint<1>
  // CHECK:   strictconnect %1, %a_d_1 : !firrtl.uint<1>
  // CHECK:   strictconnect %a_q_0, %0 : !firrtl.uint<1>
  // CHECK:   strictconnect %a_q_1, %1 : !firrtl.uint<1>

// Test RegOp lowering without name attribute
// https://github.com/llvm/circt/issues/795
  // CHECK-LABEL: module private @lowerRegOpNoName
  module private @lowerRegOpNoName(in %clock: !firrtl.clock, in %a_d: !firrtl.vector<uint<1>, 2>, out %a_q: !firrtl.vector<uint<1>, 2>) {
    %r = reg %clock {name = ""} : !firrtl.clock, !firrtl.vector<uint<1>, 2>
      connect %r, %a_d : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
      connect %a_q, %r : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
 // CHECK:    %0 = reg %clock : !firrtl.clock, !firrtl.uint<1>
 // CHECK:    %1 = reg %clock : !firrtl.clock, !firrtl.uint<1>
 // CHECK:    strictconnect %0, %a_d_0 : !firrtl.uint<1>
 // CHECK:    strictconnect %1, %a_d_1 : !firrtl.uint<1>
 // CHECK:    strictconnect %a_q_0, %0 : !firrtl.uint<1>
 // CHECK:    strictconnect %a_q_1, %1 : !firrtl.uint<1>

// Test that InstanceOp Annotations are copied to the new instance.
  module private @Bar(out %a: !firrtl.vector<uint<1>, 2>) {
    %0 = invalidvalue : !firrtl.vector<uint<1>, 2>
    connect %a, %0 : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
  module private @AnnotationsInstanceOp() {
    %bar_a = instance bar {annotations = [{a = "a"}]} @Bar(out a: !firrtl.vector<uint<1>, 2>)
  }
  // CHECK: instance
  // CHECK-SAME: annotations = [{a = "a"}]

// Test that MemOp Annotations are copied to lowered MemOps.
  // COMMON-LABEL: module private @AnnotationsMemOp
  module private @AnnotationsMemOp() {
    %bar_r, %bar_w = mem Undefined  {annotations = [{a = "a"}], depth = 16 : i64, name = "bar", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 2>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: vector<uint<8>, 2>, mask: vector<uint<1>, 2>>
  }
  // COMMON: mem
  // COMMON-SAME: annotations = [{a = "a"}]
  // COMMON: mem
  // COMMON-SAME: annotations = [{a = "a"}]

// Test that WireOp Annotations are copied to lowered WireOps.
  // CHECK-LABEL: module private @AnnotationsWireOp
  module private @AnnotationsWireOp() {
    %bar = wire  {annotations = [{a = "a"}]} : !firrtl.vector<uint<1>, 2>
  }
  // CHECK: wire
  // CHECK-SAME: annotations = [{a = "a"}]
  // CHECK: wire
  // CHECK-SAME: annotations = [{a = "a"}]

// Test that WireOp annotations which are sensitive to field IDs are annotated
// with the lowered field IDs.
  // COMMON-LABEL: module private @AnnotationsWithFieldIdWireOp
  module private @AnnotationsWithFieldIdWireOp() {
    %foo = wire {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation"}]} : !firrtl.uint<1>
    %bar = wire {annotations = [{class = "sifive.enterprise.grandcentral.SignalDriverAnnotation"}]} : !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<1>>
    %baz = wire {annotations = [{circt.fieldID = 2 : i32, class = "sifive.enterprise.grandcentral.SignalDriverAnnotation"}]} : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
  }
  // CHECK: %foo = wire
  // CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation"}
  // CHECK: %bar_a_0 = wire
  // CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", fieldID = 2 : i64}
  // CHECK: %bar_a_1 = wire
  // CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", fieldID = 3 : i64}
  // CHECK: %bar_b = wire
  // CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", fieldID = 4 : i64}
  // CHECK: %baz_a = wire
  // CHECK-NOT:  {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation"}
  // CHECK: %baz_b_0 = wire
  // CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", fieldID = 1 : i64}
  // CHECK: %baz_b_1 = wire
  // CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", fieldID = 2 : i64}
  // AGGREGATE:  %baz = wire
  // AGGREGATE-SAME: {annotations = [{circt.fieldID = 2 : i32, class = "sifive.enterprise.grandcentral.SignalDriverAnnotation"}]}

// Test that Reg/RegResetOp Annotations are copied to lowered registers.
  // CHECK-LABEL: module private @AnnotationsRegOp
  module private @AnnotationsRegOp(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %bazInit = wire  : !firrtl.vector<uint<1>, 2>
    %0 = subindex %bazInit[0] : !firrtl.vector<uint<1>, 2>
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = subindex %bazInit[1] : !firrtl.vector<uint<1>, 2>
    connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %bar = reg %clock  {annotations = [{a = "a"}], name = "bar"} : !firrtl.clock, !firrtl.vector<uint<1>, 2>
    %baz = regreset %clock, %reset, %bazInit  {annotations = [{b = "b"}], name = "baz"} : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
  // CHECK: reg
  // CHECK-SAME: annotations = [{a = "a"}]
  // CHECK: reg
  // CHECK-SAME: annotations = [{a = "a"}]
  // CHECK: regreset
  // CHECK-SAME: annotations = [{b = "b"}]
  // CHECK: regreset
  // CHECK-SAME: annotations = [{b = "b"}]

// Test that WhenOp with regions has its regions lowered.
// CHECK-LABEL: module private @WhenOp
  module private @WhenOp (in %p: !firrtl.uint<1>,
                         in %in : !firrtl.bundle<a: uint<1>, b: uint<1>>,
                         out %out : !firrtl.bundle<a: uint<1>, b: uint<1>>) {
    // No else region.
    when %p : !firrtl.uint<1> {
      // CHECK: strictconnect %out_a, %in_a : !firrtl.uint<1>
      // CHECK: strictconnect %out_b, %in_b : !firrtl.uint<1>
      connect %out, %in : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
    }

    // Else region.
    when %p : !firrtl.uint<1> {
      // CHECK: strictconnect %out_a, %in_a : !firrtl.uint<1>
      // CHECK: strictconnect %out_b, %in_b : !firrtl.uint<1>
      connect %out, %in : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
    } else {
      // CHECK: strictconnect %out_a, %in_a : !firrtl.uint<1>
      // CHECK: strictconnect %out_b, %in_b : !firrtl.uint<1>
      connect %out, %in : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
    }
  }

// Test that subfield annotations on wire are lowred to appropriate instance based on fieldID.
  // CHECK-LABEL: module private @AnnotationsBundle
  module private @AnnotationsBundle() {
    %bar = wire  {annotations = [
      {circt.fieldID = 3, one},
      {circt.fieldID = 5, two}
    ]} : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>

      // TODO: Enable this
      // CHECK: %bar_0_baz = wire  : !firrtl.uint<1>
      // CHECK: %bar_0_qux = wire {annotations = [{one}]} : !firrtl.uint<1>
      // CHECK: %bar_1_baz = wire {annotations = [{two}]} : !firrtl.uint<1>
      // CHECK: %bar_1_qux = wire  : !firrtl.uint<1>

    %quux = wire  {annotations = [
      {circt.fieldID = 0, zero}
    ]} : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
      // CHECK: %quux_0_baz = wire {annotations = [{zero}]} : !firrtl.uint<1>
      // CHECK: %quux_0_qux = wire {annotations = [{zero}]} : !firrtl.uint<1>
      // CHECK: %quux_1_baz = wire {annotations = [{zero}]} : !firrtl.uint<1>
      // CHECK: %quux_1_qux = wire {annotations = [{zero}]} : !firrtl.uint<1>
  }

// Test that subfield annotations on reg are lowred to appropriate instance based on fieldID.
 // CHECK-LABEL: module private @AnnotationsBundle2
  module private @AnnotationsBundle2(in %clock: !firrtl.clock) {
    %bar = reg %clock  {annotations = [
      {circt.fieldID = 3, one},
      {circt.fieldID = 5, two}
    ]} : !firrtl.clock, !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>

    // TODO: Enable this
    // CHECK: %bar_0_baz = reg %clock  : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_0_qux = reg %clock  {annotations = [{one}]} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_1_baz = reg %clock  {annotations = [{two}]} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_1_qux = reg %clock  : !firrtl.clock, !firrtl.uint<1>
  }

// Test that subfield annotations on reg are lowred to appropriate instance based on fieldID. Ignore un-flattened array targets
// circuit Foo: %[[{"one":null,"target":"~Foo|Foo>bar[0].qux[0]"},{"two":null,"target":"~Foo|Foo>bar[1].baz"},{"three":null,"target":"~Foo|Foo>bar[0].yes"} ]]

 // CHECK-LABEL: module private @AnnotationsBundle3
  module private @AnnotationsBundle3(in %clock: !firrtl.clock) {
    %bar = reg %clock  {
      annotations = [
        {circt.fieldID = 6, one},
        {circt.fieldID = 12, two},
        {circt.fieldID = 8, three}
      ]} : !firrtl.clock, !firrtl.vector<bundle<baz: vector<uint<1>, 2>, qux: vector<uint<1>, 2>, yes: bundle<a: uint<1>, b: uint<1>>>, 2>

    // TODO: Enable this
    // CHECK: %bar_0_baz_0 = reg %clock  : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_0_baz_1 = reg %clock  : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_0_qux_0 = reg %clock  {annotations = [{one}]} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_0_qux_1 = reg %clock  : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_0_yes_a = reg %clock  {annotations = [{three}]} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_0_yes_b = reg %clock  {annotations = [{three}]} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_1_baz_0 = reg %clock  {annotations = [{two}]} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_1_baz_1 = reg %clock  {annotations = [{two}]} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_1_qux_0 = reg %clock  : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_1_qux_1 = reg %clock  : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_1_yes_a = reg %clock  : !firrtl.clock, !firrtl.uint<1>
    // CHECK: %bar_1_yes_b = reg %clock  : !firrtl.clock, !firrtl.uint<1>
  }

// Test wire connection semantics.  Based on the flippedness of the destination
// type, the connection may be reversed.
// CHECK-LABEL module private @WireSemantics
  module private @WireSemantics() {
    %a = wire  : !firrtl.bundle<a: bundle<a: uint<1>>>
    %ax = wire  : !firrtl.bundle<a: bundle<a: uint<1>>>
    // CHECK:  %a_a_a = wire
    // CHECK-NEXT:  %ax_a_a = wire
    connect %a, %ax : !firrtl.bundle<a: bundle<a: uint<1>>>, !firrtl.bundle<a: bundle<a: uint<1>>>
    // a <= ax
    // CHECK-NEXT: strictconnect %a_a_a, %ax_a_a : !firrtl.uint<1>
    %0 = subfield %a[a] : !firrtl.bundle<a: bundle<a: uint<1>>>
    %1 = subfield %ax[a] : !firrtl.bundle<a: bundle<a: uint<1>>>
    connect %0, %1 : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    // a.a <= ax.a
    // CHECK: strictconnect %a_a_a, %ax_a_a : !firrtl.uint<1>
    %2 = subfield %a[a] : !firrtl.bundle<a: bundle<a: uint<1>>>
    %3 = subfield %2[a] : !firrtl.bundle<a: uint<1>>
    %4 = subfield %ax[a] : !firrtl.bundle<a: bundle<a: uint<1>>>
    %5 = subfield %4[a] : !firrtl.bundle<a: uint<1>>
    connect %3, %5 : !firrtl.uint<1>, !firrtl.uint<1>
    // a.a.a <= ax.a.a
    // CHECK: connect %a_a_a, %ax_a_a
    %b = wire  : !firrtl.bundle<a: bundle<a flip: uint<1>>>
    %bx = wire  : !firrtl.bundle<a: bundle<a flip: uint<1>>>
    // CHECK %b_a_a = wire
    // CHECK %bx_a_a = wire
    connect %b, %bx : !firrtl.bundle<a: bundle<a flip: uint<1>>>, !firrtl.bundle<a: bundle<a flip: uint<1>>>
    // b <= bx
    // CHECK: strictconnect %bx_a_a, %b_a_a
    %6 = subfield %b[a] : !firrtl.bundle<a: bundle<a flip: uint<1>>>
    %7 = subfield %bx[a] : !firrtl.bundle<a: bundle<a flip: uint<1>>>
    connect %6, %7 : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
    // b.a <= bx.a
    // CHECK: strictconnect %bx_a_a, %b_a_a
    %8 = subfield %b[a] : !firrtl.bundle<a: bundle<a flip: uint<1>>>
    %9 = subfield %8[a] : !firrtl.bundle<a flip: uint<1>>
    %10 = subfield %bx[a] : !firrtl.bundle<a: bundle<a flip: uint<1>>>
    %11 = subfield %10[a] : !firrtl.bundle<a flip: uint<1>>
    connect %9, %11 : !firrtl.uint<1>, !firrtl.uint<1>
    // b.a.a <= bx.a.a
    // CHECK: connect %b_a_a, %bx_a_a
    %c = wire  : !firrtl.bundle<a flip: bundle<a: uint<1>>>
    %cx = wire  : !firrtl.bundle<a flip: bundle<a: uint<1>>>
    // CHECK: %c_a_a = wire : !firrtl.uint<1>
    // CHECK-NEXT: %cx_a_a = wire : !firrtl.uint<1>
    connect %c, %cx : !firrtl.bundle<a flip: bundle<a: uint<1>>>, !firrtl.bundle<a flip: bundle<a: uint<1>>>
    // c <= cx
    // CHECK: strictconnect %cx_a_a, %c_a_a
    %12 = subfield %c[a] : !firrtl.bundle<a flip: bundle<a: uint<1>>>
    %13 = subfield %cx[a] : !firrtl.bundle<a flip: bundle<a: uint<1>>>
    connect %12, %13 : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    // c.a <= cx.a
    // CHECK: strictconnect %c_a_a, %cx_a_a
    %14 = subfield %c[a] : !firrtl.bundle<a flip: bundle<a: uint<1>>>
    %15 = subfield %14[a] : !firrtl.bundle<a: uint<1>>
    %16 = subfield %cx[a] : !firrtl.bundle<a flip: bundle<a: uint<1>>>
    %17 = subfield %16[a] : !firrtl.bundle<a: uint<1>>
    connect %15, %17 : !firrtl.uint<1>, !firrtl.uint<1>
    // c.a.a <= cx.a.a
    // CHECK: connect %c_a_a, %cx_a_a
    %d = wire  : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    %dx = wire  : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    // CHECK: %d_a_a = wire : !firrtl.uint<1>
    // CHECK-NEXT: %dx_a_a = wire : !firrtl.uint<1>
    connect %d, %dx : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>, !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    // d <= dx
    // CHECK: strictconnect %d_a_a, %dx_a_a
    %18 = subfield %d[a] : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    %19 = subfield %dx[a] : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    connect %18, %19 : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
    // d.a <= dx.a
    // CHECK: strictconnect %dx_a_a, %d_a_a
    %20 = subfield %d[a] : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    %21 = subfield %20[a] : !firrtl.bundle<a flip: uint<1>>
    %22 = subfield %dx[a] : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
    %23 = subfield %22[a] : !firrtl.bundle<a flip: uint<1>>
    connect %21, %23 : !firrtl.uint<1>, !firrtl.uint<1>
    // d.a.a <= dx.a.a
    // CHECK: connect %d_a_a, %dx_a_a
  }

// Test that a vector of bundles with a write works.
 // CHECK-LABEL: module private @aofs
    module private @aofs(in %a: !firrtl.uint<1>, in %sel: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>>, 4>) {
      %0 = subindex %b[0] : !firrtl.vector<bundle<wo: uint<1>>, 4>
      %1 = subfield %0[wo] : !firrtl.bundle<wo: uint<1>>
      %invalid_ui1 = invalidvalue : !firrtl.uint<1>
      connect %1, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = subindex %b[1] : !firrtl.vector<bundle<wo: uint<1>>, 4>
      %3 = subfield %2[wo] : !firrtl.bundle<wo: uint<1>>
      %invalid_ui1_0 = invalidvalue : !firrtl.uint<1>
      connect %3, %invalid_ui1_0 : !firrtl.uint<1>, !firrtl.uint<1>
      %4 = subindex %b[2] : !firrtl.vector<bundle<wo: uint<1>>, 4>
      %5 = subfield %4[wo] : !firrtl.bundle<wo: uint<1>>
      %invalid_ui1_1 = invalidvalue : !firrtl.uint<1>
      connect %5, %invalid_ui1_1 : !firrtl.uint<1>, !firrtl.uint<1>
      %6 = subindex %b[3] : !firrtl.vector<bundle<wo: uint<1>>, 4>
      %7 = subfield %6[wo] : !firrtl.bundle<wo: uint<1>>
      %invalid_ui1_2 = invalidvalue : !firrtl.uint<1>
      connect %7, %invalid_ui1_2 : !firrtl.uint<1>, !firrtl.uint<1>
      %8 = subaccess %b[%sel] : !firrtl.vector<bundle<wo: uint<1>>, 4>, !firrtl.uint<2>
      %9 = subfield %8[wo] : !firrtl.bundle<wo: uint<1>>
      connect %9, %a : !firrtl.uint<1>, !firrtl.uint<1>
    }


// Test that annotations on aggregate ports are copied.
  extmodule @Sub1(in a: !firrtl.vector<uint<1>, 2> [{a}])
  // CHECK-LABEL: extmodule @Sub1
  // CHECK-COUNT-2: [{b}]
  // CHECK-NOT: [{a}]
  module private @Port(in %a: !firrtl.vector<uint<1>, 2> [{b}]) {
    %sub_a = instance sub @Sub1(in a: !firrtl.vector<uint<1>, 2>)
    connect %sub_a, %a : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }

// Test that annotations on subfield/subindices of ports are only applied to
// matching fieldIDs.
    // The annotation should be copied to just a.a.  The hello arg
    // attribute should be copied to each new port.
    module private @PortBundle(in %a: !firrtl.bundle<a: uint<1>, b flip: uint<1>> [{circt.fieldID = 1, a}]) {}
    // CHECK-LABEL: module private @PortBundle
    // CHECK-SAME:    in %a_a: !firrtl.uint<1> [{a}]

// circuit Foo:
//   module Foo:
//     input a: UInt<2>[2][2]
//     input sel: UInt<2>
//     output b: UInt<2>
//
//     b <= a[sel][sel]

  module private @multidimRead(in %a: !firrtl.vector<vector<uint<2>, 2>, 2>, in %sel: !firrtl.uint<2>, out %b: !firrtl.uint<2>) {
    %0 = subaccess %a[%sel] : !firrtl.vector<vector<uint<2>, 2>, 2>, !firrtl.uint<2>
    %1 = subaccess %0[%sel] : !firrtl.vector<uint<2>, 2>, !firrtl.uint<2>
    connect %b, %1 : !firrtl.uint<2>, !firrtl.uint<2>
  }

// CHECK-LABEL: module private @multidimRead(in %a_0_0: !firrtl.uint<2>, in %a_0_1: !firrtl.uint<2>, in %a_1_0: !firrtl.uint<2>, in %a_1_1: !firrtl.uint<2>, in %sel: !firrtl.uint<2>, out %b: !firrtl.uint<2>) {
// CHECK-NEXT:      %0 = multibit_mux %sel, %a_1_0, %a_0_0 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:      %1 = multibit_mux %sel, %a_1_1, %a_0_1 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:      %2 = multibit_mux %sel, %1, %0 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:      connect %b, %2 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }

//  module Foo:
//    input b: UInt<1>
//    input sel: UInt<2>
//    input default: UInt<1>[4]
//    output a: UInt<1>[4]
//
//     a <= default
//     a[sel] <= b

  module private @write1D(in %b: !firrtl.uint<1>, in %sel: !firrtl.uint<2>, in %default: !firrtl.vector<uint<1>, 2>, out %a: !firrtl.vector<uint<1>, 2>) {
    connect %a, %default : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
    %0 = subaccess %a[%sel] : !firrtl.vector<uint<1>, 2>, !firrtl.uint<2>
    connect %0, %b : !firrtl.uint<1>, !firrtl.uint<1>
  }
// CHECK-LABEL:    module private @write1D(in %b: !firrtl.uint<1>, in %sel: !firrtl.uint<2>, in %default_0: !firrtl.uint<1>, in %default_1: !firrtl.uint<1>, out %a_0: !firrtl.uint<1>, out %a_1: !firrtl.uint<1>) {
// CHECK-NEXT:      strictconnect %a_0, %default_0 : !firrtl.uint<1>
// CHECK-NEXT:      strictconnect %a_1, %default_1 : !firrtl.uint<1>
// CHECK-NEXT:      %c0_ui1 = constant 0 : !firrtl.uint<1>
// CHECK-NEXT:      %0 = eq %sel, %c0_ui1 : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      when %0 : !firrtl.uint<1> {
// CHECK-NEXT:        strictconnect %a_0, %b : !firrtl.uint<1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %c1_ui1 = constant 1 : !firrtl.uint<1>
// CHECK-NEXT:      %1 = eq %sel, %c1_ui1 : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      when %1 : !firrtl.uint<1> {
// CHECK-NEXT:        strictconnect %a_1, %b : !firrtl.uint<1>
// CHECK-NEXT:      }
// CHECK-NEXT:    }


// circuit Foo:
//   module Foo:
//     input sel: UInt<1>
//     input b: UInt<2>
//     output a: UInt<2>[2][2]
//
//     a[sel][sel] <= b

  module private @multidimWrite(in %sel: !firrtl.uint<1>, in %b: !firrtl.uint<2>, out %a: !firrtl.vector<vector<uint<2>, 2>, 2>) {
    %0 = subaccess %a[%sel] : !firrtl.vector<vector<uint<2>, 2>, 2>, !firrtl.uint<1>
    %1 = subaccess %0[%sel] : !firrtl.vector<uint<2>, 2>, !firrtl.uint<1>
    connect %1, %b : !firrtl.uint<2>, !firrtl.uint<2>
  }
// CHECK-LABEL:    module private @multidimWrite(in %sel: !firrtl.uint<1>, in %b: !firrtl.uint<2>, out %a_0_0: !firrtl.uint<2>, out %a_0_1: !firrtl.uint<2>, out %a_1_0: !firrtl.uint<2>, out %a_1_1: !firrtl.uint<2>) {
// CHECK-NEXT:      %c0_ui1 = constant 0 : !firrtl.uint<1>
// CHECK-NEXT:      %0 = eq %sel, %c0_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      when %0 : !firrtl.uint<1> {
// CHECK-NEXT:        %c0_ui1_0 = constant 0 : !firrtl.uint<1>
// CHECK-NEXT:        %2 = eq %sel, %c0_ui1_0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:        when %2 : !firrtl.uint<1> {
// CHECK-NEXT:          strictconnect %a_0_0, %b : !firrtl.uint<2>
// CHECK-NEXT:        }
// CHECK-NEXT:        %c1_ui1_1 = constant 1 : !firrtl.uint<1>
// CHECK-NEXT:        %3 = eq %sel, %c1_ui1_1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:        when %3 : !firrtl.uint<1> {
// CHECK-NEXT:          strictconnect %a_0_1, %b : !firrtl.uint<2>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %c1_ui1 = constant 1 : !firrtl.uint<1>
// CHECK-NEXT:      %1 = eq %sel, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      when %1 : !firrtl.uint<1> {
// CHECK-NEXT:        %c0_ui1_0 = constant 0 : !firrtl.uint<1>
// CHECK-NEXT:        %2 = eq %sel, %c0_ui1_0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:        when %2 : !firrtl.uint<1> {
// CHECK-NEXT:          strictconnect %a_1_0, %b : !firrtl.uint<2>
// CHECK-NEXT:        }
// CHECK-NEXT:        %c1_ui1_1 = constant 1 : !firrtl.uint<1>
// CHECK-NEXT:        %3 = eq %sel, %c1_ui1_1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:        when %3 : !firrtl.uint<1> {
// CHECK-NEXT:          strictconnect %a_1_1, %b : !firrtl.uint<2>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// circuit Foo:
//   module Foo:
//     input a: {wo: UInt<1>, valid: UInt<2>}
//     input def: {wo: UInt<1>, valid: UInt<2>}[4]
//     input sel: UInt<2>
//     output b: {wo: UInt<1>, valid: UInt<2>}[4]
//
//     b <= def
//     b[sel].wo <= a.wo
  module private @writeVectorOfBundle1D(in %a: !firrtl.bundle<wo: uint<1>, valid: uint<2>>, in %def: !firrtl.vector<bundle<wo: uint<1>, valid: uint<2>>, 2>, in %sel: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, valid: uint<2>>, 2>) {
    connect %b, %def : !firrtl.vector<bundle<wo: uint<1>, valid: uint<2>>, 2>, !firrtl.vector<bundle<wo: uint<1>, valid: uint<2>>, 2>
    %0 = subaccess %b[%sel] : !firrtl.vector<bundle<wo: uint<1>, valid: uint<2>>, 2>, !firrtl.uint<2>
    %1 = subfield %0[wo] : !firrtl.bundle<wo: uint<1>, valid: uint<2>>
    %2 = subfield %a[wo] : !firrtl.bundle<wo: uint<1>, valid: uint<2>>
    connect %1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }

// CHECK-LABEL:    module private @writeVectorOfBundle1D(in %a_wo: !firrtl.uint<1>, in %a_valid: !firrtl.uint<2>, in %def_0_wo: !firrtl.uint<1>, in %def_0_valid: !firrtl.uint<2>, in %def_1_wo: !firrtl.uint<1>, in %def_1_valid: !firrtl.uint<2>, in %sel: !firrtl.uint<2>, out %b_0_wo: !firrtl.uint<1>, out %b_0_valid: !firrtl.uint<2>, out %b_1_wo: !firrtl.uint<1>, out %b_1_valid: !firrtl.uint<2>) {
// CHECK-NEXT:      strictconnect %b_0_wo, %def_0_wo : !firrtl.uint<1>
// CHECK-NEXT:      strictconnect %b_0_valid, %def_0_valid : !firrtl.uint<2>
// CHECK-NEXT:      strictconnect %b_1_wo, %def_1_wo : !firrtl.uint<1>
// CHECK-NEXT:      strictconnect %b_1_valid, %def_1_valid : !firrtl.uint<2>
// CHECK-NEXT:      %c0_ui1 = constant 0 : !firrtl.uint<1>
// CHECK-NEXT:      %0 = eq %sel, %c0_ui1 : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      when %0 : !firrtl.uint<1> {
// CHECK-NEXT:        strictconnect %b_0_wo, %a_wo : !firrtl.uint<1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %c1_ui1 = constant 1 : !firrtl.uint<1>
// CHECK-NEXT:      %1 = eq %sel, %c1_ui1 : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:      when %1 : !firrtl.uint<1> {
// CHECK-NEXT:        strictconnect %b_1_wo, %a_wo : !firrtl.uint<1>
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// circuit Foo:
//   module Foo:
//     input a: UInt<2>[2][2]
//     input sel1: UInt<1>
//     input sel2: UInt<1>
//     output b: UInt<2>
//     output c: UInt<2>
//
//     b <= a[sel1][sel1]
//     c <= a[sel1][sel2]
  module private @multiSubaccess(in %a: !firrtl.vector<vector<uint<2>, 2>, 2>, in %sel1: !firrtl.uint<1>, in %sel2: !firrtl.uint<1>, out %b: !firrtl.uint<2>, out %c: !firrtl.uint<2>) {
    %0 = subaccess %a[%sel1] : !firrtl.vector<vector<uint<2>, 2>, 2>, !firrtl.uint<1>
    %1 = subaccess %0[%sel1] : !firrtl.vector<uint<2>, 2>, !firrtl.uint<1>
    connect %b, %1 : !firrtl.uint<2>, !firrtl.uint<2>
    %2 = subaccess %a[%sel1] : !firrtl.vector<vector<uint<2>, 2>, 2>, !firrtl.uint<1>
    %3 = subaccess %2[%sel2] : !firrtl.vector<uint<2>, 2>, !firrtl.uint<1>
    connect %c, %3 : !firrtl.uint<2>, !firrtl.uint<2>
  }

// CHECK-LABEL:    module private @multiSubaccess(in %a_0_0: !firrtl.uint<2>, in %a_0_1: !firrtl.uint<2>, in %a_1_0: !firrtl.uint<2>, in %a_1_1: !firrtl.uint<2>, in %sel1: !firrtl.uint<1>, in %sel2: !firrtl.uint<1>, out %b: !firrtl.uint<2>, out %c: !firrtl.uint<2>) {
// CHECK-NEXT:      %0 = multibit_mux %sel1, %a_1_0, %a_0_0 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      %1 = multibit_mux %sel1, %a_1_1, %a_0_1 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      %2 = multibit_mux %sel1, %1, %0 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      connect %b, %2 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:      %3 = multibit_mux %sel1, %a_1_0, %a_0_0 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      %4 = multibit_mux %sel1, %a_1_1, %a_0_1 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      %5 = multibit_mux %sel2, %4, %3 : !firrtl.uint<1>, !firrtl.uint<2>
// CHECK-NEXT:      connect %c, %5 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT:    }


// Handle zero-length vector subaccess
  // CHECK-LABEL: zvec
  module private @zvec(in %i: !firrtl.vector<bundle<a: uint<8>, b: uint<4>>, 0>, in %sel: !firrtl.uint<1>, out %foo: !firrtl.vector<uint<1>, 0>, out %o: !firrtl.uint<8>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %0 = subaccess %foo[%c0_ui1] : !firrtl.vector<uint<1>, 0>, !firrtl.uint<1>
    connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = subaccess %i[%sel] : !firrtl.vector<bundle<a: uint<8>, b: uint<4>>, 0>, !firrtl.uint<1>
    %2 = subfield %1[a] : !firrtl.bundle<a: uint<8>, b: uint<4>>
    connect %o, %2 : !firrtl.uint<8>, !firrtl.uint<8>
  // CHECK: connect %o, %invalid_ui8
  }

// Test InstanceOp with port annotations.
// CHECK-LABEL module private @Bar3
  module private @Bar3(in %a: !firrtl.uint<1>, out %b: !firrtl.bundle<baz: uint<1>, qux: uint<1>>) {
  }
  // CHECK-LABEL module private @Foo3
  module private @Foo3() {
    // CHECK: in a: !firrtl.uint<1> [{one}], out b_baz: !firrtl.uint<1> [{two}], out b_qux: !firrtl.uint<1>
    %bar_a, %bar_b = instance bar @Bar3(
      in a: !firrtl.uint<1> [{one}],
      out b: !firrtl.bundle<baz: uint<1>, qux: uint<1>> [{circt.fieldID = 1, two}]
    )
  }


// Test MemOp with port annotations.
// circuit Foo: %[[{"a":null,"target":"~Foo|Foo>bar.r"},
//                 {"b":null,"target":"~Foo|Foo>bar.r.data"},
//                 {"c":null,"target":"~Foo|Foo>bar.w.en"},
//                 {"d":null,"target":"~Foo|Foo>bar.w.data.qux"},
//                 {"e":null,"target":"~Foo|Foo>bar.rw.wmode"}
//                 {"f":null,"target":"~Foo|Foo>bar.rw.wmask.baz"}]]

// CHECK-LABEL: module private @Foo4
  module private @Foo4() {
    // CHECK: mem
    // CHECK-SAME: portAnnotations = [
    // CHECK-SAME: [{a}, {b, circt.fieldID = 4 : i32}],
    // CHECK-SAME: [{c, circt.fieldID = 2 : i32}]
    // CHECK-SAME: [{circt.fieldID = 4 : i32, e}, {circt.fieldID = 7 : i32, f}]

    // CHECK: mem
    // CHECK-SAME: portAnnotations = [
    // CHECK-SAME: [{a}, {b, circt.fieldID = 4 : i32}],
    // CHECK-SAME: [{c, circt.fieldID = 2 : i32}, {circt.fieldID = 4 : i32, d}]
    // CHECK-SAME: [{circt.fieldID = 4 : i32, e}]

    %bar_r, %bar_w, %bar_rw = mem Undefined  {depth = 16 : i64, name = "bar",
        portAnnotations = [
          [{a}, {circt.fieldID = 4 : i32, b}],
          [{circt.fieldID = 2 : i32, c}, {circt.fieldID = 6 : i32, d}],
          [{circt.fieldID = 4 : i32, e}, {circt.fieldID = 12 : i32, f}]
        ],
        portNames = ["r", "w", "rw"], readLatency = 0 : i32, writeLatency = 1 : i32} :
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<baz: uint<8>, qux: uint<8>>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<baz: uint<8>, qux: uint<8>>, mask: bundle<baz: uint<1>, qux: uint<1>>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<baz: uint<8>, qux: uint<8>>, wmode: uint<1>, wdata: bundle<baz: uint<8>, qux: uint<8>>, wmask: bundle<baz: uint<1>, qux: uint<1>>>
  }

// This simply has to not crash
// CHECK-LABEL: module private @vecmem
firrtl.module private @vecmem(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
  %vmem_MPORT, %vmem_rdwrPort = mem Undefined  {depth = 32 : i64, name = "vmem", portNames = ["MPORT", "rdwrPort"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: vector<uint<17>, 8>>, !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, rdata flip: vector<uint<17>, 8>, wmode: uint<1>, wdata: vector<uint<17>, 8>, wmask: vector<uint<1>, 8>>
}

// Issue 1436
firrtl.extmodule @is1436_BAR(out io: !firrtl.bundle<llWakeup flip: vector<uint<1>, 1>>)
// CHECK-LABEL: module private @is1436_FOO
firrtl.module private @is1436_FOO() {
  %thing_io = instance thing @is1436_BAR(out io: !firrtl.bundle<llWakeup flip: vector<uint<1>, 1>>)
  %0 = subfield %thing_io[llWakeup] : !firrtl.bundle<llWakeup flip: vector<uint<1>, 1>>
  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %1 = subaccess %0[%c0_ui2] : !firrtl.vector<uint<1>, 1>, !firrtl.uint<2>
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: module private @BitCast1
  module private @BitCast1(out %o_0: !firrtl.uint<1>, out %o_1: !firrtl.uint<1>) {
    %a1 = wire : !firrtl.uint<4>
    %b = bitcast %a1 : (!firrtl.uint<4>) -> (!firrtl.vector<uint<2>, 2>)
    // CHECK:  %[[v0:.+]] = bits %a1 1 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<2>
    // CHECK-NEXT:  %[[v1:.+]] = bits %a1 3 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<2>
    // CHECK-NEXT:  %[[v2:.+]] = cat %[[v1]], %[[v0]] : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<4>
    %c = bitcast %b : (!firrtl.vector<uint<2>, 2>) -> (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<2>>)
    // CHECK-NEXT:  %[[v3:.+]] = bits %[[v2]] 0 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT:  %[[v4:.+]] = bits %[[v2]] 1 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<1>
    // CHECK-NEXT:  %[[v5:.+]] = bits %[[v2]] 3 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<2>
    %d = wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<2>>
    // CHECK-NEXT:  %d_valid = wire  : !firrtl.uint<1>
    // CHECK-NEXT:  %d_ready = wire  : !firrtl.uint<1>
    // CHECK-NEXT:  %d_data = wire  : !firrtl.uint<2>
    connect %d , %c: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<2>>, !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<2>>
    // CHECK-NEXT:  strictconnect %d_valid, %[[v3]] : !firrtl.uint<1>
    // CHECK-NEXT:  strictconnect %d_ready, %[[v4]] : !firrtl.uint<1>
    // CHECK-NEXT:  strictconnect %d_data, %[[v5]] : !firrtl.uint<2>
    %e = bitcast %d : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<2>>) -> (!firrtl.bundle<addr: uint<2>, data : vector<uint<1>, 2>>)
    // CHECK-NEXT:  %[[v6:.+]] = cat %d_ready, %d_valid : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    // CHECK-NEXT:  %[[v7:.+]] = cat %d_data, %[[v6]] : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<4>
    // CHECK-NEXT:  %[[v8:.+]] = bits %[[v7]] 1 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<2>
    // CHECK-NEXT:  %[[v9:.+]] = bits %[[v7]] 3 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  %o1 = subfield %e[data] : !firrtl.bundle<addr: uint<2>, data : vector<uint<1>, 2>>
   %c2 = bitcast %a1 : (!firrtl.uint<4>) -> (!firrtl.bundle<valid: bundle<re: bundle<a: uint<1>>, aa: uint<1>>, ready: uint<1>, data: uint<1>>)
    %d2 = wire : !firrtl.bundle<valid: bundle<re: bundle<a: uint<1>>, aa: uint<1>>, ready: uint<1>, data: uint<1>>
    connect %d2 , %c2: !firrtl.bundle<valid: bundle<re: bundle<a: uint<1>>, aa: uint<1>>, ready: uint<1>, data:
    uint<1>>, !firrtl.bundle<valid: bundle<re: bundle<a: uint<1>>, aa: uint<1>>, ready: uint<1>, data: uint<1>>
   //CHECK: %[[v10:.+]] = bits %[[v9]] 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
   //CHECK: %[[v11:.+]] = bits %[[v9]] 1 to 1 : (!firrtl.uint<2>) -> !firrtl.uint<1>
   //CHECK: %[[v12:.+]] = bits %a1 1 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<2>
   //CHECK: %[[v13:.+]] = bits %[[v12]] 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
   //CHECK: %[[v14:.+]] = bits %[[v13]] 0 to 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
   //CHECK: %[[v15:.+]] = bits %[[v12]] 1 to 1 : (!firrtl.uint<2>) -> !firrtl.uint<1>
   //CHECK: %[[v16:.+]] = bits %a1 2 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<1>
   //CHECK: %[[v17:.+]] = bits %a1 3 to 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
   //CHECK: %[[d2_valid_re_a:.+]] = wire  : !firrtl.uint<1>
   //CHECK: %[[d2_valid_aa:.+]] = wire  : !firrtl.uint<1>
   //CHECK: %[[d2_ready:.+]] = wire  : !firrtl.uint<1>
   //CHECK: %[[d2_data:.+]] = wire  : !firrtl.uint<1>
   //CHECK: strictconnect %[[d2_valid_re_a]], %[[v14]] : !firrtl.uint<1>
   //CHECK: strictconnect %[[d2_valid_aa]], %[[v15]] : !firrtl.uint<1>
   //CHECK: strictconnect %[[d2_ready]], %[[v16]] : !firrtl.uint<1>
   //CHECK: strictconnect %[[d2_data]], %[[v17]] : !firrtl.uint<1>

  }

  // Issue #2315: Narrow index constants overflow when subaccessing long vectors.
  // https://github.com/llvm/circt/issues/2315
  // CHECK-LABEL: module private @Issue2315
  module private @Issue2315(in %x: !firrtl.vector<uint<10>, 5>, in %source: !firrtl.uint<2>, out %z: !firrtl.uint<10>) {
    %0 = subaccess %x[%source] : !firrtl.vector<uint<10>, 5>, !firrtl.uint<2>
    connect %z, %0 : !firrtl.uint<10>, !firrtl.uint<10>
    // The width of multibit mux index will be converted at LowerToHW,
    // so it is ok that the type of `%source` is uint<2> here.
    // CHECK:      %0 = multibit_mux %source, %x_4, %x_3, %x_2, %x_1, %x_0 : !firrtl.uint<2>, !firrtl.uint<10>
    // CHECK-NEXT: connect %z, %0 : !firrtl.uint<10>, !firrtl.uint<10>
  }

  module private @SendRefTypeBundles1(in %source: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>, out %sink: !firrtl.probe<bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>>) {
    // CHECK:  module private @SendRefTypeBundles1(
    // CHECK-SAME:  in %source_valid: !firrtl.uint<1>,
    // CHECK-SAME:  in %source_ready: !firrtl.uint<1>,
    // CHECK-SAME:  in %source_data: !firrtl.uint<64>,
    // CHECK-SAME:  out %sink_valid: !firrtl.probe<uint<1>>,
    // CHECK-SAME:  out %sink_ready: !firrtl.probe<uint<1>>,
    // CHECK-SAME:  out %sink_data: !firrtl.probe<uint<64>>) {
    %0 = ref.send %source : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
    // CHECK:  %0 = ref.send %source_valid : !firrtl.uint<1>
    // CHECK:  %1 = ref.send %source_ready : !firrtl.uint<1>
    // CHECK:  %2 = ref.send %source_data : !firrtl.uint<64>
    ref.define %sink, %0 : !firrtl.probe<bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>>
    // CHECK:  ref.define %sink_valid, %0 : !firrtl.probe<uint<1>>
    // CHECK:  ref.define %sink_ready, %1 : !firrtl.probe<uint<1>>
    // CHECK:  ref.define %sink_data, %2 : !firrtl.probe<uint<64>>
  }
  module private @SendRefTypeVectors1(in %a: !firrtl.vector<uint<1>, 2>, out %b: !firrtl.probe<vector<uint<1>, 2>>) {
    // CHECK-LABEL: module private @SendRefTypeVectors1
    // CHECK-SAME: in %a_0: !firrtl.uint<1>, in %a_1: !firrtl.uint<1>, out %b_0: !firrtl.probe<uint<1>>, out %b_1: !firrtl.probe<uint<1>>)
    %0 = ref.send %a : !firrtl.vector<uint<1>, 2>
    // CHECK:  %0 = ref.send %a_0 : !firrtl.uint<1>
    // CHECK:  %1 = ref.send %a_1 : !firrtl.uint<1>
    ref.define %b, %0 : !firrtl.probe<vector<uint<1>, 2>>
    // CHECK:  ref.define %b_0, %0 : !firrtl.probe<uint<1>>
    // CHECK:  ref.define %b_1, %1 : !firrtl.probe<uint<1>>
  }
  module private @RefTypeBundles2() {
    %x = wire   : !firrtl.bundle<a: uint<1>, b: uint<2>>
    %0 = ref.send %x : !firrtl.bundle<a: uint<1>, b: uint<2>>
    // CHECK:   %0 = ref.send %x_a : !firrtl.uint<1>
    // CHECK:   %1 = ref.send %x_b : !firrtl.uint<2>
    %1 = ref.resolve %0 : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>
    // CHECK:   %2 = ref.resolve %0 : !firrtl.probe<uint<1>>
    // CHECK:   %3 = ref.resolve %1 : !firrtl.probe<uint<2>>
  }
  module private @RefTypeVectors(out %c: !firrtl.vector<uint<1>, 2>) {
    %x = wire   : !firrtl.vector<uint<1>, 2>
    %0 = ref.send %x : !firrtl.vector<uint<1>, 2>
    // CHECK:  %0 = ref.send %x_0 : !firrtl.uint<1>
    // CHECK:  %1 = ref.send %x_1 : !firrtl.uint<1>
    %1 = ref.resolve %0 : !firrtl.probe<vector<uint<1>, 2>>
    // CHECK:  %2 = ref.resolve %0 : !firrtl.probe<uint<1>>
    // CHECK:  %3 = ref.resolve %1 : !firrtl.probe<uint<1>>
    strictconnect %c, %1 : !firrtl.vector<uint<1>, 2>
    // CHECK:  strictconnect %c_0, %2 : !firrtl.uint<1>
    // CHECK:  strictconnect %c_1, %3 : !firrtl.uint<1>
  }

  // CHECK-LABEL: module private @RefTypeBV_RW
  module private @RefTypeBV_RW(
    // CHECK-SAME: rwprobe<vector<uint<1>, 2>>
    out %vec_ref: !firrtl.rwprobe<vector<uint<1>, 2>>,
    // CHECK-NOT: vector
    out %vec: !firrtl.vector<uint<1>, 2>,
    // CHECK-SAME: rwprobe<bundle
    out %bov_ref: !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>,
    // CHECK-NOT: bundle
    out %bov: !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>,
    // CHECK: probe
    out %probe: !firrtl.probe<uint<2>>
  ) {
    // Forceable declaration are never expanded into ground elements.
    // CHECK-NEXT: %{{.+}}, %[[X_REF:.+]] = wire forceable : !firrtl.bundle<a: vector<uint<1>, 2>, b flip: uint<2>>, !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>
    %x, %x_ref = wire forceable : !firrtl.bundle<a: vector<uint<1>, 2>, b flip: uint<2>>, !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>

    // Define using forceable ref preserved.
    // CHECK-NEXT: ref.define %{{.+}}, %[[X_REF]] : !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>
    ref.define %bov_ref, %x_ref : !firrtl.rwprobe<bundle<a: vector<uint<1> ,2>, b: uint<2>>>

    // Preserve ref.sub uses.
    // CHECK-NEXT: %[[X_REF_A:.+]] = ref.sub %[[X_REF]][0]
    // CHECK-NEXT: %[[X_A:.+]] = ref.resolve %[[X_REF_A]]
    // CHECK-NEXT: %[[v_0:.+]] = subindex %[[X_A]][0]
    // CHECK-NEXT: strictconnect %vec_0, %[[v_0]]
    // CHECK-NEXT: %[[v_1:.+]] = subindex %[[X_A]][1]
    // CHECK-NEXT: strictconnect %vec_1, %[[v_1]]
    %x_ref_a = ref.sub %x_ref[0] : !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>
    %x_a = ref.resolve %x_ref_a : !firrtl.rwprobe<vector<uint<1>, 2>>
    strictconnect %vec, %x_a : !firrtl.vector<uint<1>, 2>

    // Check chained ref.sub's work.
    // CHECK-NEXT: %[[X_A_1_REF:.+]] = ref.sub %[[X_REF_A]][1]
    // CHECK-NEXT: ref.resolve %[[X_A_1_REF]]
    %x_ref_a_1 = ref.sub %x_ref_a[1] : !firrtl.rwprobe<vector<uint<1>, 2>>
    %x_a_1 = ref.resolve %x_ref_a_1 : !firrtl.rwprobe<uint<1>>

    // Ref to flipped field.
    // CHECK-NEXT: %[[X_B_REF:.+]] = ref.sub %[[X_REF]][1]
    // CHECK-NEXT: ref.resolve %[[X_B_REF]]
    %x_ref_b = ref.sub %x_ref[1] : !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>
    %x_b = ref.resolve %x_ref_b : !firrtl.rwprobe<uint<2>>

    // TODO: Handle rwprobe --> probe define, enable this.
    // ref.define %probe, %x_ref_b : !firrtl.probe<uint<2>>

    // Check resolve of rwprobe is preserved.
    // CHECK-NEXT: = ref.resolve %[[X_REF]]
    // CHECK: strictconnect %bov_a_0,
    // CHECK: strictconnect %bov_a_1,
    // CHECK: strictconnect %bov_b,
    %x_read = ref.resolve %x_ref : !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>
    strictconnect %bov, %x_read : !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>
    // CHECK-NEXT: }
  }
  // Check how rwprobe's of aggregates in instances are handled.
  // CHECK-LABEL: module private @InstWithRWProbeOfAgg
  module private @InstWithRWProbeOfAgg(in %clock : !firrtl.clock, in %pred : !firrtl.uint<1>) {
    // CHECK: {{((%[^,]+, ){3})}}
    // CHECK-SAME: %[[BOV_REF:[^,]+]],
    // CHECK-SAME: %[[BOV_A_0:.+]],        %[[BOV_A_1:.+]],        %[[BOV_B:.+]],        %{{.+}} = instance
    // CHECK-NOT: probe
    // CHECK-SAME: probe: !firrtl.probe<uint<2>>)
    %inst_vec_ref, %inst_vec, %inst_bov_ref, %inst_bov, %inst_probe = instance inst @RefTypeBV_RW(
      out vec_ref: !firrtl.rwprobe<vector<uint<1>, 2>>,
      out vec: !firrtl.vector<uint<1>, 2>,
      out bov_ref: !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>,
      out bov: !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>,
      out probe: !firrtl.probe<uint<2>>)

    // Check lowering force and release operations.
    // Use self-assigns for simplicity.
    // Source operand may need to be materialized from its elements.
    // CHECK: vectorcreate
    // CHECK: bundlecreate
    // CHECK: ref.force %clock, %pred, %[[BOV_REF]],
    ref.force %clock, %pred, %inst_bov_ref, %inst_bov : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>
    // CHECK: ref.force_initial %pred, %[[BOV_REF]],
    ref.force_initial %pred, %inst_bov_ref, %inst_bov : !firrtl.uint<1>, !firrtl.bundle<a: vector<uint<1>, 2>, b: uint<2>>
    // CHECK: ref.release %clock, %pred, %[[BOV_REF]] :
    ref.release %clock, %pred, %inst_bov_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>
    // CHECK: ref.release_initial %pred, %[[BOV_REF]] :
    ref.release_initial %pred, %inst_bov_ref : !firrtl.uint<1>, !firrtl.rwprobe<bundle<a: vector<uint<1>, 2>, b: uint<2>>>
    // CHECK-NEXT: }
  }

  // CHECK-LABEL: module private @ForeignTypes
  module private @ForeignTypes() {
    // CHECK-NEXT: wire : index
    %0 = wire : index
  }

  // CHECK-LABEL: module @MergeBundle
  module @MergeBundle(out %o: !firrtl.bundle<valid: uint<1>, ready: uint<1>>, in %i: !firrtl.uint<1>)
  {
    %a = wire   : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
    strictconnect %o, %a : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
    %0 = bundlecreate %i, %i : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.bundle<valid: uint<1>, ready: uint<1>>
    strictconnect %a, %0 : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
    // CHECK:  %a_valid = wire   : !firrtl.uint<1>
    // CHECK:  %a_ready = wire   : !firrtl.uint<1>
    // CHECK:  strictconnect %o_valid, %a_valid : !firrtl.uint<1>
    // CHECK:  strictconnect %o_ready, %a_ready : !firrtl.uint<1>
    // CHECK:  strictconnect %a_valid, %i : !firrtl.uint<1>
    // CHECK:  strictconnect %a_ready, %i : !firrtl.uint<1>
  }

  // CHECK-LABEL: module @MergeVector
  module @MergeVector(out %o: !firrtl.vector<uint<1>, 3>, in %i: !firrtl.uint<1>) {
    %a = wire   : !firrtl.vector<uint<1>, 3>
    strictconnect %o, %a : !firrtl.vector<uint<1>, 3>
    %0 = vectorcreate %i, %i, %i : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 3>
    strictconnect %a, %0 : !firrtl.vector<uint<1>, 3>
    // CHECK:  %a_0 = wire   : !firrtl.uint<1>
    // CHECK:  %a_1 = wire   : !firrtl.uint<1>
    // CHECK:  %a_2 = wire   : !firrtl.uint<1>
    // CHECK:  strictconnect %o_0, %a_0 : !firrtl.uint<1>
    // CHECK:  strictconnect %o_1, %a_1 : !firrtl.uint<1>
    // CHECK:  strictconnect %o_2, %a_2 : !firrtl.uint<1>
    // CHECK:  strictconnect %a_0, %i : !firrtl.uint<1>
    // CHECK:  strictconnect %a_1, %i : !firrtl.uint<1>
    // CHECK:  strictconnect %a_2, %i : !firrtl.uint<1>
  }

  // Check that an instance with attributes known and unknown to FIRRTL Dialect
  // are copied to the lowered instance.
  module @SubmoduleWithAggregate(out %a: !firrtl.vector<uint<1>, 1>) {}
  // CHECK-LABEL: module @ModuleWithInstanceAttributes
  module @ModuleWithInstanceAttributes() {
    // CHECK-NEXT: instance
    // CHECK-SAME:   hello = "world"
    // CHECK-SAME:   lowerToBind
    // CHECK-SAME:   output_file = #hw.output_file<"Foo.sv">
    %sub_a = instance sub {
      hello = "world",
      lowerToBind,
      output_file = #hw.output_file<"Foo.sv">
    } @SubmoduleWithAggregate(out a: !firrtl.vector<uint<1>, 1>)
  }

  // COMMON-LABEL: module @ElementWise
  module @ElementWise(in %a: !firrtl.vector<uint<1>, 1>, in %b: !firrtl.vector<uint<1>, 1>, out %c_0: !firrtl.vector<uint<1>, 1>, out %c_1: !firrtl.vector<uint<1>, 1>, out %c_2: !firrtl.vector<uint<1>, 1>) {
    // CHECK-NEXT: %0 = or %a_0, %b_0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %c_0_0, %0 : !firrtl.uint<1>
    // CHECK-NEXT: %1 = and %a_0, %b_0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %c_1_0, %1 : !firrtl.uint<1>
    // CHECK-NEXT: %2 = xor %a_0, %b_0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %c_2_0, %2 : !firrtl.uint<1>
    // Check that elementwise_* are preserved.
    // AGGREGATE: elementwise_or
    // AGGREGATE: elementwise_and
    // AGGREGATE: elementwise_xor
    %0 = elementwise_or %a, %b : (!firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>) -> !firrtl.vector<uint<1>, 1>
    strictconnect %c_0, %0 : !firrtl.vector<uint<1>, 1>
    %1 = elementwise_and %a, %b : (!firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>) -> !firrtl.vector<uint<1>, 1>
    strictconnect %c_1, %1 : !firrtl.vector<uint<1>, 1>
    %2 = elementwise_xor %a, %b : (!firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>) -> !firrtl.vector<uint<1>, 1>
    strictconnect %c_2, %2 : !firrtl.vector<uint<1>, 1>
  }

} // CIRCUIT

// Check that we don't lose the DontTouchAnnotation when it is not the last
// annotation in the list of annotations.
// https://github.com/llvm/circt/issues/3504
// CHECK-LABEL: circuit "DontTouch"
firrtl.circuit "DontTouch" {
  // CHECK: in %port_field: !firrtl.uint<1> [{class = "firrtl.transforms.DontTouchAnnotation"}, {class = "Test"}]
  module @DontTouch (in %port: !firrtl.bundle<field: uint<1>> [
    {circt.fieldID = 1 : i32, class = "firrtl.transforms.DontTouchAnnotation"},
    {circt.fieldID = 1 : i32, class = "Test"}
  ]) {
 }
}

// Check that we don't create symbols for non-local annotations.
firrtl.circuit "Foo"  {
  hw.hierpath private @nla [@Foo::@bar, @Bar]
  // CHECK:       module private @Bar(in %a_b:
  // CHECK-SAME:    !firrtl.uint<1> [{circt.nonlocal = @nla, class = "circt.test"}])
  module private @Bar(in %a: !firrtl.bundle<b: uint<1>>
      [{circt.fieldID = 1 : i32, circt.nonlocal = @nla, class = "circt.test"}]) {
  }
  module @Foo() {
    %bar_a = instance bar sym @bar @Bar(in a: !firrtl.bundle<b: uint<1>>)
    %invalid = invalidvalue : !firrtl.bundle<b: uint<1>>
    strictconnect %bar_a, %invalid : !firrtl.bundle<b: uint<1>>
  }
}
