// RUN: circt-opt --verify-diagnostics %s | FileCheck %s

firrtl.circuit "Foo" {
  extmodule @Bar()

  // CHECK-LABEL: module @Foo
  // CHECK-SAME: in %value: !firrtl.uint<42> sym @symValue
  // CHECK-SAME: in %clock: !firrtl.clock sym @symClock
  // CHECK-SAME: in %reset: !firrtl.asyncreset sym @symReset
  module @Foo(
    in %value: !firrtl.uint<42> sym @symValue,
    in %clock: !firrtl.clock sym @symClock,
    in %reset: !firrtl.asyncreset sym @symReset
  ) {
    // CHECK: instance instName sym @instSym @Bar()
    instance instName sym @instSym @Bar()
    // CHECK: %nodeName = node sym @nodeSym %value : !firrtl.uint<42>
    %nodeName = node sym @nodeSym %value : !firrtl.uint<42>
    // CHECK: %wireName = wire sym @wireSym : !firrtl.uint<42>
    %wireName = wire sym @wireSym : !firrtl.uint<42>
    // CHECK: %regName = reg sym @regSym %clock : !firrtl.clock, !firrtl.uint<42>
    %regName = reg sym @regSym %clock : !firrtl.clock, !firrtl.uint<42>
    // CHECK: %regResetName = regreset sym @regResetSym %clock, %reset, %value : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<42>, !firrtl.uint<42>
    %regResetName = regreset sym @regResetSym %clock, %reset, %value : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<42>, !firrtl.uint<42>
    // CHECK: %memName_port = mem sym @memSym Undefined {depth = 8 : i64, name = "memName", portNames = ["port"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<42>>
    %memName_port = mem sym @memSym Undefined {depth = 8 : i64, name = "memName", portNames = ["port"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<42>>
  }
}
