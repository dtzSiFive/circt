// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-infer-rw)))' %s | FileCheck %s

firrtl.circuit "TLRAM" {
// Test the case when the enable is a simple not of write enable.
// CHECK-LABEL: module @TLRAM
    module @TLRAM(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %index: !firrtl.uint<4>, in %index2: !firrtl.uint<4>, in %data_0: !firrtl.uint<8>, in %wen: !firrtl.uint<1>, in %_T_29: !firrtl.uint<1>, out %auto_0: !firrtl.uint<8>, out %dbg_0: !firrtl.probe<vector<uint<8>, 16>>) {
      %mem_MPORT_en = wire  : !firrtl.uint<1>
      %mem_MPORT_data_0 = wire  : !firrtl.uint<8>
      %debug, %mem_0_MPORT, %mem_0_MPORT_1 = mem Undefined  {depth = 16 : i64, name = "mem_0", portNames = ["dbgs", "MPORT", "MPORT_1"], prefix = "foo_", readLatency = 1 : i32, writeLatency = 1 : i32} :  !firrtl.probe<vector<uint<8>, 16>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
      ref.define %dbg_0, %debug : !firrtl.probe<vector<uint<8>, 16>>
      %0 = subfield %mem_0_MPORT[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      connect %0, %index2 : !firrtl.uint<4>, !firrtl.uint<4>
      %1 = subfield %mem_0_MPORT[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      connect %1, %mem_MPORT_en : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = subfield %mem_0_MPORT[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      connect %2, %clock : !firrtl.clock, !firrtl.clock
      %3 = subfield %mem_0_MPORT[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      connect %mem_MPORT_data_0, %3 : !firrtl.uint<8>, !firrtl.uint<8>
      %4 = subfield %mem_0_MPORT_1[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
      connect %4, %index : !firrtl.uint<4>, !firrtl.uint<4>
      %5 = subfield %mem_0_MPORT_1[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
      connect %5, %wen : !firrtl.uint<1>, !firrtl.uint<1>
      %6 = subfield %mem_0_MPORT_1[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
      connect %6, %clock : !firrtl.clock, !firrtl.clock
      %7 = subfield %mem_0_MPORT_1[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
      connect %7, %data_0 : !firrtl.uint<8>, !firrtl.uint<8>
      %8 = subfield %mem_0_MPORT_1[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
      connect %8, %_T_29 : !firrtl.uint<1>, !firrtl.uint<1>
      %9 = not %wen : (!firrtl.uint<1>) -> !firrtl.uint<1>
      connect %mem_MPORT_en, %9 : !firrtl.uint<1>, !firrtl.uint<1>
      %REG = reg %clock  : !firrtl.clock, !firrtl.uint<1>
      connect %REG, %9 : !firrtl.uint<1>, !firrtl.uint<1>
      %r_0 = reg %clock  : !firrtl.clock, !firrtl.uint<8>
      %10 = mux(%REG, %mem_MPORT_data_0, %r_0) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
      connect %r_0, %10 : !firrtl.uint<8>, !firrtl.uint<8>
      %11 = mux(%REG, %mem_MPORT_data_0, %r_0) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
      connect %auto_0, %11 : !firrtl.uint<8>, !firrtl.uint<8>

// CHECK: %mem_0_dbgs, %mem_0_rw = mem  Undefined  {depth = 16 : i64, name = "mem_0", portNames = ["dbgs", "rw"], prefix = "foo_", readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.probe<vector<uint<8>, 16>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>
// CHECK:  %[[v7:.+]] = mux(%[[writeEnable:.+]], %[[writeAddr:.+]], %[[readAddr:.+]]) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
// CHECK:  strictconnect %[[v0:.+]], %[[v7]] : !firrtl.uint<4>
// CHECK:  %[[v8:.+]] = or %[[readEnable:.+]], %[[writeEnable]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:  strictconnect %[[v1:.+]], %[[v8]] : !firrtl.uint<1>
// CHECK:  ref.define %dbg_0, %mem_0_dbgs : !firrtl.probe<vector<uint<8>, 16>>
// CHECK:  connect %[[readAddr]], %[[index2:.+]] : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK:  connect %[[readEnable]], %mem_MPORT_en : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:  connect %[[writeAddr]], %index : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK:  connect %[[writeEnable]], %wen : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:  %[[v10:.+]] = not %wen : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:  connect %mem_MPORT_en, %[[v10]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:  strictconnect %[[v4:.+]], %wen : !firrtl.uint<1>
    }

// Test the pattern of enable  with Mux (sel, high, 0)
// CHECK-LABEL: module @memTest4t
  module @memTest4t(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_addr: !firrtl.uint<11>, in %io_ren: !firrtl.uint<1>, in %io_wen: !firrtl.uint<1>, in %io_dataIn: !firrtl.uint<32>, out %io_dataOut: !firrtl.uint<32>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %mem__T_14, %mem__T_22 = mem Undefined  {depth = 2048 : i64, name = "mem", portNames = ["_T_14", "_T_22"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>, !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
// CHECK:    %mem_rw = mem Undefined  {depth = 2048 : i64, name = "mem", portNames = ["rw"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    %0 = subfield %mem__T_14[addr] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %1 = subfield %mem__T_14[en] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %2 = subfield %mem__T_14[clk] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %3 = subfield %mem__T_14[data] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %4 = subfield %mem__T_14[mask] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %5 = subfield %mem__T_22[addr] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %6 = subfield %mem__T_22[en] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %7 = subfield %mem__T_22[clk] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %8 = subfield %mem__T_22[data] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    connect %0, %io_addr : !firrtl.uint<11>, !firrtl.uint<11>
    connect %1, %io_wen : !firrtl.uint<1>, !firrtl.uint<1>
    connect %2, %clock : !firrtl.clock, !firrtl.clock
    connect %4, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %3, %io_dataIn : !firrtl.uint<32>, !firrtl.uint<32>
    %9 = not %io_wen : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %10 = mux(%9, %io_ren, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %6, %10 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %5, %io_addr : !firrtl.uint<11>, !firrtl.uint<11>
    connect %7, %clock : !firrtl.clock, !firrtl.clock
    connect %io_dataOut, %8 : !firrtl.uint<32>, !firrtl.uint<32>
    // CHECK:   strictconnect %4, %io_wen : !firrtl.uint<1>
  }

// Test the pattern of enable  with an And tree and Mux (sel, high, 0)
// CHECK-LABEL: module @memTest6t
  module @memTest6t(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_addr: !firrtl.uint<11>, in %io_valid: !firrtl.uint<1>, in %io_write: !firrtl.uint<1>, in %io_dataIn: !firrtl.uint<32>, out %io_dataOut: !firrtl.uint<32>) {
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %mem__T_14, %mem__T_22 = mem Undefined  {depth = 2048 : i64, name = "mem", portNames = ["_T_14", "_T_22"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>, !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
// CHECK: %mem_rw = mem Undefined  {depth = 2048 : i64, name = "mem", portNames = ["rw"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
// CHECK:   %[[v7:.+]] = mux(%writeEnable, %writeAddr, %readAddr) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint<11>) -> !firrtl.uint<11>
// CHECK:   strictconnect %[[v0:.+]], %[[v7]]
// CHECK:   %[[v8:.+]] = or %readEnable, %writeEnable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %0 = subfield %mem__T_14[addr] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %1 = subfield %mem__T_14[en] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %2 = subfield %mem__T_14[clk] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %3 = subfield %mem__T_14[data] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %4 = subfield %mem__T_14[mask] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %5 = subfield %mem__T_22[addr] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %6 = subfield %mem__T_22[en] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %7 = subfield %mem__T_22[clk] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %8 = subfield %mem__T_22[data] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %9 = and %io_valid, %io_write : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %10 = not %io_write : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %11 = and %io_valid, %10 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %0, %io_addr : !firrtl.uint<11>, !firrtl.uint<11>
    connect %1, %9 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %2, %clock : !firrtl.clock, !firrtl.clock
    connect %4, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %3, %io_dataIn : !firrtl.uint<32>, !firrtl.uint<32>
    %12 = not %9 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %13 = mux(%12, %11, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %6, %13 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %5, %io_addr : !firrtl.uint<11>, !firrtl.uint<11>
    connect %7, %clock : !firrtl.clock, !firrtl.clock
    connect %io_dataOut, %8 : !firrtl.uint<32>, !firrtl.uint<32>
    // CHECK:  strictconnect %4, %io_write : !firrtl.uint<1>
  }

// Cannot merge read and write, since the pattern is enable = Mux (sel, high, 1)
// CHECK-LABEL: module @memTest7t
  module @memTest7t(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_addr: !firrtl.uint<11>, in %io_valid: !firrtl.uint<1>, in %io_write: !firrtl.uint<1>, in %io_dataIn: !firrtl.uint<32>, out %io_dataOut: !firrtl.uint<32>) {
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %mem__T_14, %mem__T_22 = mem Undefined  {depth = 2048 : i64, name = "mem", portNames = ["_T_14", "_T_22"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>, !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
// CHECK: %mem__T_14, %mem__T_22 = mem Undefined  {depth = 2048 : i64, name = "mem", portNames = ["_T_14", "_T_22"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>, !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %0 = subfield %mem__T_14[addr] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %1 = subfield %mem__T_14[en] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %2 = subfield %mem__T_14[clk] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %3 = subfield %mem__T_14[data] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %4 = subfield %mem__T_14[mask] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %5 = subfield %mem__T_22[addr] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %6 = subfield %mem__T_22[en] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %7 = subfield %mem__T_22[clk] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %8 = subfield %mem__T_22[data] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %9 = and %io_valid, %io_write : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %10 = not %io_write : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %11 = and %io_valid, %10 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %0, %io_addr : !firrtl.uint<11>, !firrtl.uint<11>
    connect %1, %9 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %2, %clock : !firrtl.clock, !firrtl.clock
    connect %4, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %3, %io_dataIn : !firrtl.uint<32>, !firrtl.uint<32>
    %12 = not %9 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %13 = mux(%12, %11, %c1_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %6, %13 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %5, %io_addr : !firrtl.uint<11>, !firrtl.uint<11>
    connect %7, %clock : !firrtl.clock, !firrtl.clock
    connect %io_dataOut, %8 : !firrtl.uint<32>, !firrtl.uint<32>
  }

// Cannot merge, since the clocks are different.
// CHECK-LABEL: module @memTest5t
  module @memTest5t(in %clk1: !firrtl.clock, in %clk2: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_en: !firrtl.uint<1>, in %io_wen: !firrtl.uint<1>, in %io_waddr: !firrtl.uint<8>, in %io_wdata: !firrtl.uint<32>, in %io_raddr: !firrtl.uint<8>, out %io_rdata: !firrtl.uint<32>) {
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %mem_T_3, %mem_T_5 = mem Undefined  {depth = 128 : i64, name = "mem", portNames = ["T_3", "T_5"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>, !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    // CHECK:    %mem_T_3, %mem_T_5 = mem Undefined  {depth = 128 : i64, name = "mem", portNames = ["T_3", "T_5"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>, !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %0 = subfield %mem_T_3[addr] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>
    %1 = subfield %mem_T_3[en] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>
    %2 = subfield %mem_T_3[clk] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>
    %3 = subfield %mem_T_3[data] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>
    %4 = subfield %mem_T_5[addr] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %5 = subfield %mem_T_5[en] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %6 = subfield %mem_T_5[clk] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %7 = subfield %mem_T_5[data] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %8 = subfield %mem_T_5[mask] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %9 = not %io_wen : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %10 = and %io_en, %9 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %1, %10 : !firrtl.uint<1>, !firrtl.uint<1>
    %11 = bits %io_raddr 6 to 0 : (!firrtl.uint<8>) -> !firrtl.uint<7>
    connect %0, %11 : !firrtl.uint<7>, !firrtl.uint<7>
    connect %2, %clk1 : !firrtl.clock, !firrtl.clock
    connect %io_rdata, %3 : !firrtl.uint<32>, !firrtl.uint<32>
    %12 = and %io_en, %io_wen : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %13 = bits %io_waddr 6 to 0 : (!firrtl.uint<8>) -> !firrtl.uint<7>
    connect %4, %13 : !firrtl.uint<7>, !firrtl.uint<7>
    connect %5, %12 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %6, %clk2 : !firrtl.clock, !firrtl.clock
    connect %8, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %7, %io_wdata : !firrtl.uint<32>, !firrtl.uint<32>
  }

// Check for a complement term in the And expression tree.
// CHECK-LABEL: module @memTest3t
  module @memTest3t(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_en: !firrtl.uint<1>, in %io_wen: !firrtl.uint<1>, in %io_waddr: !firrtl.uint<8>, in %io_wdata: !firrtl.uint<32>, in %io_raddr: !firrtl.uint<8>, out %io_rdata: !firrtl.uint<32>) {
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %dbg0, %mem_T_3, %mem_T_5, %dbg = mem Undefined  {depth = 128 : i64, name = "mem", portNames = ["dbg0", "T_3", "T_5", "dbg"], readLatency = 1 : i32, writeLatency = 1 : i32} :  !firrtl.probe<vector<uint<32>,128>>, !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>, !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>, !firrtl.probe<vector<uint<32>,128>>
// CHECK: %mem_dbg0, %mem_dbg, %mem_rw = mem  Undefined  {depth = 128 : i64, name = "mem", portNames = ["dbg0", "dbg", "rw"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.probe<vector<uint<32>, 128>>, !firrtl.probe<vector<uint<32>, 128>>, !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    %0 = subfield %mem_T_3[addr] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>
    %1 = subfield %mem_T_3[en] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>
    %2 = subfield %mem_T_3[clk] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>
    %3 = subfield %mem_T_3[data] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data flip: uint<32>>
    %4 = subfield %mem_T_5[addr] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %5 = subfield %mem_T_5[en] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %6 = subfield %mem_T_5[clk] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %7 = subfield %mem_T_5[data] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %8 = subfield %mem_T_5[mask] : !firrtl.bundle<addr: uint<7>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %9 = not %io_wen : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %10 = and %io_en, %9 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %1, %10 : !firrtl.uint<1>, !firrtl.uint<1>
    %11 = bits %io_raddr 6 to 0 : (!firrtl.uint<8>) -> !firrtl.uint<7>
    connect %0, %11 : !firrtl.uint<7>, !firrtl.uint<7>
    connect %2, %clock : !firrtl.clock, !firrtl.clock
    connect %io_rdata, %3 : !firrtl.uint<32>, !firrtl.uint<32>
    %12 = and %io_en, %io_wen : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %13 = bits %io_waddr 6 to 0 : (!firrtl.uint<8>) -> !firrtl.uint<7>
    connect %4, %13 : !firrtl.uint<7>, !firrtl.uint<7>
    connect %5, %12 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %6, %clock : !firrtl.clock, !firrtl.clock
    connect %8, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %7, %io_wdata : !firrtl.uint<32>, !firrtl.uint<32>
    // CHECK:  strictconnect %4, %io_wen : !firrtl.uint<1>
  }

// Check for indirect connection to clock
// CHECK-LABEL: module @memTest2t
  module @memTest2t(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_addr: !firrtl.uint<11>, in %io_ren: !firrtl.uint<1>, in %io_wen: !firrtl.uint<1>, in %io_dataIn: !firrtl.uint<32>, out %io_dataOut: !firrtl.uint<32>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %mem__T_14, %mem__T_22 = mem Undefined  {depth = 2048 : i64, name = "mem", portNames = ["_T_14", "_T_22"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>, !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
// CHECK:    %mem_rw = mem Undefined  {depth = 2048 : i64, name = "mem", portNames = ["rw"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    %0 = subfield %mem__T_14[addr] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %1 = subfield %mem__T_14[en] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %2 = subfield %mem__T_14[clk] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %3 = subfield %mem__T_14[data] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %4 = subfield %mem__T_14[mask] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    %5 = subfield %mem__T_22[addr] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %6 = subfield %mem__T_22[en] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %7 = subfield %mem__T_22[clk] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    %8 = subfield %mem__T_22[data] : !firrtl.bundle<addr: uint<11>, en: uint<1>, clk: clock, data flip: uint<32>>
    connect %0, %io_addr : !firrtl.uint<11>, !firrtl.uint<11>
    connect %1, %io_wen : !firrtl.uint<1>, !firrtl.uint<1>
    %xc = wire : !firrtl.clock
    connect %2, %xc : !firrtl.clock, !firrtl.clock
    connect %xc, %clock : !firrtl.clock, !firrtl.clock
    connect %4, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %3, %io_dataIn : !firrtl.uint<32>, !firrtl.uint<32>
    %9 = not %io_wen : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %10 = mux(%9, %io_ren, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %6, %10 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %5, %io_addr : !firrtl.uint<11>, !firrtl.uint<11>
    connect %7, %clock : !firrtl.clock, !firrtl.clock
    connect %io_dataOut, %8 : !firrtl.uint<32>, !firrtl.uint<32>
  }

    module @constMask(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %index: !firrtl.uint<4>, in %index2: !firrtl.uint<4>, in %data_0: !firrtl.uint<8>, in %wen: !firrtl.uint<1>, in %_T_29: !firrtl.uint<1>, out %auto_0: !firrtl.uint<8>) {
      %mem_MPORT_en = wire  : !firrtl.uint<1>
      %mem_MPORT_data_0 = wire  : !firrtl.uint<8>
      %mem_0_MPORT, %mem_0_MPORT_1 = mem Undefined  {depth = 16 : i64, name = "mem_0", portNames = ["MPORT", "MPORT_1"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<2>>

      // CHECK: %mem_0_rw = mem Undefined  {depth = 16 : i64, name = "mem_0", portNames = ["rw"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>
      // CHECK: %[[v6:.+]] = subfield %mem_0_rw[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>
      // CHECK: %[[c1_ui1:.+]] = constant 1 : !firrtl.uint<1>
      // CHECK: strictconnect %[[v6]], %[[c1_ui1]]
      %0 = subfield %mem_0_MPORT[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      connect %0, %index2 : !firrtl.uint<4>, !firrtl.uint<4>
      %1 = subfield %mem_0_MPORT[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      connect %1, %mem_MPORT_en : !firrtl.uint<1>, !firrtl.uint<1>
      %2 = subfield %mem_0_MPORT[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      connect %2, %clock : !firrtl.clock, !firrtl.clock
      %3 = subfield %mem_0_MPORT[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
      connect %mem_MPORT_data_0, %3 : !firrtl.uint<8>, !firrtl.uint<8>
      %4 = subfield %mem_0_MPORT_1[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<2>>
      connect %4, %index : !firrtl.uint<4>, !firrtl.uint<4>
      %5 = subfield %mem_0_MPORT_1[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<2>>
      connect %5, %wen : !firrtl.uint<1>, !firrtl.uint<1>
      %6 = subfield %mem_0_MPORT_1[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<2>>
      connect %6, %clock : !firrtl.clock, !firrtl.clock
      %7 = subfield %mem_0_MPORT_1[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<2>>
      connect %7, %data_0 : !firrtl.uint<8>, !firrtl.uint<8>
      %8 = subfield %mem_0_MPORT_1[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<2>>
      %c1 = constant 3 : !firrtl.uint<2>
      connect %8, %c1 : !firrtl.uint<2>, !firrtl.uint<2>
      %9 = not %wen : (!firrtl.uint<1>) -> !firrtl.uint<1>
      connect %mem_MPORT_en, %9 : !firrtl.uint<1>, !firrtl.uint<1>
      %REG = reg %clock  : !firrtl.clock, !firrtl.uint<1>
      connect %REG, %9 : !firrtl.uint<1>, !firrtl.uint<1>
      %r_0 = reg %clock  : !firrtl.clock, !firrtl.uint<8>
      %10 = mux(%REG, %mem_MPORT_data_0, %r_0) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
      connect %r_0, %10 : !firrtl.uint<8>, !firrtl.uint<8>
      %11 = mux(%REG, %mem_MPORT_data_0, %r_0) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
      connect %auto_0, %11 : !firrtl.uint<8>, !firrtl.uint<8>

    }
}

