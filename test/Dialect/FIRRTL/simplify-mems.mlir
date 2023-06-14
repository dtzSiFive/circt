// RUN: circt-opt --split-input-file -canonicalize='top-down=true region-simplify=true' %s | FileCheck %s


firrtl.circuit "ReadOnlyMemory" {
  // CHECK-LABEL: module public @ReadOnlyMemory
  module public @ReadOnlyMemory(in %clock: !firrtl.clock, in %addr: !firrtl.uint<4>) {

    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %Memory_r = mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["read"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>


    // CHECK-NOT: mem
    %2 = subfield %Memory_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %2, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %3 = subfield %Memory_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %3, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %4 = subfield %Memory_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %4, %clock : !firrtl.clock, !firrtl.clock
  }
}

// -----

firrtl.circuit "WriteOnlyMemory" {
  // CHECK-LABEL: module public @WriteOnlyMemory
  module public @WriteOnlyMemory(in %clock: !firrtl.clock, in %addr: !firrtl.uint<4>, in %indata: !firrtl.uint<42>) {

    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %Memory_write = mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["write"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>

    // CHECK-NOT: mem
    %10 = subfield %Memory_write[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %10, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %11 = subfield %Memory_write[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %11, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %12 = subfield %Memory_write[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %12, %clock : !firrtl.clock, !firrtl.clock
    %13 = subfield %Memory_write[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %13, %indata : !firrtl.uint<42>, !firrtl.uint<42>
    %14 = subfield %Memory_write[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %14, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "ReadWriteToWrite" {
  module public @ReadWriteToWrite(in %clock: !firrtl.clock, in %addr: !firrtl.uint<4>, in %indata: !firrtl.uint<42>, out %result: !firrtl.uint<42>) {

    %c1_ui1 = constant 1 : !firrtl.uint<1>

    // CHECK: %Memory_rw, %Memory_r = mem  Undefined
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>

    %Memory_rw, %Memory_r = mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["rw", "r"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } :
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>

    // CHECK: [[ADDR:%.+]] = subfield %Memory_rw[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    // CHECK: [[END:%.+]] = subfield %Memory_rw[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    // CHECK: [[CLK:%.+]] = subfield %Memory_rw[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    // CHECK: [[DATA:%.+]] = subfield %Memory_rw[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    // CHECK: [[MASK:%.+]] = subfield %Memory_rw[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    // CHECK: [[DUMMY_WMODE:%.+]] = wire : !firrtl.uint<1>
    // CHECK: strictconnect [[ADDR]], %addr : !firrtl.uint<4>
    // CHECK: strictconnect [[END]], %c1_ui1 : !firrtl.uint<1>
    // CHECK: strictconnect [[CLK]], %clock : !firrtl.clock
    // CHECK: strictconnect [[DUMMY_WMODE]], %c1_ui1 : !firrtl.uint<1>
    // CHECK: strictconnect [[DATA]], %indata : !firrtl.uint<42>
    // CHECK: strictconnect [[MASK]], %c1_ui1 : !firrtl.uint<1>

    %0 = subfield %Memory_rw[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %0, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %1 = subfield %Memory_rw[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %1, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = subfield %Memory_rw[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %2, %clock : !firrtl.clock, !firrtl.clock
    %3 = subfield %Memory_rw[wmode] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %3, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %4 = subfield %Memory_rw[wdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %4, %indata : !firrtl.uint<42>, !firrtl.uint<42>
    %5 = subfield %Memory_rw[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %5, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    %6 = subfield %Memory_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %6, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %7 = subfield %Memory_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %7, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %8 = subfield %Memory_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %8, %clock : !firrtl.clock, !firrtl.clock
    %9 = subfield %Memory_r[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %result, %9 : !firrtl.uint<42>, !firrtl.uint<42>
  }
}

// -----

firrtl.circuit "UnusedPorts" {
  module public @UnusedPorts(
      in %clock: !firrtl.clock,
      in %addr: !firrtl.uint<4>,
      in %in_data: !firrtl.uint<42>,
      in %wmode_rw: !firrtl.uint<1>,
      out %result_read: !firrtl.uint<42>,
      out %result_rw: !firrtl.uint<42>,
      out %result_pinned: !firrtl.uint<42>) {

    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %c1_ui1 = constant 1 : !firrtl.uint<1>

    // CHECK: %Memory_pinned = mem  Undefined
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>

    %Memory_read, %Memory_rw, %Memory_write, %Memory_pinned = mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["read", "rw", "write", "pinned"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } :
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>

    // CHECK: [[REG1:%.+]] = reg %c0_clock : !firrtl.clock, !firrtl.uint<42>
    // CHECK: [[REG2:%.+]] = reg %c0_clock : !firrtl.clock, !firrtl.uint<42>
    // CHECK: strictconnect %result_read, [[REG1]] : !firrtl.uint<42>
    %read_addr = subfield %Memory_read[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %read_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %read_en = subfield %Memory_read[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %read_en, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %read_clk = subfield %Memory_read[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %read_clk, %clock : !firrtl.clock, !firrtl.clock
    %read_data = subfield %Memory_read[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %result_read, %read_data : !firrtl.uint<42>, !firrtl.uint<42>

    // CHECK: strictconnect %result_rw, [[REG2]] : !firrtl.uint<42>
    %rw_addr = subfield %Memory_rw[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %rw_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %rw_en = subfield %Memory_rw[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %rw_en, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %rw_clk = subfield %Memory_rw[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %rw_clk, %clock : !firrtl.clock, !firrtl.clock
    %rw_rdata = subfield %Memory_rw[rdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %result_rw, %rw_rdata : !firrtl.uint<42>, !firrtl.uint<42>
    %rw_wmode = subfield %Memory_rw[wmode] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %rw_wmode, %wmode_rw : !firrtl.uint<1>, !firrtl.uint<1>
    %rw_wdata = subfield %Memory_rw[wdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %rw_wdata, %in_data : !firrtl.uint<42>, !firrtl.uint<42>
    %rw_wmask = subfield %Memory_rw[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %rw_wmask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    %write_addr = subfield %Memory_write[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %write_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %write_en = subfield %Memory_write[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %write_en, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %write_clk = subfield %Memory_write[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %write_clk, %clock : !firrtl.clock, !firrtl.clock
    %write_data = subfield %Memory_write[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %write_data, %in_data : !firrtl.uint<42>, !firrtl.uint<42>
    %write_mask = subfield %Memory_write[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %write_mask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    %pinned_addr = subfield %Memory_pinned[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %pinned_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %pinned_en = subfield %Memory_pinned[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %pinned_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %pinned_clk = subfield %Memory_pinned[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %pinned_clk, %clock : !firrtl.clock, !firrtl.clock
    %pinned_rdata = subfield %Memory_pinned[rdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %result_pinned, %pinned_rdata : !firrtl.uint<42>, !firrtl.uint<42>
    %pinned_wmode = subfield %Memory_pinned[wmode] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %pinned_wmode, %wmode_rw : !firrtl.uint<1>, !firrtl.uint<1>
    %pinned_wdata = subfield %Memory_pinned[wdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %pinned_wdata, %in_data : !firrtl.uint<42>, !firrtl.uint<42>
    %pinned_wmask = subfield %Memory_pinned[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %pinned_wmask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "UnusedBits" {
  module public @UnusedBits(
      in %clock: !firrtl.clock,
      in %addr: !firrtl.uint<4>,
      in %in_data: !firrtl.uint<42>,
      in %wmode_rw: !firrtl.uint<1>,
      out %result_read: !firrtl.uint<5>,
      out %result_rw: !firrtl.uint<5>) {

    %c1_ui1 = constant 1 : !firrtl.uint<1>

    // CHECK: %Memory_read, %Memory_rw, %Memory_write = mem Undefined
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<10>>
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<10>, wmode: uint<1>, wdata: uint<10>, wmask: uint<1>>
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<10>, mask: uint<1>>
    %Memory_read, %Memory_rw, %Memory_write = mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["read", "rw", "write"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } :
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>

    %read_addr = subfield %Memory_read[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %read_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %read_en = subfield %Memory_read[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %read_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %read_clk = subfield %Memory_read[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %read_clk, %clock : !firrtl.clock, !firrtl.clock
    %read_data = subfield %Memory_read[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %read_data_slice = bits %read_data 7 to 3 : (!firrtl.uint<42>) -> !firrtl.uint<5>
    connect %result_read, %read_data_slice : !firrtl.uint<5>, !firrtl.uint<5>

    // CHECK-DAG: [[RW_FIELD:%.+]] = subfield %Memory_rw[wdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<10>, wmode: uint<1>, wdata: uint<10>, wmask: uint<1>>
    // CHECK-DAG: [[RW_SLICE_LO:%.+]] = bits %in_data 7 to 3 : (!firrtl.uint<42>) -> !firrtl.uint<5>
    // CHECK-DAG: [[RW_SLICE_HI:%.+]] = bits %in_data 24 to 20 : (!firrtl.uint<42>) -> !firrtl.uint<5>
    // CHECK-DAG: [[RW_SLICE_JOIN:%.+]] = cat [[RW_SLICE_HI]], [[RW_SLICE_LO]] : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<10>
    // CHECK-DAG: strictconnect [[RW_FIELD]], [[RW_SLICE_JOIN]] : !firrtl.uint<10>

    // CHECK-DAG: [[W_FIELD:%.+]] = subfield %Memory_write[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<10>, mask: uint<1>>
    // CHECK-DAG: [[W_SLICE_LO:%.+]] = bits %in_data 7 to 3 : (!firrtl.uint<42>) -> !firrtl.uint<5>
    // CHECK-DAG: [[W_SLICE_HI:%.+]] = bits %in_data 24 to 20 : (!firrtl.uint<42>) -> !firrtl.uint<5>
    // CHECK-DAG: [[W_SLICE_JOIN:%.+]] = cat [[W_SLICE_HI]], [[W_SLICE_LO]] : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<10>
    // CHECK-DAG: strictconnect [[W_FIELD]], [[W_SLICE_JOIN]] : !firrtl.uint<10>

    %rw_addr = subfield %Memory_rw[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %rw_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %rw_en = subfield %Memory_rw[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %rw_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %rw_clk = subfield %Memory_rw[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %rw_clk, %clock : !firrtl.clock, !firrtl.clock
    %rw_rdata = subfield %Memory_rw[rdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    %rw_rdata_slice = bits %rw_rdata 24 to 20 : (!firrtl.uint<42>) -> !firrtl.uint<5>
    connect %result_rw, %rw_rdata_slice : !firrtl.uint<5>, !firrtl.uint<5>
    %rw_wmode = subfield %Memory_rw[wmode] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %rw_wmode, %wmode_rw : !firrtl.uint<1>, !firrtl.uint<1>
    %rw_wdata = subfield %Memory_rw[wdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %rw_wdata, %in_data : !firrtl.uint<42>, !firrtl.uint<42>
    %rw_wmask = subfield %Memory_rw[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %rw_wmask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>


    %write_addr = subfield %Memory_write[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %write_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %write_en = subfield %Memory_write[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %write_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %write_clk = subfield %Memory_write[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %write_clk, %clock : !firrtl.clock, !firrtl.clock
    %write_data = subfield %Memory_write[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %write_data, %in_data : !firrtl.uint<42>, !firrtl.uint<42>
    %write_mask = subfield %Memory_write[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %write_mask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "UnusedBitsAtEnd" {
  module public @UnusedBitsAtEnd(
      in %clock: !firrtl.clock,
      in %addr: !firrtl.uint<4>,
      in %in_data: !firrtl.uint<42>,
      out %result_read: !firrtl.uint<5>) {

    %c1_ui1 = constant 1 : !firrtl.uint<1>

    // CHECK: %Memory_read, %Memory_write = mem Undefined
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<5>>
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>
    %Memory_read, %Memory_write = mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["read", "write"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } :
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>

    // CHECK: [[RDATA:%.+]] = subfield %Memory_read[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<5>>
    // CHECK: strictconnect %result_read, [[RDATA]] : !firrtl.uint<5>
    %read_addr = subfield %Memory_read[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %read_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %read_en = subfield %Memory_read[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %read_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %read_clk = subfield %Memory_read[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %read_clk, %clock : !firrtl.clock, !firrtl.clock
    %read_data = subfield %Memory_read[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %read_data_slice = bits %read_data 41 to 37 : (!firrtl.uint<42>) -> !firrtl.uint<5>
    connect %result_read, %read_data_slice : !firrtl.uint<5>, !firrtl.uint<5>

    // CHECK: bits %in_data 41 to 37 : (!firrtl.uint<42>) -> !firrtl.uint<5>
    %write_addr = subfield %Memory_write[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %write_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %write_en = subfield %Memory_write[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %write_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %write_clk = subfield %Memory_write[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %write_clk, %clock : !firrtl.clock, !firrtl.clock
    %write_data = subfield %Memory_write[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %write_data, %in_data : !firrtl.uint<42>, !firrtl.uint<42>
    %write_mask = subfield %Memory_write[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %write_mask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "OneAddressMasked" {
  module public @OneAddressMasked(
      in %clock: !firrtl.clock,
      in %addr: !firrtl.uint<1>,
      in %in_data: !firrtl.uint<32>,
      in %in_mask: !firrtl.uint<2>,
      in %in_wen: !firrtl.uint<1>,
      out %result_read: !firrtl.uint<32>) {


    %c1_ui1 = constant 1 : !firrtl.uint<1>

    %Memory_read, %Memory_write = mem Undefined
      {
        depth = 1 : i64,
        name = "Memory",
        portNames = ["read", "write"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } :
        !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>,
        !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<2>>
    // CHECK: %Memory = reg %clock : !firrtl.clock, !firrtl.uint<32>

    // CHECK: strictconnect %result_read, %Memory : !firrtl.uint<32>

    %read_addr = subfield %Memory_read[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    connect %read_addr, %addr : !firrtl.uint<1>, !firrtl.uint<1>
    %read_en = subfield %Memory_read[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    connect %read_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %read_clk = subfield %Memory_read[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    connect %read_clk, %clock : !firrtl.clock, !firrtl.clock
    %read_data = subfield %Memory_read[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    connect %result_read, %read_data : !firrtl.uint<32>, !firrtl.uint<32>

    // CHECK: [[DATA_0:%.+]] = bits %in_data 15 to 0 : (!firrtl.uint<32>) -> !firrtl.uint<16>
    // CHECK: [[NEXT_0:%.+]] = bits %Memory 15 to 0 : (!firrtl.uint<32>) -> !firrtl.uint<16>
    // CHECK: [[MASK_0:%.+]] = bits %in_mask 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
    // CHECK: [[CHUNK_0:%.+]] = mux([[MASK_0]], [[DATA_0]], [[NEXT_0]]) : (!firrtl.uint<1>, !firrtl.uint<16>, !firrtl.uint<16>) -> !firrtl.uint<16>
    // CHECK: [[DATA_1:%.+]] = bits %in_data 31 to 16 : (!firrtl.uint<32>) -> !firrtl.uint<16>
    // CHECK: [[NEXT_1:%.+]] = bits %Memory 31 to 16 : (!firrtl.uint<32>) -> !firrtl.uint<16>
    // CHECK: [[MASK_1:%.+]] = bits %in_mask 1 to 1 : (!firrtl.uint<2>) -> !firrtl.uint<1>
    // CHECK: [[CHUNK_1:%.+]] = mux([[MASK_1]], [[DATA_1]], [[NEXT_1]]) : (!firrtl.uint<1>, !firrtl.uint<16>, !firrtl.uint<16>) -> !firrtl.uint<16>
    // CHECK: [[NEXT:%.+]] = cat [[CHUNK_1]], [[CHUNK_0]] : (!firrtl.uint<16>, !firrtl.uint<16>) -> !firrtl.uint<32>
    // CHECK: [[NEXT_EN:%.+]] = mux(%in_wen, [[NEXT]], %Memory) : (!firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>
    // CHECK: strictconnect %Memory, [[NEXT_EN]] : !firrtl.uint<32>

    %write_addr = subfield %Memory_write[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<2>>
    connect %write_addr, %addr : !firrtl.uint<1>, !firrtl.uint<1>
    %write_en = subfield %Memory_write[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<2>>
    connect %write_en, %in_wen : !firrtl.uint<1>, !firrtl.uint<1>
    %write_clk = subfield %Memory_write[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<2>>
    connect %write_clk, %clock : !firrtl.clock, !firrtl.clock
    %write_data = subfield %Memory_write[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<2>>
    connect %write_data, %in_data : !firrtl.uint<32>, !firrtl.uint<32>
    %write_mask = subfield %Memory_write[mask] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<2>>
    connect %write_mask, %in_mask : !firrtl.uint<2>, !firrtl.uint<2>
  }
}

// -----

firrtl.circuit "OneAddressNoMask" {
  module public @OneAddressNoMask(
      in %clock: !firrtl.clock,
      in %addr: !firrtl.uint<1>,
      in %in_data: !firrtl.uint<32>,
      in %wmode_rw: !firrtl.uint<1>,
      in %in_wen: !firrtl.uint<1>,
      in %in_rwen: !firrtl.uint<1>,
      out %result_read: !firrtl.uint<32>,
      out %result_rw: !firrtl.uint<32>) {

    // Pipeline the inputs.
    // TODO: It would be good to de-duplicate these either in the pass or in a canonicalizer.

    // CHECK: %Memory_write_en_0 = reg %clock : !firrtl.clock, !firrtl.uint<1>
    // CHECK: strictconnect %Memory_write_en_0, %in_wen : !firrtl.uint<1>
    // CHECK: %Memory_write_en_1 = reg %clock : !firrtl.clock, !firrtl.uint<1>
    // CHECK: strictconnect %Memory_write_en_1, %Memory_write_en_0 : !firrtl.uint<1>
    // CHECK: %Memory_write_en_2 = reg %clock : !firrtl.clock, !firrtl.uint<1>
    // CHECK: strictconnect %Memory_write_en_2, %Memory_write_en_1 : !firrtl.uint<1>

    // CHECK: %Memory_write_data_0 = reg %clock : !firrtl.clock, !firrtl.uint<32>
    // CHECK: strictconnect %Memory_write_data_0, %in_data : !firrtl.uint<32>
    // CHECK: %Memory_write_data_1 = reg %clock : !firrtl.clock, !firrtl.uint<32>
    // CHECK: strictconnect %Memory_write_data_1, %Memory_write_data_0 : !firrtl.uint<32>
    // CHECK: %Memory_write_data_2 = reg %clock : !firrtl.clock, !firrtl.uint<32>
    // CHECK: strictconnect %Memory_write_data_2, %Memory_write_data_1 : !firrtl.uint<32>

    // CHECK: %Memory_rw_wdata_0 = reg %clock : !firrtl.clock, !firrtl.uint<32>
    // CHECK: strictconnect %Memory_rw_wdata_0, %in_data : !firrtl.uint<32>
    // CHECK: %Memory_rw_wdata_1 = reg %clock : !firrtl.clock, !firrtl.uint<32>
    // CHECK: strictconnect %Memory_rw_wdata_1, %Memory_rw_wdata_0 : !firrtl.uint<32>
    // CHECK: %Memory_rw_wdata_2 = reg %clock : !firrtl.clock, !firrtl.uint<32>
    // CHECK: strictconnect %Memory_rw_wdata_2, %Memory_rw_wdata_1 : !firrtl.uint<32>

    %c1_ui1 = constant 1 : !firrtl.uint<1>

    %Memory_read, %Memory_rw, %Memory_write = mem Undefined
      {
        depth = 1 : i64,
        name = "Memory",
        portNames = ["read", "rw", "write"],
        readLatency = 2 : i32,
        writeLatency = 4 : i32
      } :
        !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>,
        !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>,
        !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>

    // CHECK: %Memory = reg %clock : !firrtl.clock, !firrtl.uint<32>

    // CHECK: strictconnect %result_read, %Memory : !firrtl.uint<32>
    %read_addr = subfield %Memory_read[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    connect %read_addr, %addr : !firrtl.uint<1>, !firrtl.uint<1>
    %read_en = subfield %Memory_read[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    connect %read_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %read_clk = subfield %Memory_read[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    connect %read_clk, %clock : !firrtl.clock, !firrtl.clock
    %read_data = subfield %Memory_read[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    connect %result_read, %read_data : !firrtl.uint<32>, !firrtl.uint<32>

    // CHECK: strictconnect %result_rw, %Memory : !firrtl.uint<32>
    %rw_addr = subfield %Memory_rw[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    connect %rw_addr, %addr : !firrtl.uint<1>, !firrtl.uint<1>
    %rw_en = subfield %Memory_rw[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    connect %rw_en, %in_rwen : !firrtl.uint<1>, !firrtl.uint<1>
    %rw_clk = subfield %Memory_rw[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    connect %rw_clk, %clock : !firrtl.clock, !firrtl.clock
    %rw_rdata = subfield %Memory_rw[rdata] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    connect %result_rw, %rw_rdata : !firrtl.uint<32>, !firrtl.uint<32>
    %rw_wmode = subfield %Memory_rw[wmode] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    connect %rw_wmode, %wmode_rw : !firrtl.uint<1>, !firrtl.uint<1>
    %rw_wdata = subfield %Memory_rw[wdata] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    connect %rw_wdata, %in_data : !firrtl.uint<32>, !firrtl.uint<32>
    %rw_wmask = subfield %Memory_rw[wmask] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    connect %rw_wmask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK: [[WRITING:%.+]] = and %in_rwen, %wmode_rw : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK: %Memory_rw_wen_0 = reg %clock : !firrtl.clock, !firrtl.uint<1>
    // CHECK: strictconnect %Memory_rw_wen_0, [[WRITING]] : !firrtl.uint<1>
    // CHECK: %Memory_rw_wen_1 = reg %clock : !firrtl.clock, !firrtl.uint<1>
    // CHECK: strictconnect %Memory_rw_wen_1, %Memory_rw_wen_0 : !firrtl.uint<1>
    // CHECK: %Memory_rw_wen_2 = reg %clock : !firrtl.clock, !firrtl.uint<1>
    // CHECK: strictconnect %Memory_rw_wen_2, %Memory_rw_wen_1 : !firrtl.uint<1>
    // CHECK: [[WRITE_RW:%.+]] = mux(%Memory_rw_wen_2, %Memory_rw_wdata_2, %Memory) : (!firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>
    // CHECK: [[WRITE_W:%.+]] = mux(%Memory_write_en_2, %Memory_write_data_2, [[WRITE_RW]]) : (!firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>
    // CHECK: strictconnect %Memory, [[WRITE_W]] : !firrtl.uint<32>
    %write_addr = subfield %Memory_write[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    connect %write_addr, %addr : !firrtl.uint<1>, !firrtl.uint<1>
    %write_en = subfield %Memory_write[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    connect %write_en, %in_wen : !firrtl.uint<1>, !firrtl.uint<1>
    %write_clk = subfield %Memory_write[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    connect %write_clk, %clock : !firrtl.clock, !firrtl.clock
    %write_data = subfield %Memory_write[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    connect %write_data, %in_data : !firrtl.uint<32>, !firrtl.uint<32>
    %write_mask = subfield %Memory_write[mask] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    connect %write_mask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}
