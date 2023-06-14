// RUN: circt-opt -firrtl-lower-memory %s | FileCheck %s

// Test basic lowering of the three port types.
// CHECK-LABEL: circuit "ReadWrite" {
firrtl.circuit "ReadWrite" {
firrtl.module @ReadWrite() {
  %MReadWrite_readwrite = mem Undefined {depth = 12 : i64, name = "MReadWrite", portNames = ["readwrite"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
  // CHECK: instance MReadWrite  @MReadWrite(in RW0_addr: !firrtl.uint<4>, in RW0_en: !firrtl.uint<1>, in RW0_clk: !firrtl.clock, in RW0_wmode: !firrtl.uint<1>, in RW0_wdata: !firrtl.uint<42>, out RW0_rdata: !firrtl.uint<42>)
}
// CHECK: module private @MReadWrite
// CHECK:   instance MReadWrite_ext  @MReadWrite_ext
// CHECK:   strictconnect %MReadWrite_ext_RW0_addr, %RW0_addr
// CHECK:   strictconnect %RW0_rdata, %MReadWrite_ext_RW0_rdata
// CHECK: }
}

// CHECK-LABEL: circuit "Write"
firrtl.circuit "Write" {
firrtl.module @Write() {
  %MWrite_write = mem Undefined {depth = 12 : i64, name = "MWrite", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  // CHECK: instance MWrite  @MWrite(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
}
}

// SeqMems need at least 1 write port, but this is primarily testing the Read
// port.
firrtl.circuit "Read" {
firrtl.module @Read() {
  %MRead_read, %MRead_write = mem Undefined {depth = 12 : i64, name = "MRead", portNames = ["read", "write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  // CHECK: instance MRead  @MRead(in R0_addr: !firrtl.uint<4>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, out R0_data: !firrtl.uint<42>, in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
}
// CHECK: memmodule private @MRead_ext
// CHECK-SAME: {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
}

// Test that the memory wrapper module is renamed when there is a collision.
// CHECK-LABEL: circuit "Collision0"
firrtl.circuit "Collision0" {
// @test collides with the name of the wrapper module.
firrtl.extmodule @test()
firrtl.module @Collision0() {
  %MWrite_write = mem Undefined {depth = 12 : i64, name = "test", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  // CHECK: instance test  @test_0
}
// CHECK: module private @test_0
// CHECK-NEXT: instance test_ext  @test_ext
}

// Test that the memory module is renamed when there is a collision.
// CHECK-LABEL: circuit "Collision1"
firrtl.circuit "Collision1" {
// @test_ext collides with the name of the external memory module.
firrtl.extmodule @test_ext()
firrtl.module @Collision1() {
  %MWrite_write = mem Undefined {depth = 12 : i64, name = "test", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  // CHECK: instance test @test
}
// CHECK: module private @test
// CHECK-NEXT: instance test_0_ext @test_0_ext

// CHECK: memmodule private @test_0_ext
}

// Test that the memory modules are deduplicated.
// CHECK-LABEL: circuit "Dedup"
firrtl.circuit "Dedup" {
firrtl.module @Dedup() {
  %mem0_write = mem Undefined {depth = 12 : i64, name = "mem0", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  %mem1_write = mem Undefined {depth = 12 : i64, name = "mem1", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  // CHECK: instance mem0  @mem0(
  // CHECK: instance mem1  @mem1(
}
// CHECK: module private @mem0
// CHECK-NEXT: instance mem0_ext  @mem0_ext

// CHECK: memmodule private @mem0_ext

// CHECK: module private @mem1
// CHECK-NEXT: instance mem0_ext @mem0_ext
}

// Test that memories in the testharness are not deduped with other memories in
// the test harness.
// CHECK-LABEL: circuit "NoTestharnessDedup0"
firrtl.circuit "NoTestharnessDedup0" {
firrtl.module @NoTestharnessDedup0() {
  %mem0_write = mem Undefined {depth = 12 : i64, name = "mem0", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  // CHECK: instance mem0  @mem0
  %mem1_write = mem Undefined {depth = 12 :i64, name = "mem1", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  // CHECK: instance mem1  @mem1
  instance dut @DUT()
}
firrtl.module @DUT() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} { }

// CHECK: module private @mem0
// CHECK-NEXT: instance mem0_ext  @mem0_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)

// CHECK: memmodule private @mem0_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}

// CHECK: module private @mem1
// CHECK-NEXT: instance mem1_ext  @mem1_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)

// CHECK: memmodule private @mem1_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
}

// Test that memories in the testharness are not deduped with other memories in the DUT.
// CHECK-LABEL: circuit "NoTestharnessDedup1"
firrtl.circuit "NoTestharnessDedup1" {
firrtl.module @NoTestharnessDedup1() {
  %mem0_write = mem Undefined {depth = 12 : i64, name = "mem0", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  // CHECK: instance mem0  @mem0
  instance dut @DUT()
}
firrtl.module @DUT() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
  %mem1_write = mem Undefined {depth = 12 :i64, name = "mem1", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
  // CHECK: instance mem1  @mem1
}

// CHECK: module private @mem0
// CHECK-NEXT: instance mem0_ext  @mem0_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)

// CHECK: memmodule private @mem0_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}

// CHECK: module private @mem1
// CHECK-NEXT: instance mem1_ext  @mem1_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)

// CHECK: memmodule private @mem1_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
}

// Check that when the mask is 1-bit, it is removed from the memory and the
// enable signal is and'd with the mask signal.
// CHECK-LABEL: circuit "NoMask"
firrtl.circuit "NoMask" {
  module @NoMask(in %en: !firrtl.uint<1>, in %mask: !firrtl.uint<1>) {
    %MemSimple_read, %MemSimple_write = mem Undefined {depth = 12 : i64, name = "MemSimple", portNames = ["read", "write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>

    // Enable:
    %0 = subfield %MemSimple_write[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %0, %en : !firrtl.uint<1>, !firrtl.uint<1>

    // Mask:
    %1 = subfield %MemSimple_write[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %1, %mask : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK: [[AND:%.+]] = and %mask, %en
    // CHECK: connect %MemSimple_W0_en, [[AND]]
  }
}

// Check a memory with a mask gets lowered properly.
// CHECK-LABEL: circuit "YesMask"
firrtl.circuit "YesMask" {
  module @YesMask(in %en: !firrtl.uint<1>, in %mask: !firrtl.uint<4>) {
    %MemSimple_read, %MemSimple_write = mem Undefined {depth = 1022 : i64, name = "MemSimple", portNames = ["read", "write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data flip: uint<40>>, !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: uint<40>, mask: uint<4>>

    // Enable:
    %0 = subfield %MemSimple_write[en] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: uint<40>, mask: uint<4>>
    connect %0, %en : !firrtl.uint<1>, !firrtl.uint<1>

    // Mask:
    %1 = subfield %MemSimple_write[mask] : !firrtl.bundle<addr: uint<10>, en: uint<1>, clk: clock, data: uint<40>, mask: uint<4>>
    connect %1, %mask : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK: connect %MemSimple_W0_en, %en
    // CHECK: connect %MemSimple_W0_mask, %mask
  }
}

// CHECK-LABEL: circuit "MemDepth1"
firrtl.circuit "MemDepth1" {
  module @MemDepth1(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>,
                           in %addr: !firrtl.uint<1>, in %data: !firrtl.uint<32>) {
    // CHECK: instance mem0  @mem0(in W0_addr: !firrtl.uint<1>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<32>, in W0_mask: !firrtl.uint<4>)
    // CHECK: connect %mem0_W0_data, %data : !firrtl.uint<32>, !firrtl.uint<32>
    %mem0_write = mem Old {depth = 1 : i64, name = "mem0", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<4>>
    %1 = subfield %mem0_write[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<4>>
    connect %1, %addr : !firrtl.uint<1>, !firrtl.uint<1>
    %3 = subfield %mem0_write[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<4>>
    connect %3, %en : !firrtl.uint<1>, !firrtl.uint<1>
    %0 = subfield %mem0_write[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<4>>
    connect %0, %clock : !firrtl.clock, !firrtl.clock
    %2 = subfield %mem0_write[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<4>>
    connect %2, %data : !firrtl.uint<32>, !firrtl.uint<32>
}
// CHECK: memmodule private @mem0_ext
// CHECK-SAME: depth = 1
}

// CHECK-LABEL: circuit "inferUnmaskedMemory"
firrtl.circuit "inferUnmaskedMemory" {
  module @inferUnmaskedMemory(in %clock: !firrtl.clock, in %rAddr: !firrtl.uint<4>, in %rEn: !firrtl.uint<1>, out %rData: !firrtl.uint<8>, in %wMode: !firrtl.uint<1>, in %wMask: !firrtl.uint<1>, in %wData: !firrtl.uint<8>) {
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
    // CHECK: [[AND:%.+]] = and %wMask, %rEn
    // CHECK: connect %tbMemoryKind1_W0_en, %0
    %MReadWrite_readwrite = mem Undefined {depth = 12 : i64, name = "MReadWrite", portNames = ["readwrite"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    %rw_en = subfield %MReadWrite_readwrite[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    %rw_wmode = subfield %MReadWrite_readwrite[wmode] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    %rw_mask = subfield %MReadWrite_readwrite[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    connect %rw_en, %rEn : !firrtl.uint<1>, !firrtl.uint<1>
    connect %rw_wmode, %wMode : !firrtl.uint<1>, !firrtl.uint<1>
    connect %rw_mask, %wMask : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:  %[[MReadWrite_RW0_addr:.+]], %[[MReadWrite_RW0_en:.+]], %[[MReadWrite_RW0_clk:.+]], %[[MReadWrite_RW0_wmode:.+]], %[[MReadWrite_RW0_wdata:.+]], %[[MReadWrite_RW0_rdata:.+]] = instance MReadWrite  @MReadWrite
    // CHECK:   connect %[[MReadWrite_RW0_en]], %rEn : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:   %1 = and %wMask, %wMode : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK:   connect %[[MReadWrite_RW0_wmode]], %1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// Check that annotations are copied over to the instance.
// CHECK-LABEL: circuit "Annotations"
firrtl.circuit "Annotations" {
firrtl.module @Annotations() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
  // No annotations copied to this instance.
  // CHECK: irrtl.instance mem0  @mem0
  %mem0_write = mem Undefined {annotations = [{class = "test"}], depth = 12 : i64, name = "mem0", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
}
// CHECK: module private @mem0
// CHECK-NEXT: instance mem0_ext  {annotations = [{class = "test"}]} @mem0_ex
}

// Check that annotations are copied over to the instance.
// CHECK-LABEL: circuit "NonLocalAnnotation"
firrtl.circuit "NonLocalAnnotation" {

// CHECK:  hw.hierpath private @[[nla_0:.+]] [@NonLocalAnnotation::@dut, @DUT::@mem0, @mem0]
hw.hierpath private @nla0 [@NonLocalAnnotation::@dut, @DUT::@mem0]
// CHECK:  hw.hierpath private @[[nla_1:.+]] [@NonLocalAnnotation::@dut, @DUT::@mem1, @mem1]
hw.hierpath private @nla1 [@NonLocalAnnotation::@dut, @DUT]

// CHECK: module @NonLocalAnnotation()
firrtl.module @NonLocalAnnotation()  {
  instance dut sym @dut @DUT()
}
// CHECK: module @DUT()
firrtl.module @DUT() {
  // This memory has a symbol and an NLA directly targetting it.
  // CHECK: instance mem0 sym @mem0 @mem0
  %mem0_write = mem sym @mem0 Undefined {annotations = [{circt.nonlocal = @nla0, class = "test0"}], depth = 12 : i64, name = "mem0", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>

  // This memory does not have a symbol already attached.
  // CHECK: instance mem1 sym @mem1 @mem1
  %mem1_write = mem Undefined {annotations = [{circt.nonlocal = @nla1, class = "test1"}], depth = 12 : i64, name = "mem1", portNames = ["write"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>

// LowerMemory should ignore MemOps that are not seqmems. The following memory is a combmem with readLatency=0.
  %MRead_read = mem Undefined {depth = 12 : i64, name = "MRead", portNames = ["read"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
// CHECK:   %MRead_read = mem Undefined
}

// CHECK: module private @mem0
// CHECK:   instance mem0_ext sym @mem0_ext
// CHECK-SAME: {annotations = [{circt.nonlocal = @[[nla_0]], class = "test0"}]}
// CHECK-SAME:  @mem0_ext(
// CHECK: }

// CHECK: module private @mem1
// CHECK:   instance mem0_ext sym @mem0_ext
// CHECK-SAME:  {annotations = [{circt.nonlocal = @[[nla_1]], class = "test1"}]}
// CHECK-SAME:  @mem0_ext(
}
