// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-flatten-memory)))' %s | FileCheck  %s


firrtl.circuit "Mem" {
  module public  @Mem(in %clock: !firrtl.clock, in %rAddr: !firrtl.uint<4>, in %rEn: !firrtl.uint<1>, out %rData: !firrtl.bundle<a: uint<8>, b: uint<8>>, in %wAddr: !firrtl.uint<4>, in %wEn: !firrtl.uint<1>, in %wMask: !firrtl.bundle<a: uint<1>, b: uint<1>>, in %wData: !firrtl.bundle<a: uint<8>, b: uint<8>>) {
    %memory_r, %memory_w = mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r", "w"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    %0 = subfield %memory_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    strictconnect %0, %clock : !firrtl.clock
    %1 = subfield %memory_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    strictconnect %1, %rEn : !firrtl.uint<1>
    %2 = subfield %memory_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    strictconnect %2, %rAddr : !firrtl.uint<4>
    %3 = subfield %memory_r[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    strictconnect %rData, %3 : !firrtl.bundle<a: uint<8>, b: uint<8>>
    %4 = subfield %memory_w[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    strictconnect %4, %clock : !firrtl.clock
    %5 = subfield %memory_w[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    strictconnect %5, %wEn : !firrtl.uint<1>
    %6 = subfield %memory_w[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    strictconnect %6, %wAddr : !firrtl.uint<4>
    %7 = subfield %memory_w[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    strictconnect %7, %wMask : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %8 = subfield %memory_w[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    strictconnect %8, %wData : !firrtl.bundle<a: uint<8>, b: uint<8>>
    // ---------------------------------------------------------------------------------
    // After flattenning the memory data
    // CHECK: %[[memory_r:.+]], %[[memory_w:.+]] = mem Undefined  {depth = 16 : i64, name = "memory", portNames = ["r", "w"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32}
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<16>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<16>, mask: uint<2>>
    // CHECK: %[[memory_r_0:.+]] = wire  {name = "memory_r"} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    // CHECK: %[[v0:.+]] = subfield %[[memory_r]][addr]
    // CHECK: strictconnect %[[v0]], %[[memory_r_addr:.+]] :
    // CHECK: %[[v1:.+]] = subfield %[[memory_r]][en]
    // CHECK: strictconnect %[[v1]], %[[memory_r_en:.+]] :
    // CHECK: %[[v2:.+]] = subfield %[[memory_r]][clk]
    // CHECK: strictconnect %[[v2]], %[[memory_r_clk:.+]] :
    // CHECK: %[[v3:.+]] = subfield %[[memory_r]][data]
    //
    // ---------------------------------------------------------------------------------
    // Read ports
    // CHECK:  %[[v4:.+]] = bitcast %[[v3]] : (!firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<8>>
    // CHECK:  strictconnect %[[memory_r_data:.+]], %[[v4]] :
    // --------------------------------------------------------------------------------
    // Write Ports
    // CHECK:  %[[memory_w_1:.+]] = wire  {name = "memory_w"} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // CHECK:  %[[v9:.+]] = subfield %[[memory_w]][data]
    // CHECK:  %[[v17:.+]] = bitcast %[[v15:.+]] : (!firrtl.bundle<a: uint<8>, b: uint<8>>) -> !firrtl.uint<16>
    // CHECK:  strictconnect %[[v9]], %[[v17]]
    //
    // --------------------------------------------------------------------------------
    // Mask Ports
    //  CHECK: %[[v11:.+]] = subfield %[[memory_w]][mask]
    //  CHECK: %[[v12:.+]] = bitcast %[[v18:.+]] : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<2>
    //  CHECK: strictconnect %[[v11]], %[[v12]]
    // --------------------------------------------------------------------------------
    // Connections to module ports
    // CHECK:  %[[v21:.+]] = subfield %[[memory_r_0]][clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    // CHECK:  strictconnect %[[v21]], %clock :
    // CHECK:  %[[v22:.+]]  = subfield %[[memory_r_0]][en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    // CHECK:  strictconnect %[[v22]], %rEn : !firrtl.uint<1>
    // CHECK:  %[[v23:.+]]  = subfield %[[memory_r_0]][addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    // CHECK:  strictconnect %[[v23]], %rAddr : !firrtl.uint<4>
    // CHECK:  %[[v24:.+]]  = subfield %[[memory_r_0]][data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
    // CHECK:  strictconnect %rData, %[[v24]] : !firrtl.bundle<a: uint<8>, b: uint<8>>
    // CHECK:  %[[v25:.+]]  = subfield %[[memory_w_1]][clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // CHECK:  strictconnect %[[v25]], %clock : !firrtl.clock
    // CHECK:  %[[v26:.+]]  = subfield %[[memory_w_1]][en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // CHECK:  strictconnect %[[v26]], %wEn : !firrtl.uint<1>
    // CHECK:  %[[v27:.+]]  = subfield %[[memory_w_1]][addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // CHECK:  strictconnect %[[v27]], %wAddr : !firrtl.uint<4>
    // CHECK:  %[[v28:.+]]  = subfield %[[memory_w_1]][mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // CHECK:  strictconnect %[[v28]], %wMask : !firrtl.bundle<a: uint<1>, b: uint<1>>
    // CHECK:  %[[v29:.+]]  = subfield %[[memory_w_1]][data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: uint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // CHECK:  strictconnect %[[v29]], %wData : !firrtl.bundle<a: uint<8>, b: uint<8>>
  }

firrtl.module @MemoryRWSplit(in %clock: !firrtl.clock, in %rwEn: !firrtl.uint<1>, in %rwMode: !firrtl.uint<1>, in %rwAddr: !firrtl.uint<4>, in %rwMask: !firrtl.bundle<a: uint<1>, b: uint<1>>, in %rwDataIn: !firrtl.bundle<a: uint<8>, b: uint<9>>, out %rwDataOut: !firrtl.bundle<a: uint<8>, b: uint<9>>) {
  %memory_rw = mem Undefined  {depth = 16 : i64, name = "memory", portNames = ["rw"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>
  // CHECK:  %memory_rw = mem Undefined  {depth = 16 : i64, name = "memory", portNames = ["rw"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<17>, wmode: uint<1>, wdata: uint<17>, wmask: uint<17>>
  // CHECK:  %[[memory_rw_0:.+]] = wire  {name = "memory_rw"} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>
  %0 = subfield %memory_rw[rdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>
  %1 = subfield %memory_rw[wdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>
  %2 = subfield %memory_rw[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>
  %3 = subfield %memory_rw[wmode] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>
  %4 = subfield %memory_rw[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>
  %5 = subfield %memory_rw[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>
  %6 = subfield %memory_rw[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint<8>, b: uint<9>>, wmode: uint<1>, wdata: bundle<a: uint<8>, b: uint<9>>, wmask: bundle<a: uint<1>, b: uint<1>>>
  connect %6, %clock : !firrtl.clock, !firrtl.clock
  connect %5, %rwEn : !firrtl.uint<1>, !firrtl.uint<1>
  connect %4, %rwAddr : !firrtl.uint<4>, !firrtl.uint<4>
  connect %3, %rwMode : !firrtl.uint<1>, !firrtl.uint<1>
  connect %2, %rwMask : !firrtl.bundle<a: uint<1>, b: uint<1>>, !firrtl.bundle<a: uint<1>, b: uint<1>>
  connect %1, %rwDataIn : !firrtl.bundle<a: uint<8>, b: uint<9>>, !firrtl.bundle<a: uint<8>, b: uint<9>>
  connect %rwDataOut, %0 : !firrtl.bundle<a: uint<8>, b: uint<9>>, !firrtl.bundle<a: uint<8>, b: uint<9>>
  // CHECK:  %[[v6:.+]] = subfield %[[memory_rw_0]][rdata] :
  // CHECK:  %[[v7:.+]] = subfield %memory_rw[rdata] :
  // CHECK:  %[[v8:.+]] = bitcast %[[v7]] :
  // CHECK:  strictconnect %[[v6]], %[[v8]] :
  // CHECK:  %[[v9:.+]] = subfield %[[memory_rw_0]][wmode] :
  // CHECK:  %[[v10:.+]] = subfield %memory_rw[wmode] :
  // CHECK:  strictconnect %[[v10]], %[[v9]] : !firrtl.uint<1>
  // CHECK:  %[[v11:.+]] = subfield %[[memory_rw_0]][wdata] :
  // CHECK:  %[[v12:.+]] = subfield %memory_rw[wdata] :
  // CHECK:  %[[v13:.+]] = bitcast %[[v11]] : (!firrtl.bundle<a: uint<8>, b: uint<9>>) -> !firrtl.uint<17>
  // CHECK:  strictconnect %[[v12]], %[[v13]] :
  // CHECK:  %[[v14:.+]] = subfield %[[memory_rw_0]][wmask] :
  // CHECK:  %[[v15:.+]] = subfield %memory_rw[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<17>, wmode: uint<1>, wdata: uint<17>, wmask: uint<17>>
  // CHECK:  %[[v16:.+]] = bitcast %14 : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<2>
  // CHECK:  %[[v17:.+]] = bits %16 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  // CHECK:  %[[v18:.+]] = cat %[[v17]], %[[v17]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK:  %[[v19:.+]] = cat %[[v17]], %[[v18]] : (!firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<3>
  // CHECK:  %[[v24:.+]] = cat %[[v17]], %[[v23:.+]] : (!firrtl.uint<1>, !firrtl.uint<7>) -> !firrtl.uint<8>
  // CHECK:  %[[v25:.+]] = bits %16 1 to 1 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  // CHECK:  %[[v26:.+]] = cat %[[v25]], %[[v24]] : (!firrtl.uint<1>, !firrtl.uint<8>) -> !firrtl.uint<9>
  // CHECK:  %[[v27:.+]] = cat %[[v25]], %[[v26]] : (!firrtl.uint<1>, !firrtl.uint<9>) -> !firrtl.uint<10>
  // CHECK:  %[[v28:.+]] = cat %[[v25]], %[[v27]] : (!firrtl.uint<1>, !firrtl.uint<10>) -> !firrtl.uint<11>
  // CHECK:  %[[v34:.+]] = cat %[[v25]], %[[v33:.+]] : (!firrtl.uint<1>, !firrtl.uint<16>) -> !firrtl.uint<17>
  // CHECK:  strictconnect %[[v15]], %[[v34]] :
  // Ensure 0 bit fields are handled properly.
  %ram_MPORT = mem Undefined  {depth = 4 : i64, name = "ram", portNames = ["MPORT"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data: bundle<entry: bundle<a: uint<0>, b: uint<1>, c: uint<2>>>, mask: bundle<entry: bundle<a: uint<1>, b: uint<1>, c: uint<1>>>>
  // CHECK: %ram_MPORT = mem Undefined  {depth = 4 : i64, name = "ram", portNames = ["MPORT"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data: uint<3>, mask: uint<3>>

}


  module @ZeroBitMasks(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io: !firrtl.bundle<a: uint<0>, b: uint<20>>) {
    %invalid = invalidvalue : !firrtl.bundle<a: uint<1>, b: uint<1>>
    %invalid_0 = invalidvalue : !firrtl.bundle<a: uint<0>, b: uint<20>>
    %ram_MPORT = mem Undefined  {depth = 1 : i64, name = "ram", portNames = ["MPORT"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<20>>, mask: bundle<a: uint<1>, b: uint<1>>>
    %3 = subfield %ram_MPORT[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<20>>, mask: bundle<a: uint<1>, b: uint<1>>>
    strictconnect %3, %invalid_0 : !firrtl.bundle<a: uint<0>, b: uint<20>>
    %4 = subfield %ram_MPORT[mask] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<20>>, mask: bundle<a: uint<1>, b: uint<1>>>
    strictconnect %4, %invalid : !firrtl.bundle<a: uint<1>, b: uint<1>>
    // CHECK:  %ram_MPORT = mem Undefined  {depth = 1 : i64, name = "ram", portNames = ["MPORT"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<20>, mask: uint<1>>
    // CHECK:  %ram_MPORT_1 = wire  {name = "ram_MPORT"} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<20>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // CHECK:  %[[v6:.+]] = subfield %ram_MPORT_1[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<20>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // CHECK:  %[[v7:.+]] = subfield %ram_MPORT[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<20>, mask: uint<1>>
    // CHECK:  %[[v8:.+]] = bitcast %6 : (!firrtl.bundle<a: uint<0>, b: uint<20>>) -> !firrtl.uint<20>
    // CHECK:  strictconnect %7, %8 : !firrtl.uint<20>
    // CHECK:  %[[v9:.+]] = subfield %ram_MPORT_1[mask] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<20>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // CHECK:  %[[v10:.+]] = subfield %ram_MPORT[mask] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<20>, mask: uint<1>>
    // CHECK:  %[[v11:.+]] = bitcast %9 : (!firrtl.bundle<a: uint<1>, b: uint<1>>) -> !firrtl.uint<2>
    // CHECK:  %[[v12:.+]] = bits %11 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
    // CHECK:  %[[v13:.+]] = bits %11 1 to 1 : (!firrtl.uint<2>) -> !firrtl.uint<1>
    // CHECK:  strictconnect %[[v10]], %[[v13]] : !firrtl.uint<1>
    // CHECK:  %[[v14:.+]] = subfield %ram_MPORT_1[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<20>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // CHECK:  strictconnect %[[v14]], %invalid_0 : !firrtl.bundle<a: uint<0>, b: uint<20>>
    // CHECK:  %[[v15:.+]] = subfield %ram_MPORT_1[mask] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<20>>, mask: bundle<a: uint<1>, b: uint<1>>>
    connect %3, %io : !firrtl.bundle<a: uint<0>, b: uint<20>>, !firrtl.bundle<a: uint<0>, b: uint<20>>
  }

  // Tests all the cases when the memory is ignored and not flattened.
  module @ZeroBitMem(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io: !firrtl.bundle<a: uint<0>, b: uint<20>>) {
    // Case 1: No widths.
    %ram_MPORT = mem Undefined  {depth = 1 : i64, name = "ram", portNames = ["MPORT"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<>, mask: bundle<>>
    // CHECK: %ram_MPORT = mem Undefined  {depth = 1 : i64, name = "ram", portNames = ["MPORT"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<>, mask: bundle<>>
    // Case 2: All widths of the data add up to zero.
    %ram_MPORT1 = mem Undefined  {depth = 1 : i64, name = "ram", portNames = ["MPORT"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<0>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // CHECK: = mem Undefined  {depth = 1 : i64, name = "ram", portNames = ["MPORT"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<a: uint<0>, b: uint<0>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // Case 3: Aggregate contains only a single element.
    %single     = mem Undefined  {depth = 1 : i64, name = "ram", portNames = ["MPORT"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<b: uint<10>>, mask: bundle<b: uint<1>>>
    // CHECK: = mem Undefined  {depth = 1 : i64, name = "ram", portNames = ["MPORT"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<b: uint<10>>, mask: bundle<b: uint<1>>>
    // Case 4: Ground Type with zero width.
    %ram_MPORT2, %ram_io_deq_bits_MPORT = mem  Undefined  {depth = 2 : i64, name = "ram", portNames = ["MPORT", "io_deq_bits_MPORT"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<0>>
    // CHECK:  = mem  Undefined  {depth = 2 : i64, name = "ram", portNames = ["MPORT", "io_deq_bits_MPORT"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<0>>
    // Case 5: Any Ground Type.
    %ram_MPORT3, %ram_io_deq_bits_MPORT2 = mem  Undefined  {depth = 2 : i64, name = "ram", portNames = ["MPORT", "io_deq_bits_MPORT"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<1>, mask: uint<1>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    // CHECK:  = mem  Undefined  {depth = 2 : i64, name = "ram", portNames = ["MPORT", "io_deq_bits_MPORT"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<1>, mask: uint<1>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
  }
}
