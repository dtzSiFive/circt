// RUN: circt-translate --export-firrtl --verify-diagnostics %s -o %t
// RUN: cat %t | FileCheck %s --strict-whitespace
// RUN: circt-translate --import-firrtl %t --mlir-print-debuginfo | circt-translate --export-firrtl | diff - %t

// Check emission at various widths, ensuring still parses and round-trips back to same FIRRTL as default width (inc debug info).
// RUN: circt-translate --export-firrtl %s --target-line-length=10 | circt-translate --import-firrtl --mlir-print-debuginfo | circt-translate --export-firrtl | diff - %t
// RUN: circt-translate --export-firrtl %s --target-line-length=1000 | circt-translate --import-firrtl --mlir-print-debuginfo | circt-translate --export-firrtl | diff - %t

// Sanity-check line length control:
// Check if printing with very long line length, no line ends with a comma.
// RUN: circt-translate --export-firrtl %s --target-line-length=1000 | FileCheck %s --implicit-check-not "{{,$}}" --check-prefix PRETTY
// Check if printing with very short line length, removing info locators (@[...]), no line is longer than 5x line length.
// RUN: circt-translate --export-firrtl %s --target-line-length=10 | sed -e 's/ @\[.*\]//' | FileCheck %s --implicit-check-not "{{^(.{50})}}" --check-prefix PRETTY

// CHECK-LABEL: circuit Foo :
// PRETTY-LABEL: circuit Foo :
firrtl.circuit "Foo" {
  // CHECK-LABEL: module Foo :
  module @Foo() {}

  // CHECK-LABEL: module PortsAndTypes :
  module @PortsAndTypes(
    // CHECK-NEXT: input a00 : Clock
    // CHECK-NEXT: input a01 : Reset
    // CHECK-NEXT: input a02 : AsyncReset
    // CHECK-NEXT: input a03 : UInt
    // CHECK-NEXT: input a04 : SInt
    // CHECK-NEXT: input a05 : Analog
    // CHECK-NEXT: input a06 : UInt<42>
    // CHECK-NEXT: input a07 : SInt<42>
    // CHECK-NEXT: input a08 : Analog<42>
    // CHECK-NEXT: input a09 : { a : UInt, flip b : UInt }
    // CHECK-NEXT: input a10 : UInt[42]
    // CHECK-NEXT: output b0 : UInt
    // CHECK-NEXT: output b1 : Probe<UInt<1>>
    // CHECK-NEXT: output b2 : RWProbe<UInt<1>>
    in %a00: !firrtl.clock,
    in %a01: !firrtl.reset,
    in %a02: !firrtl.asyncreset,
    in %a03: !firrtl.uint,
    in %a04: !firrtl.sint,
    in %a05: !firrtl.analog,
    in %a06: !firrtl.uint<42>,
    in %a07: !firrtl.sint<42>,
    in %a08: !firrtl.analog<42>,
    in %a09: !firrtl.bundle<a: uint, b flip: uint>,
    in %a10: !firrtl.vector<uint, 42>,
    out %b0: !firrtl.uint,
    out %b1: !firrtl.probe<uint<1>>,
    out %b2: !firrtl.rwprobe<uint<1>>
  ) {}

  // CHECK-LABEL: module Simple :
  // CHECK:         input someIn : UInt<1>
  // CHECK:         output someOut : UInt<1>
  module @Simple(in %someIn: !firrtl.uint<1>, out %someOut: !firrtl.uint<1>) {
    skip
  }

  // CHECK-LABEL: module Statements :
  module @Statements(in %ui1: !firrtl.uint<1>, in %someAddr: !firrtl.uint<8>, in %someClock: !firrtl.clock, in %someReset: !firrtl.reset, out %someOut: !firrtl.uint<1>, out %ref: !firrtl.probe<uint<1>>) {
    // CHECK: when ui1 :
    // CHECK:   skip
    when %ui1 : !firrtl.uint<1> {
      skip
    }
    // CHECK: when ui1 :
    // CHECK:   skip
    // CHECK: else :
    // CHECK:   skip
    when %ui1 : !firrtl.uint<1> {
      skip
    } else {
      skip
    }
    // CHECK: when ui1 :
    // CHECK:   skip
    // CHECK: else when ui1 :
    // CHECK:   skip
    when %ui1 : !firrtl.uint<1> {
      skip
    } else {
      when %ui1 : !firrtl.uint<1> {
        skip
      }
    }
    // CHECK: wire someWire : UInt<1>
    %someWire = wire : !firrtl.uint<1>
    // CHECK: reg someReg : UInt<1>, someClock
    %someReg = reg %someClock : !firrtl.clock, !firrtl.uint<1>
    // CHECK: reg someReg2 : UInt<1>, someClock with :
    // CHECK:   reset => (someReset, ui1)
    %someReg2 = regreset %someClock, %someReset, %ui1 : !firrtl.clock, !firrtl.reset, !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: node someNode = ui1
    %someNode = node %ui1 : !firrtl.uint<1>
    // CHECK: stop(someClock, ui1, 42) : foo
    stop %someClock, %ui1, 42 {name = "foo"} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: skip
    skip
    // CHECK: printf(someClock, ui1, "some\n magic\"stuff\"", ui1, someReset) : foo
    printf %someClock, %ui1, "some\n magic\"stuff\"" {name = "foo"} (%ui1, %someReset) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.reset
    // CHECK: assert(someClock, ui1, ui1, "msg") : foo
    // CHECK: assume(someClock, ui1, ui1, "msg") : foo
    // CHECK: cover(someClock, ui1, ui1, "msg") : foo
    assert %someClock, %ui1, %ui1, "msg" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {name = "foo"}
    assume %someClock, %ui1, %ui1, "msg" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {name = "foo"}
    cover %someClock, %ui1, %ui1, "msg" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {name = "foo"}
    // CHECK: someOut <= ui1
    connect %someOut, %ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: inst someInst of Simple
    // CHECK: someInst.someIn <= ui1
    // CHECK: someOut <= someInst.someOut
    %someInst_someIn, %someInst_someOut = instance someInst @Simple(in someIn: !firrtl.uint<1>, out someOut: !firrtl.uint<1>)
    connect %someInst_someIn, %ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %someOut, %someInst_someOut : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NOT: _invalid
    // CHECK: someOut is invalid
    %invalid_ui1 = invalidvalue : !firrtl.uint<1>
    connect %someOut, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NOT: _invalid
    // CHECK: someOut is invalid
    %invalid_ui2 = invalidvalue : !firrtl.uint<1>
    strictconnect %someOut, %invalid_ui2 : !firrtl.uint<1>

    // CHECK: unknownWidth <= knownWidth
    %knownWidth = wire : !firrtl.uint<1>
    %unknownWidth = wire : !firrtl.uint
    %widthCast = widthCast %knownWidth :
      (!firrtl.uint<1>) -> !firrtl.uint
    strictconnect %unknownWidth, %widthCast : !firrtl.uint

    // CHECK: unknownReset <= knownReset
    %knownReset = wire : !firrtl.asyncreset
    %unknownReset = wire : !firrtl.reset
    %resetCast = resetCast %knownReset :
      (!firrtl.asyncreset) -> !firrtl.reset
    strictconnect %unknownReset, %resetCast : !firrtl.reset

    // CHECK: attach(an0, an1)
    %an0 = wire : !firrtl.analog<1>
    %an1 = wire : !firrtl.analog<1>
    attach %an0, %an1 : !firrtl.analog<1>, !firrtl.analog<1>

    // CHECK: node k0 = UInt<19>(42)
    // CHECK: node k1 = SInt<19>(42)
    // CHECK: node k2 = UInt(42)
    // CHECK: node k3 = SInt(42)
    %0 = constant 42 : !firrtl.uint<19>
    %1 = constant 42 : !firrtl.sint<19>
    %2 = constant 42 : !firrtl.uint
    %3 = constant 42 : !firrtl.sint
    %k0 = node %0 : !firrtl.uint<19>
    %k1 = node %1 : !firrtl.sint<19>
    %k2 = node %2 : !firrtl.uint
    %k3 = node %3 : !firrtl.sint

    // CHECK: node k4 = asClock(UInt<1>(0))
    // CHECK: node k5 = asAsyncReset(UInt<1>(0))
    // CHECK: node k6 = UInt<1>(0)
    %4 = specialconstant 0 : !firrtl.clock
    %5 = specialconstant 0 : !firrtl.asyncreset
    %6 = specialconstant 0 : !firrtl.reset
    %k4 = node %4 : !firrtl.clock
    %k5 = node %5 : !firrtl.asyncreset
    %k6 = node %6 : !firrtl.reset

    // CHECK: wire bundle : { a : UInt, flip b : UInt }
    // CHECK: wire vector : UInt[42]
    // CHECK: node subfield = bundle.a
    // CHECK: node subindex = vector[19]
    // CHECK: node subaccess = vector[ui1]
    %bundle = wire : !firrtl.bundle<a: uint, b flip: uint>
    %vector = wire : !firrtl.vector<uint, 42>
    %subfield_tmp = subfield %bundle[a] : !firrtl.bundle<a: uint, b flip: uint>
    %subindex_tmp = subindex %vector[19] : !firrtl.vector<uint, 42>
    %subaccess_tmp = subaccess %vector[%ui1] : !firrtl.vector<uint, 42>, !firrtl.uint<1>
    %subfield = node %subfield_tmp : !firrtl.uint
    %subindex = node %subindex_tmp : !firrtl.uint
    %subaccess = node %subaccess_tmp : !firrtl.uint

    %x = node %2 : !firrtl.uint
    %y = node %2 : !firrtl.uint

    // CHECK: node addPrimOp = add(x, y)
    // CHECK: node subPrimOp = sub(x, y)
    // CHECK: node mulPrimOp = mul(x, y)
    // CHECK: node divPrimOp = div(x, y)
    // CHECK: node remPrimOp = rem(x, y)
    // CHECK: node andPrimOp = and(x, y)
    // CHECK: node orPrimOp = or(x, y)
    // CHECK: node xorPrimOp = xor(x, y)
    // CHECK: node leqPrimOp = leq(x, y)
    // CHECK: node ltPrimOp = lt(x, y)
    // CHECK: node geqPrimOp = geq(x, y)
    // CHECK: node gtPrimOp = gt(x, y)
    // CHECK: node eqPrimOp = eq(x, y)
    // CHECK: node neqPrimOp = neq(x, y)
    // CHECK: node catPrimOp = cat(x, y)
    // CHECK: node dShlPrimOp = dshl(x, y)
    // CHECK: node dShlwPrimOp = dshlw(x, y)
    // CHECK: node dShrPrimOp = dshr(x, y)
    %addPrimOp_tmp = add %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %subPrimOp_tmp = sub %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %mulPrimOp_tmp = mul %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %divPrimOp_tmp = div %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %remPrimOp_tmp = rem %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %andPrimOp_tmp = and %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %orPrimOp_tmp = or %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %xorPrimOp_tmp = xor %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %leqPrimOp_tmp = leq %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %ltPrimOp_tmp = lt %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %geqPrimOp_tmp = geq %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %gtPrimOp_tmp = gt %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %eqPrimOp_tmp = eq %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %neqPrimOp_tmp = neq %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %catPrimOp_tmp = cat %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %dShlPrimOp_tmp = dshl %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %dShlwPrimOp_tmp = dshlw %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %dShrPrimOp_tmp = dshr %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %addPrimOp = node %addPrimOp_tmp : !firrtl.uint
    %subPrimOp = node %subPrimOp_tmp : !firrtl.uint
    %mulPrimOp = node %mulPrimOp_tmp : !firrtl.uint
    %divPrimOp = node %divPrimOp_tmp : !firrtl.uint
    %remPrimOp = node %remPrimOp_tmp : !firrtl.uint
    %andPrimOp = node %andPrimOp_tmp : !firrtl.uint
    %orPrimOp = node %orPrimOp_tmp : !firrtl.uint
    %xorPrimOp = node %xorPrimOp_tmp : !firrtl.uint
    %leqPrimOp = node %leqPrimOp_tmp : !firrtl.uint<1>
    %ltPrimOp = node %ltPrimOp_tmp : !firrtl.uint<1>
    %geqPrimOp = node %geqPrimOp_tmp : !firrtl.uint<1>
    %gtPrimOp = node %gtPrimOp_tmp : !firrtl.uint<1>
    %eqPrimOp = node %eqPrimOp_tmp : !firrtl.uint<1>
    %neqPrimOp = node %neqPrimOp_tmp : !firrtl.uint<1>
    %catPrimOp = node %catPrimOp_tmp : !firrtl.uint
    %dShlPrimOp = node %dShlPrimOp_tmp : !firrtl.uint
    %dShlwPrimOp = node %dShlwPrimOp_tmp : !firrtl.uint
    %dShrPrimOp = node %dShrPrimOp_tmp : !firrtl.uint

    // CHECK: node asSIntPrimOp = asSInt(x)
    // CHECK: node asUIntPrimOp = asUInt(x)
    // CHECK: node asAsyncResetPrimOp = asAsyncReset(x)
    // CHECK: node asClockPrimOp = asClock(x)
    // CHECK: node cvtPrimOp = cvt(x)
    // CHECK: node negPrimOp = neg(x)
    // CHECK: node notPrimOp = not(x)
    // CHECK: node andRPrimOp = andr(x)
    // CHECK: node orRPrimOp = orr(x)
    // CHECK: node xorRPrimOp = xorr(x)
    %asSIntPrimOp_tmp = asSInt %x : (!firrtl.uint) -> !firrtl.sint
    %asUIntPrimOp_tmp = asUInt %x : (!firrtl.uint) -> !firrtl.uint
    %asAsyncResetPrimOp_tmp = asAsyncReset %x : (!firrtl.uint) -> !firrtl.asyncreset
    %asClockPrimOp_tmp = asClock %x : (!firrtl.uint) -> !firrtl.clock
    %cvtPrimOp_tmp = cvt %x : (!firrtl.uint) -> !firrtl.sint
    %negPrimOp_tmp = neg %x : (!firrtl.uint) -> !firrtl.sint
    %notPrimOp_tmp = not %x : (!firrtl.uint) -> !firrtl.uint
    %andRPrimOp_tmp = andr %x : (!firrtl.uint) -> !firrtl.uint<1>
    %orRPrimOp_tmp = orr %x : (!firrtl.uint) -> !firrtl.uint<1>
    %xorRPrimOp_tmp = xorr %x : (!firrtl.uint) -> !firrtl.uint<1>
    %asSIntPrimOp = node %asSIntPrimOp_tmp : !firrtl.sint
    %asUIntPrimOp = node %asUIntPrimOp_tmp : !firrtl.uint
    %asAsyncResetPrimOp = node %asAsyncResetPrimOp_tmp : !firrtl.asyncreset
    %asClockPrimOp = node %asClockPrimOp_tmp : !firrtl.clock
    %cvtPrimOp = node %cvtPrimOp_tmp : !firrtl.sint
    %negPrimOp = node %negPrimOp_tmp : !firrtl.sint
    %notPrimOp = node %notPrimOp_tmp : !firrtl.uint
    %andRPrimOp = node %andRPrimOp_tmp : !firrtl.uint<1>
    %orRPrimOp = node %orRPrimOp_tmp : !firrtl.uint<1>
    %xorRPrimOp = node %xorRPrimOp_tmp : !firrtl.uint<1>

    // CHECK: node bitsPrimOp = bits(x, 4, 2)
    // CHECK: node headPrimOp = head(x, 4)
    // CHECK: node tailPrimOp = tail(x, 4)
    // CHECK: node padPrimOp = pad(x, 16)
    // CHECK: node muxPrimOp = mux(ui1, x, y)
    // CHECK: node shlPrimOp = shl(x, 4)
    // CHECK: node shrPrimOp = shr(x, 4)
    %bitsPrimOp_tmp = bits %x 4 to 2 : (!firrtl.uint) -> !firrtl.uint<3>
    %headPrimOp_tmp = head %x, 4 : (!firrtl.uint) -> !firrtl.uint<4>
    %tailPrimOp_tmp = tail %x, 4 : (!firrtl.uint) -> !firrtl.uint
    %padPrimOp_tmp = pad %x, 16 : (!firrtl.uint) -> !firrtl.uint
    %muxPrimOp_tmp = mux(%ui1, %x, %y) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %shlPrimOp_tmp = shl %x, 4 : (!firrtl.uint) -> !firrtl.uint
    %shrPrimOp_tmp = shr %x, 4 : (!firrtl.uint) -> !firrtl.uint
    %bitsPrimOp = node %bitsPrimOp_tmp : !firrtl.uint<3>
    %headPrimOp = node %headPrimOp_tmp : !firrtl.uint<4>
    %tailPrimOp = node %tailPrimOp_tmp : !firrtl.uint
    %padPrimOp = node %padPrimOp_tmp : !firrtl.uint
    %muxPrimOp = node %muxPrimOp_tmp : !firrtl.uint
    %shlPrimOp = node %shlPrimOp_tmp : !firrtl.uint
    %shrPrimOp = node %shrPrimOp_tmp : !firrtl.uint

    %MyMem_a, %MyMem_b, %MyMem_c = mem Undefined {depth = 8, name = "MyMem", portNames = ["a", "b", "c"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<4>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<4>, mask: uint<1>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: uint<4>, wmode: uint<1>, wdata: uint<4>, wmask: uint<1>>
    %MyMem_a_clk = subfield %MyMem_a[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<4>>
    %MyMem_b_clk = subfield %MyMem_b[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<4>, mask: uint<1>>
    %MyMem_c_clk = subfield %MyMem_c[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: uint<4>, wmode: uint<1>, wdata: uint<4>, wmask: uint<1>>
    connect %MyMem_a_clk, %someClock : !firrtl.clock, !firrtl.clock
    connect %MyMem_b_clk, %someClock : !firrtl.clock, !firrtl.clock
    connect %MyMem_c_clk, %someClock : !firrtl.clock, !firrtl.clock
    // CHECK:       mem MyMem :
    // CHECK-NEXT:    data-type => UInt<4>
    // CHECK-NEXT:    depth => 8
    // CHECK-NEXT:    read-latency => 0
    // CHECK-NEXT:    write-latency => 1
    // CHECK-NEXT:    reader => a
    // CHECK-NEXT:    writer => b
    // CHECK-NEXT:    readwriter => c
    // CHECK-NEXT:    read-under-write => undefined
    // CHECK-NEXT:  MyMem.a.clk <= someClock
    // CHECK-NEXT:  MyMem.b.clk <= someClock
    // CHECK-NEXT:  MyMem.c.clk <= someClock

    %combmem = chirrtl.combmem : !chirrtl.cmemory<uint<3>, 256>
    %port0_data, %port0_port = chirrtl.memoryport Infer %combmem {name = "port0"} : (!chirrtl.cmemory<uint<3>, 256>) -> (!firrtl.uint<3>, !chirrtl.cmemoryport)
    when %ui1 : !firrtl.uint<1> {
      chirrtl.memoryport.access %port0_port[%someAddr], %someClock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
    }
    // CHECK:      cmem combmem : UInt<3>[256]
    // CHECK-NEXT: when ui1 :
    // CHECK-NEXT:   infer mport port0 = combmem[someAddr], someClock

    %seqmem = chirrtl.seqmem Undefined : !chirrtl.cmemory<uint<3>, 256>
    %port1_data, %port1_port = chirrtl.memoryport Infer %seqmem {name = "port1"} : (!chirrtl.cmemory<uint<3>, 256>) -> (!firrtl.uint<3>, !chirrtl.cmemoryport)
    when %ui1 : !firrtl.uint<1> {
      chirrtl.memoryport.access %port1_port[%someAddr], %someClock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
    }
    // CHECK:      smem seqmem : UInt<3>[256] undefined
    // CHECK-NEXT: when ui1 :
    // CHECK-NEXT:   infer mport port1 = seqmem[someAddr], someClock

    connect %port0_data, %port1_data : !firrtl.uint<3>, !firrtl.uint<3>
    // CHECK: port0 <= port1

    %invalid_clock = invalidvalue : !firrtl.clock
    %dummyReg = reg %invalid_clock : !firrtl.clock, !firrtl.uint<42>
    // CHECK: wire [[INV:_invalid.*]] : Clock
    // CHECK-NEXT: [[INV]] is invalid
    // CHECK-NEXT: reg dummyReg : UInt<42>, [[INV]]
  }

  // CHECK-LABEL: module RefSource
  module @RefSource(out %a_ref: !firrtl.probe<uint<1>>,
                           out %a_rwref: !firrtl.rwprobe<uint<1>>) {
    %a, %_a_rwref = wire forceable : !firrtl.uint<1>,
      !firrtl.rwprobe<uint<1>>
    // CHECK: define a_ref = probe(a)
    // CHECK: define a_rwref = rwprobe(a)
    %a_ref_send = ref.send %a : !firrtl.uint<1>
    ref.define %a_ref, %a_ref_send : !firrtl.probe<uint<1>>
    ref.define %a_rwref, %_a_rwref : !firrtl.rwprobe<uint<1>>
  }

  // CHECK-LABEL: module RefSink
  module @RefSink(
    in %clock: !firrtl.clock,
    in %enable: !firrtl.uint<1>
  ) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    // CHECK: node b = read(refSource.a_ref)
    %refSource_a_ref, %refSource_a_rwref =
      instance refSource @RefSource(
        out a_ref: !firrtl.probe<uint<1>>,
        out a_rwref: !firrtl.rwprobe<uint<1>>
      )
    %a_ref_resolve =
      ref.resolve %refSource_a_ref : !firrtl.probe<uint<1>>
    %b = node %a_ref_resolve : !firrtl.uint<1>
    // CHECK-NEXT: force_initial(refSource.a_rwref, UInt<1>(0))
    ref.force_initial %c1_ui1, %refSource_a_rwref, %c0_ui1 :
      !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT: release_initial(refSource.a_rwref)
    ref.release_initial %c1_ui1, %refSource_a_rwref :
      !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    // CHECK-NEXT: when enable :
    // CHECK-NEXT:   force_initial(refSource.a_rwref, UInt<1>(0))
    when %enable : !firrtl.uint<1> {
      ref.force_initial %c1_ui1, %refSource_a_rwref, %c0_ui1 :
        !firrtl.uint<1>, !firrtl.uint<1>
    }
    // CHECK-NEXT: when enable :
    // CHECK-NEXT:   release_initial(refSource.a_rwref)
    when %enable : !firrtl.uint<1> {
      ref.release_initial %c1_ui1, %refSource_a_rwref :
        !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    }
    // CHECK-NEXT: force(clock, enable, refSource.a_rwref, UInt<1>(1))
    ref.force %clock, %enable, %refSource_a_rwref, %c1_ui1 :
      !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT: release(clock, enable, refSource.a_rwref)
    ref.release %clock, %enable, %refSource_a_rwref :
      !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
  }

  // CHECK-LABEL: module RefExport
  module @RefExport(out %a_ref: !firrtl.probe<uint<1>>,
                           out %a_rwref: !firrtl.rwprobe<uint<1>>) {
    // CHECK: define a_ref = refSource.a_ref
    // CHECK: define a_rwref = refSource.a_rwref
    %refSource_a_ref, %refSource_a_rwref =
      instance refSource @RefSource(
        out a_ref: !firrtl.probe<uint<1>>,
        out a_rwref: !firrtl.rwprobe<uint<1>>
      )
    ref.define %a_ref, %refSource_a_ref : !firrtl.probe<uint<1>>
    ref.define %a_rwref, %refSource_a_rwref : !firrtl.rwprobe<uint<1>>
  }

  // CHECK-LABEL: extmodule ExtOpenAgg
  // CHECK-NEXT:  output out : { a : { data : UInt<1> },
  // CHECK-NEXT:                 b : { x : UInt<2>, y : Probe<UInt<2>[3]> }[2] }
  extmodule @ExtOpenAgg(
      out out: !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<vector<uint<2>, 3>>>, 2>>)

  // CHECK-LABEL: module OpenAggTest
  module @OpenAggTest(
  // CHECK-NEXT: output out_b_0_y_2 : Probe<UInt<2>>
  // CHECK-EMPTY:
      out %out_b_0_y_2 : !firrtl.probe<uint<2>>) {

    // CHECK-NEXT: inst oa of ExtOpenAgg
    %oa_out = instance oa @ExtOpenAgg(out out: !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<vector<uint<2>, 3>>>, 2>>)

    %a = opensubfield %oa_out[a] : !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<vector<uint<2>, 3>>>, 2>>
    %data = subfield %a[data] : !firrtl.bundle<data: uint<1>>
    // CHECK-NEXT:  node n_data = oa.out.a.data
    %n_data = node %data : !firrtl.uint<1>
    %b = opensubfield %oa_out[b] : !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<vector<uint<2>, 3>>>, 2>>
    %b_0 = opensubindex %b[0] : !firrtl.openvector<openbundle<x: uint<2>, y: probe<vector<uint<2>, 3>>>, 2>
    %b_0_y = opensubfield %b_0[y] : !firrtl.openbundle<x : uint<2>, y: probe<vector<uint<2>, 3>>>
    %b_0_y_2 = ref.sub %b_0_y[2] : !firrtl.probe<vector<uint<2>, 3>>
    // openagg indexing + ref.sub
    // CHECK-NEXT: define out_b_0_y_2 = oa.out.b[0].y[2]
    ref.define %out_b_0_y_2, %b_0_y_2 : !firrtl.probe<uint<2>>
  }

  extmodule @MyParameterizedExtModule<DEFAULT: i64 = 0, DEPTH: f64 = 3.242000e+01, FORMAT: none = "xyz_timeout=%d\0A", WIDTH: i8 = 32>(in in: !firrtl.uint, out out: !firrtl.uint<8>) attributes {defname = "name_thing"}
  // CHECK-LABEL: extmodule MyParameterizedExtModule :
  // CHECK-NEXT:    input in : UInt
  // CHECK-NEXT:    output out : UInt<8>
  // CHECK-NEXT:    defname = name_thing
  // CHECK-NEXT:    parameter DEFAULT = 0
  // CHECK-NEXT:    parameter DEPTH = 32.42
  // CHECK-NEXT:    parameter FORMAT = "xyz_timeout=%d\n"
  // CHECK-NEXT:    parameter WIDTH = 32

  // CHECK-LABEL: module ConstTypes :
  module @ConstTypes(
    // CHECK-NEXT: input a00 : const Clock
    // CHECK-NEXT: input a01 : const Reset
    // CHECK-NEXT: input a02 : const AsyncReset
    // CHECK-NEXT: input a03 : const UInt
    // CHECK-NEXT: input a04 : const SInt
    // CHECK-NEXT: input a05 : const Analog
    // CHECK-NEXT: input a06 : const UInt<42>
    // CHECK-NEXT: input a07 : const SInt<42>
    // CHECK-NEXT: input a08 : const Analog<42>
    // CHECK-NEXT: input a09 : const { a : UInt, flip b : UInt }
    // CHECK-NEXT: input a10 : { a : const UInt, flip b : UInt }
    // CHECK-NEXT: input a11 : const UInt[42]
    // CHECK-NEXT: output b0 : const UInt<42>
    in %a00: !firrtl.const.clock,
    in %a01: !firrtl.const.reset,
    in %a02: !firrtl.const.asyncreset,
    in %a03: !firrtl.const.uint,
    in %a04: !firrtl.const.sint,
    in %a05: !firrtl.const.analog,
    in %a06: !firrtl.const.uint<42>,
    in %a07: !firrtl.const.sint<42>,
    in %a08: !firrtl.const.analog<42>,
    in %a09: !firrtl.const.bundle<a: uint, b flip: uint>,
    in %a10: !firrtl.bundle<a: const.uint, b flip: uint>,
    in %a11: !firrtl.const.vector<uint, 42>,
    out %b0: !firrtl.const.uint<42>
  ) {
    // Make sure literals strip the 'const' prefix
    // CHECK: b0 <= UInt<42>(1)
    %c = constant 1 : !firrtl.const.uint<42>
    strictconnect %b0, %c : !firrtl.const.uint<42>
  }
}
