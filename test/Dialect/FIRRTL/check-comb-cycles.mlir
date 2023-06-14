// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-comb-loops))' --split-input-file --verify-diagnostics %s | FileCheck %s

// Loop-free circuit
// CHECK: circuit "hasnoloops"
firrtl.circuit "hasnoloops"   {
  module @thru(in %in1: !firrtl.uint<1>, in %in2: !firrtl.uint<1>, out %out1: !firrtl.uint<1>, out %out2: !firrtl.uint<1>) {
    connect %out1, %in1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %out2, %in2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  module @hasnoloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %x = wire  : !firrtl.uint<1>
    %inner_in1, %inner_in2, %inner_out1, %inner_out2 = instance inner @thru(in in1: !firrtl.uint<1>, in in2: !firrtl.uint<1>, out out1: !firrtl.uint<1>, out out2: !firrtl.uint<1>)
    connect %inner_in1, %a : !firrtl.uint<1>, !firrtl.uint<1>
    connect %x, %inner_out1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %inner_in2, %x : !firrtl.uint<1>, !firrtl.uint<1>
    connect %b, %inner_out2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Simple combinational loop
// CHECK-NOT: circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: hasloops.{y <- z <- y}}}
  module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = wire  : !firrtl.uint<1>
    %z = wire  : !firrtl.uint<1>
    connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    connect %z, %y : !firrtl.uint<1>, !firrtl.uint<1>
    connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Single-element combinational loop
// CHECK-NOT: circuit "loop"
firrtl.circuit "loop"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: loop.{w <- w}}}
  module @loop(out %y: !firrtl.uint<8>) {
    %w = wire  : !firrtl.uint<8>
    connect %w, %w : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

// Node combinational loop
// CHECK-NOT: circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: hasloops.{y <- z <- ... <- y}}}
  module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = wire  : !firrtl.uint<1>
    connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %0 = and %c, %y : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %z = node %0  : !firrtl.uint<1>
    connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Combinational loop through a combinational memory read port
// CHECK-NOT: circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: hasloops.{y <- z <- m.r.data <- m.r.addr <- y}}}
  module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = wire  : !firrtl.uint<1>
    %z = wire  : !firrtl.uint<1>
    connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %m_r = mem Undefined  {depth = 2 : i64, name = "m", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %0 = subfield %m_r[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    connect %0, %clk : !firrtl.clock, !firrtl.clock
    %1 = subfield %m_r[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    connect %1, %y : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = subfield %m_r[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %c1_ui = constant 1 : !firrtl.uint
    connect %2, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    %3 = subfield %m_r[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    connect %z, %3 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}


// -----

// Combination loop through an instance
// CHECK-NOT: circuit "hasloops"
firrtl.circuit "hasloops"   {
  module @thru(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    connect %out, %in : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: hasloops.{y <- z <- inner.out <- inner.in <- y}}}
  module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = wire  : !firrtl.uint<1>
    %z = wire  : !firrtl.uint<1>
    connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %inner_in, %inner_out = instance inner @thru(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    connect %inner_in, %y : !firrtl.uint<1>, !firrtl.uint<1>
    connect %z, %inner_out : !firrtl.uint<1>, !firrtl.uint<1>
    connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Multiple simple loops in one SCC
// CHECK-NOT: circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{hasloops.{c <- b <- ... <- a <- ... <- c}}}
  module @hasloops(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    %a = wire  : !firrtl.uint<1>
    %b = wire  : !firrtl.uint<1>
    %c = wire  : !firrtl.uint<1>
    %d = wire  : !firrtl.uint<1>
    %e = wire  : !firrtl.uint<1>
    %0 = and %c, %i : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = and %a, %d : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = and %c, %e : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %d, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %e, %b : !firrtl.uint<1>, !firrtl.uint<1>
    connect %o, %e : !firrtl.uint<1>, !firrtl.uint<1>
  }
}


// -----

firrtl.circuit "strictConnectAndConnect" {
  // expected-error @below {{strictConnectAndConnect.{b <- a <- b}}}
  module @strictConnectAndConnect(out %a: !firrtl.uint<11>, out %b: !firrtl.uint<11>) {
    %w = wire : !firrtl.uint<11>
    strictconnect %b, %w : !firrtl.uint<11>
    connect %a, %b : !firrtl.uint<11>, !firrtl.uint<11>
    strictconnect %b, %a : !firrtl.uint<11>
  }
}

// -----

firrtl.circuit "vectorRegInit"   {
  module @vectorRegInit(in %clk: !firrtl.clock) {
    %reg = reg %clk : !firrtl.clock, !firrtl.vector<uint<8>, 2>
    %0 = subindex %reg[0] : !firrtl.vector<uint<8>, 2>
    connect %0, %0 : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

firrtl.circuit "bundleRegInit"   {
  module @bundleRegInit(in %clk: !firrtl.clock) {
    %reg = reg %clk : !firrtl.clock, !firrtl.bundle<a: uint<1>>
    %0 = subfield %reg[a] : !firrtl.bundle<a: uint<1>>
    connect %0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "PortReadWrite"  {
  extmodule private @Bar(in a: !firrtl.uint<1>)
  // expected-error @below {{PortReadWrite.{a <- bar.a <- a}}}
  module @PortReadWrite() {
    %a = wire : !firrtl.uint<1>
    %bar_a = instance bar interesting_name  @Bar(in a: !firrtl.uint<1>)
    strictconnect %bar_a, %a : !firrtl.uint<1>
    strictconnect %a, %bar_a : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Foo"  {
  module private @Bar(in %a: !firrtl.uint<1>) {}
  // expected-error @below {{Foo.{bar.a <- a <- bar.a}}}
  module @Foo(out %a: !firrtl.uint<1>) {
    %bar_a = instance bar interesting_name  @Bar(in a: !firrtl.uint<1>)
    strictconnect %bar_a, %a : !firrtl.uint<1>
    strictconnect %a, %bar_a : !firrtl.uint<1>
  }
}

// -----

// Node combinational loop through vector subindex
// CHECK-NOT: circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{hasloops.{w[3] <- z <- ... <- w[3]}}}
  module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %w = wire  : !firrtl.vector<uint<1>,10>
    %y = subindex %w[3]  : !firrtl.vector<uint<1>,10>
    connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %0 = and %c, %y : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %z = node %0  : !firrtl.uint<1>
    connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Node combinational loop through vector subindex
// CHECK-NOT: circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{hasloops.{bar_a[0] <- b[0] <- bar_b[0] <- bar_a[0]}}}
  module @hasloops(out %b: !firrtl.vector<uint<1>, 2>) {
    %bar_a = wire : !firrtl.vector<uint<1>, 2>
    %bar_b = wire : !firrtl.vector<uint<1>, 2>
    %0 = subindex %b[0] : !firrtl.vector<uint<1>, 2>
    %1 = subindex %bar_a[0] : !firrtl.vector<uint<1>, 2>
    strictconnect %1, %0 : !firrtl.uint<1>
    %4 = subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
    %5 = subindex %b[0] : !firrtl.vector<uint<1>, 2>
    strictconnect %5, %4 : !firrtl.uint<1>
    %v0 = subindex %bar_a[0] : !firrtl.vector<uint<1>, 2>
    %v1 = subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
    strictconnect %v1, %v0 : !firrtl.uint<1>
  }
}

// -----

// Combinational loop through instance ports
// CHECK-NOT: circuit "hasloops"
firrtl.circuit "hasLoops"  {
  // expected-error @below {{hasLoops.{bar.a[0] <- b[0] <- bar.b[0] <- bar.a[0]}}}
  module @hasLoops(out %b: !firrtl.vector<uint<1>, 2>) {
    %bar_a, %bar_b = instance bar  @Bar(in a: !firrtl.vector<uint<1>, 2>, out b: !firrtl.vector<uint<1>, 2>)
    %0 = subindex %b[0] : !firrtl.vector<uint<1>, 2>
    %1 = subindex %bar_a[0] : !firrtl.vector<uint<1>, 2>
    strictconnect %1, %0 : !firrtl.uint<1>
    %4 = subindex %bar_b[0] : !firrtl.vector<uint<1>, 2>
    %5 = subindex %b[0] : !firrtl.vector<uint<1>, 2>
    strictconnect %5, %4 : !firrtl.uint<1>
  }
   
  module private @Bar(in %a: !firrtl.vector<uint<1>, 2>, out %b: !firrtl.vector<uint<1>, 2>) {
    %0 = subindex %a[0] : !firrtl.vector<uint<1>, 2>
    %1 = subindex %b[0] : !firrtl.vector<uint<1>, 2>
    strictconnect %1, %0 : !firrtl.uint<1>
    %2 = subindex %a[1] : !firrtl.vector<uint<1>, 2>
    %3 = subindex %b[1] : !firrtl.vector<uint<1>, 2>
    strictconnect %3, %2 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "bundleWire"   {
  // expected-error @below {{bundleWire.{w.foo.bar.baz <- out2 <- x <- w.foo.bar.baz}}}
  module @bundleWire(in %arg: !firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>,
                           out %out1: !firrtl.uint<1>, out %out2: !firrtl.sint<64>) {

    %w = wire : !firrtl.bundle<foo: bundle<bar: bundle<baz: sint<64>>, qux: uint<1>>>
    %w0 = subfield %w[foo] : !firrtl.bundle<foo: bundle<bar: bundle<baz: sint<64>>, qux: uint<1>>>
    %w0_0 = subfield %w0[bar] : !firrtl.bundle<bar: bundle<baz: sint<64>>, qux: uint<1>>
    %w0_0_0 = subfield %w0_0[baz] : !firrtl.bundle<baz: sint<64>>
    %x = wire  : !firrtl.sint<64>

    %0 = subfield %arg[foo] : !firrtl.bundle<foo: bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>>
    %1 = subfield %0[bar] : !firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>
    %2 = subfield %1[baz] : !firrtl.bundle<baz: uint<1>>
    %3 = subfield %0[qux] : !firrtl.bundle<bar: bundle<baz: uint<1>>, qux: sint<64>>
    connect %w0_0_0, %3 : !firrtl.sint<64>, !firrtl.sint<64>
    connect %x, %w0_0_0 : !firrtl.sint<64>, !firrtl.sint<64>
    connect %out2, %x : !firrtl.sint<64>, !firrtl.sint<64>
    connect %w0_0_0, %out2 : !firrtl.sint<64>, !firrtl.sint<64>
    connect %out1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "registerLoop"   {
  // CHECK: module @registerLoop(in %clk: !firrtl.clock)
  module @registerLoop(in %clk: !firrtl.clock) {
    %w = wire : !firrtl.bundle<a: uint<1>>
    %r = reg %clk : !firrtl.clock, !firrtl.bundle<a: uint<1>>
    %0 = subfield %w[a]: !firrtl.bundle<a: uint<1>>
    %1 = subfield %w[a]: !firrtl.bundle<a: uint<1>>
    %2 = subfield %r[a]: !firrtl.bundle<a: uint<1>>
    %3 = subfield %r[a]: !firrtl.bundle<a: uint<1>>
    connect %2, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Simple combinational loop
// CHECK-NOT: circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{hasloops.{y <- z <- y}}}
  module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = wire  : !firrtl.uint<1>
    %z = wire  : !firrtl.uint<1>
    connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    connect %z, %y : !firrtl.uint<1>, !firrtl.uint<1>
    connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Combinational loop through a combinational memory read port
// CHECK-NOT: circuit "hasloops"
firrtl.circuit "hasloops"   {
  // expected-error @below {{hasloops.{y <- z <- m.r.data <- m.r.en <- y}}}
  module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = wire  : !firrtl.uint<1>
    %z = wire  : !firrtl.uint<1>
    connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %m_r = mem Undefined  {depth = 2 : i64, name = "m", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %0 = subfield %m_r[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    connect %0, %clk : !firrtl.clock, !firrtl.clock
    %1 = subfield %m_r[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    %2 = subfield %m_r[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    connect %2, %y : !firrtl.uint<1>, !firrtl.uint<1>
    %c1_ui = constant 1 : !firrtl.uint
    connect %2, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    %3 = subfield %m_r[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
    connect %z, %3 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}


// -----

// Combination loop through an instance
// CHECK-NOT: circuit "hasloops"
firrtl.circuit "hasloops"   {
  module @thru1(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    connect %out, %in : !firrtl.uint<1>, !firrtl.uint<1>
  }

  module @thru2(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %inner_in, %inner_out = instance inner1 @thru1(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    connect %inner_in, %in : !firrtl.uint<1>, !firrtl.uint<1>
    connect %out, %inner_out : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{hasloops.{y <- z <- inner2.out <- inner2.in <- y}}}
  module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = wire  : !firrtl.uint<1>
    %z = wire  : !firrtl.uint<1>
    connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %inner_in, %inner_out = instance inner2 @thru2(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    connect %inner_in, %y : !firrtl.uint<1>, !firrtl.uint<1>
    connect %z, %inner_out : !firrtl.uint<1>, !firrtl.uint<1>
    connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}


// -----

// CHECK: circuit "hasloops"
firrtl.circuit "hasloops"  {
  module @thru1(in %clk: !firrtl.clock, in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %reg = reg  %clk  : !firrtl.clock, !firrtl.uint<1>
    connect %reg, %in : !firrtl.uint<1>, !firrtl.uint<1>
    connect %out, %reg : !firrtl.uint<1>, !firrtl.uint<1>
  }
  module @thru2(in %clk: !firrtl.clock, in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %inner1_clk, %inner1_in, %inner1_out = instance inner1  @thru1(in clk: !firrtl.clock, in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    connect %inner1_clk, %clk : !firrtl.clock, !firrtl.clock
    connect %inner1_in, %in : !firrtl.uint<1>, !firrtl.uint<1>
    connect %out, %inner1_out : !firrtl.uint<1>, !firrtl.uint<1>
  }
  module @hasloops(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %y = wire   : !firrtl.uint<1>
    %z = wire   : !firrtl.uint<1>
    connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
    %inner2_clk, %inner2_in, %inner2_out = instance inner2  @thru2(in clk: !firrtl.clock, in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    connect %inner2_clk, %clk : !firrtl.clock, !firrtl.clock
    connect %inner2_in, %y : !firrtl.uint<1>, !firrtl.uint<1>
    connect %z, %inner2_out : !firrtl.uint<1>, !firrtl.uint<1>
    connect %y, %z : !firrtl.uint<1>, !firrtl.uint<1>
    connect %d, %z : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "subaccess"   {
  // expected-error-re @below {{subaccess.{b[0].wo <- b[{{[0-3]}}].wo}}}
  module @subaccess(in %sel1: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>) {
    %0 = subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %1 = subfield %0[wo] : !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    %2 = subindex %b[0] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>
    %3 = subfield %2[wo]: !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    strictconnect %3, %1 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "subaccess"   {
  // expected-error-re @below {{subaccess.{b[{{[0-3]}}].wo <- b[{{[0-3]}}].wo}}}
  module @subaccess(in %sel1: !firrtl.uint<2>, in %sel2: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>) {
    %0 = subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %1 = subfield %0[wo] : !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    %2 = subaccess %b[%sel2] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %3 = subfield %2[wo]: !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    strictconnect %3, %1 : !firrtl.uint<1>
  }
}

// -----

// CHECK: circuit "subaccess"   {
firrtl.circuit "subaccess"   {
  module @subaccess(in %sel1: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>) {
    %0 = subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %1 = subfield %0[wo] : !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    %2 = subindex %b[0] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>
    %3 = subfield %2[wi]: !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    strictconnect %3, %1 : !firrtl.uint<1>
  }
}

// -----

// CHECK: circuit "subaccess"   {
firrtl.circuit "subaccess"   {
  module @subaccess(in %sel1: !firrtl.uint<2>, in %sel2: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>) {
    %0 = subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %1 = subfield %0[wo] : !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    %2 = subaccess %b[%sel2] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %3 = subfield %2[wi]: !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    strictconnect %3, %1 : !firrtl.uint<1>
  }
}

// -----

// CHECK: circuit "subaccess"   {
firrtl.circuit "subaccess"   {
  module @subaccess(in %sel1: !firrtl.uint<2>, in %sel2: !firrtl.uint<2>, out %b: !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>) {
    %0 = subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %1 = subfield %0[wo] : !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    %2 = subaccess %b[%sel1] : !firrtl.vector<bundle<wo: uint<1>, wi: uint<1>>, 4>, !firrtl.uint<2>
    %3 = subfield %2[wi]: !firrtl.bundle<wo: uint<1>, wi: uint<1>>
    strictconnect %3, %1 : !firrtl.uint<1>
  }
}

// -----

// Two input ports share part of the path to an output port.
// CHECK-NOT: circuit "revisitOps"
firrtl.circuit "revisitOps"   {
  module @thru(in %in1: !firrtl.uint<1>,in %in2: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %1 = mux(%in1, %in1, %in2)  : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %out, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{revisitOps.{inner2.out <- inner2.in2 <- x <- inner2.out}}}
  module @revisitOps() {
    %in1, %in2, %out = instance inner2 @thru(in in1: !firrtl.uint<1>,in in2: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %x = wire  : !firrtl.uint<1>
    connect %in2, %x : !firrtl.uint<1>, !firrtl.uint<1>
    connect %x, %out : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Two input ports and a wire share path to an output port.
// CHECK-NOT: circuit "revisitOps"
firrtl.circuit "revisitOps"   {
  module @thru(in %in1: !firrtl.vector<uint<1>,2>, in %in2: !firrtl.vector<uint<1>,3>, out %out: !firrtl.vector<uint<1>,2>) {
    %w = wire : !firrtl.uint<1>
    %in1_0 = subindex %in1[0] : !firrtl.vector<uint<1>,2>
    %in2_1 = subindex %in2[1] : !firrtl.vector<uint<1>,3>
    %out_1 = subindex %out[1] : !firrtl.vector<uint<1>,2>
    %1 = mux(%w, %in1_0, %in2_1)  : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %out_1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{revisitOps.{inner2.out[1] <- inner2.in2[1] <- x <- inner2.out[1]}}}
  module @revisitOps() {
    %in1, %in2, %out = instance inner2 @thru(in in1: !firrtl.vector<uint<1>,2>, in in2: !firrtl.vector<uint<1>,3>, out out: !firrtl.vector<uint<1>,2>)
    %in1_0 = subindex %in1[0] : !firrtl.vector<uint<1>,2>
    %in2_1 = subindex %in2[1] : !firrtl.vector<uint<1>,3>
    %out_1 = subindex %out[1] : !firrtl.vector<uint<1>,2>
    %x = wire  : !firrtl.uint<1>
    connect %in2_1, %x : !firrtl.uint<1>, !firrtl.uint<1>
    connect %x, %out_1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Shared comb path from input ports, ensure that all the paths to the output port are discovered.
// CHECK-NOT: circuit "revisitOps"
firrtl.circuit "revisitOps"   {
  module @thru(in %in0: !firrtl.vector<uint<1>,2>, in %in1: !firrtl.vector<uint<1>,2>, in %in2: !firrtl.vector<uint<1>,3>, out %out: !firrtl.vector<uint<1>,2>) {
    %w = wire : !firrtl.uint<1>
    %in0_0 = subindex %in0[0] : !firrtl.vector<uint<1>,2>
    %in1_0 = subindex %in1[0] : !firrtl.vector<uint<1>,2>
    %in2_1 = subindex %in2[1] : !firrtl.vector<uint<1>,3>
    %out_1 = subindex %out[1] : !firrtl.vector<uint<1>,2>
    %1 = mux(%w, %in1_0, %in2_1)  : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %2 = mux(%w, %in0_0, %1)  : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %out_1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{revisitOps.{inner2.out[1] <- inner2.in2[1] <- x <- inner2.out[1]}}}
  module @revisitOps() {
    %in0, %in1, %in2, %out = instance inner2 @thru(in in0: !firrtl.vector<uint<1>,2>, in in1: !firrtl.vector<uint<1>,2>, in in2: !firrtl.vector<uint<1>,3>, out out: !firrtl.vector<uint<1>,2>)
    %in1_0 = subindex %in1[0] : !firrtl.vector<uint<1>,2>
    %in2_1 = subindex %in2[1] : !firrtl.vector<uint<1>,3>
    %out_1 = subindex %out[1] : !firrtl.vector<uint<1>,2>
    %x = wire  : !firrtl.uint<1>
    connect %in2_1, %x : !firrtl.uint<1>, !firrtl.uint<1>
    connect %x, %out_1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Comb path from ground type to aggregate.
// CHECK-NOT: circuit "scalarToVec"
firrtl.circuit "scalarToVec"   {
  module @thru(in %in1: !firrtl.uint<1>, in %in2: !firrtl.vector<uint<1>,3>, out %out: !firrtl.vector<uint<1>,2>) {
    %out_1 = subindex %out[1] : !firrtl.vector<uint<1>,2>
    connect %out_1, %in1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // expected-error @below {{scalarToVec.{inner2.in1 <- x <- inner2.out[1] <- inner2.in1}}}
  module @scalarToVec() {
    %in1_0, %in2, %out = instance inner2 @thru(in in1: !firrtl.uint<1>, in in2: !firrtl.vector<uint<1>,3>, out out: !firrtl.vector<uint<1>,2>)
    //%in1_0 = subindex %in1[0] : !firrtl.vector<uint<1>,2>
    %out_1 = subindex %out[1] : !firrtl.vector<uint<1>,2>
    %x = wire  : !firrtl.uint<1>
    connect %in1_0, %x : !firrtl.uint<1>, !firrtl.uint<1>
    connect %x, %out_1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Check diagnostic produced if can't name anything on cycle.
// CHECK-NOT: circuit "CycleWithoutNames"
firrtl.circuit "CycleWithoutNames"   {
  // expected-error @below {{detected combinational cycle in a FIRRTL module, but unable to find names for any involved values.}}
  module @CycleWithoutNames() {
    // expected-note @below {{cycle detected here}}
    %0 = wire  : !firrtl.uint<1>
    strictconnect %0, %0 : !firrtl.uint<1>
  }
}

// -----

// Check diagnostic if starting point of detected cycle can't be named.
// Try to find something in the cycle we can name and start there.
firrtl.circuit "CycleStartsUnnammed"   {
  // expected-error @below {{sample path: CycleStartsUnnammed.{n <- ... <- n}}}
  module @CycleStartsUnnammed() {
    %0 = wire  : !firrtl.uint<1>
    %n = node %0 : !firrtl.uint<1>
    strictconnect %0, %n : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "CycleThroughForceable"   {
  // expected-error @below {{sample path: CycleThroughForceable.{w <- n <- w}}}
  module @CycleThroughForceable() {
    %w, %w_ref = wire forceable : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    %n, %n_ref = node %w forceable : !firrtl.uint<1>
    strictconnect %w, %n : !firrtl.uint<1>
  }
}
