// RUN: firtool --ir-fir %s | FileCheck %s
// Tests extracted from:
// - test/scala/firrtlTests/AsyncResetSpec.scala

// The following should not error.
// CHECK-LABEL: module @AsyncResetConst
firrtl.circuit "AsyncResetConst" {
  module @AsyncResetConst(
    in %clock: !firrtl.clock,
    in %reset: !firrtl.asyncreset,
    out %z0: !firrtl.uint<8>,
    out %z1: !firrtl.uint<8>,
    out %z2: !firrtl.bundle<a: uint<8>>,
    out %z3: !firrtl.bundle<a: uint<8>>,
    out %z4: !firrtl.vector<uint<8>, 1>,
    out %z5: !firrtl.vector<uint<8>, 1>
  ) {
    // Constant check should handle trivial cases.
    %c0_ui = constant 0 : !firrtl.uint<8>
    %0 = regreset %clock, %reset, %c0_ui : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>

    // Constant check should see through nodes.
    %node = node %c0_ui : !firrtl.uint<8>
    %1 = regreset %clock, %reset, %node : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>

    // Constant check should see through subfield connects.
    %bundle0 = wire : !firrtl.bundle<a: uint<8>>
    %bundle0.a = subfield %bundle0[a] : !firrtl.bundle<a: uint<8>>
    connect %bundle0.a, %c0_ui : !firrtl.uint<8>, !firrtl.uint<8>
    %2 = regreset %clock, %reset, %bundle0 : !firrtl.clock, !firrtl.asyncreset, !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>

    // Constant check should see through multiple connect hops.
    %bundle1 = wire : !firrtl.bundle<a: uint<8>>
    connect %bundle1, %bundle0 : !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>
    %3 = regreset %clock, %reset, %bundle1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>

    // Constant check should see through subindex connects.
    %vector0 = wire : !firrtl.vector<uint<8>, 1>
    %vector0.a = subindex %vector0[0] : !firrtl.vector<uint<8>, 1>
    connect %vector0.a, %c0_ui : !firrtl.uint<8>, !firrtl.uint<8>
    %4 = regreset %clock, %reset, %vector0 : !firrtl.clock, !firrtl.asyncreset, !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>

    // Constant check should see through multiple connect hops.
    %vector1 = wire : !firrtl.vector<uint<8>, 1>
    connect %vector1, %vector0 : !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>
    %5 = regreset %clock, %reset, %vector1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>

    connect %z0, %0 : !firrtl.uint<8>, !firrtl.uint<8>
    connect %z1, %1 : !firrtl.uint<8>, !firrtl.uint<8>
    connect %z2, %2 : !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>
    connect %z3, %3 : !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>
    connect %z4, %4 : !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>
    connect %z5, %5 : !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>
  }
}
