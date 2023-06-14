// RUN: firtool --hw --split-input-file --verify-diagnostics %s
// These will be picked up by https://github.com/llvm/circt/pull/1444

// Tests extracted from:
// - test/scala/firrtlTests/AsyncResetSpec.scala

firrtl.circuit "Foo" {
  // expected-note @+1 {{reset value defined here:}}
  module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %v: !firrtl.uint<8>) {
    // Constant check should see through subfield connects.
    %bundle0 = wire : !firrtl.bundle<a: uint<8>>
    %bundle0.a = subfield %bundle0[0] : !firrtl.bundle<a: uint<8>>
    connect %bundle0.a, %v : !firrtl.uint<8>, !firrtl.uint<8>
    // expected-error @+2 {{register with async reset requires constant reset value}}
    // expected-error @+1 {{'firrtl.regreset' op LowerToHW couldn't handle this operation}}
    %2 = regreset %clock, %reset, %bundle0 : !firrtl.clock, !firrtl.asyncreset, !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{reset value defined here:}}
  module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %v: !firrtl.uint<8>) {
    // Constant check should see through multiple connect hops.
    %bundle0 = wire : !firrtl.bundle<a: uint<8>>
    %bundle0.a = subfield %bundle0[0] : !firrtl.bundle<a: uint<8>>
    connect %bundle0.a, %v : !firrtl.uint<8>, !firrtl.uint<8>
    %bundle1 = wire : !firrtl.bundle<a: uint<8>>
    connect %bundle1, %bundle0 : !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>
    // expected-error @+2 {{register with async reset requires constant reset value}}
    // expected-error @+1 {{'firrtl.regreset' op LowerToHW couldn't handle this operation}}
    %3 = regreset %clock, %reset, %bundle1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{reset value defined here:}}
  module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %v: !firrtl.uint<8>) {
    // Constant check should see through subindex connects.
    %vector0 = wire : !firrtl.vector<uint<8>, 1>
    %vector0.a = subindex %vector0[0] : !firrtl.vector<uint<8>, 1>
    connect %vector0.a, %v : !firrtl.uint<8>, !firrtl.uint<8>
    // expected-error @+2 {{register with async reset requires constant reset value}}
    // expected-error @+1 {{'firrtl.regreset' op LowerToHW couldn't handle this operation}}
    %4 = regreset %clock, %reset, %vector0 : !firrtl.clock, !firrtl.asyncreset, !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>
  }
}

// -----

firrtl.circuit "Foo" {
  // expected-note @+1 {{reset value defined here:}}
  module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %v: !firrtl.uint<8>) {
    // Constant check should see through multiple connect hops.
    %vector0 = wire : !firrtl.vector<uint<8>, 1>
    %vector0.a = subindex %vector0[0] : !firrtl.vector<uint<8>, 1>
    connect %vector0.a, %v : !firrtl.uint<8>, !firrtl.uint<8>
    %vector1 = wire : !firrtl.vector<uint<8>, 1>
    connect %vector1, %vector0 : !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>
    // expected-error @+2 {{register with async reset requires constant reset value}}
    // expected-error @+1 {{'firrtl.regreset' op LowerToHW couldn't handle this operation}}
    %5 = regreset %clock, %reset, %vector1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>
  }
}

// -----

// Hidden Non-literals should NOT be allowed as reset values for AsyncReset
firrtl.circuit "Foo" {
  // expected-note @+1 {{reset value defined here:}}
  module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %x: !firrtl.vector<uint<1>, 4>, in %y: !firrtl.uint<1>, out %z: !firrtl.vector<uint<1>, 4>) {
    %literal = wire  : !firrtl.vector<uint<1>, 4>
    %0 = subindex %literal[0] : !firrtl.vector<uint<1>, 4>
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = subindex %literal[1] : !firrtl.vector<uint<1>, 4>
    connect %1, %y : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = subindex %literal[2] : !firrtl.vector<uint<1>, 4>
    connect %2, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %3 = subindex %literal[3] : !firrtl.vector<uint<1>, 4>
    connect %3, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // expected-error @+2 {{register with async reset requires constant reset value}}
    // expected-error @+1 {{'firrtl.regreset' op LowerToHW couldn't handle this operation}}
    %r = regreset %clock, %reset, %literal  : !firrtl.clock, !firrtl.asyncreset, !firrtl.vector<uint<1>, 4>, !firrtl.vector<uint<1>, 4>
    connect %r, %x : !firrtl.vector<uint<1>, 4>, !firrtl.vector<uint<1>, 4>
    connect %z, %r : !firrtl.vector<uint<1>, 4>, !firrtl.vector<uint<1>, 4>
  }
}

// -----

// Wire connected to non-literal should NOT be allowed as reset values for AsyncReset
firrtl.circuit "Foo"   {
  module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %x: !firrtl.uint<1>, in %y: !firrtl.uint<1>, in %cond: !firrtl.uint<1>, out %z: !firrtl.uint<1>) {
    // expected-note @+1 {{reset value defined here:}}
    %w = wire  : !firrtl.uint<1>
    %c1_ui = constant 1 : !firrtl.uint
    connect %w, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    when %cond : !firrtl.uint<1> {
      connect %w, %y : !firrtl.uint<1>, !firrtl.uint<1>
    }
    // expected-error @+2 {{register with async reset requires constant reset value}}
    // expected-error @+1 {{'firrtl.regreset' op LowerToHW couldn't handle this operation}}
    %r = regreset %clock, %reset, %w  : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
    connect %r, %x : !firrtl.uint<1>, !firrtl.uint<1>
    connect %z, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }
}
