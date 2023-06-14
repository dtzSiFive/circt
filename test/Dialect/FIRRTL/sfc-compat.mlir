// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-sfc-compat)))' --verify-diagnostics --split-input-file %s | FileCheck %s

firrtl.circuit "SFCCompatTests" {

  module @SFCCompatTests() {}

  // An invalidated regreset should be converted to a reg.
  //
  // CHECK-LABEL: @InvalidValue
  module @InvalidValue(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    // CHECK-NOT: invalid
    %invalid_ui1_dead = invalidvalue : !firrtl.uint<1>
    %invalid_ui1 = invalidvalue : !firrtl.uint<1>
    // CHECK: reg %clock
    %r = regreset %clock, %reset, %invalid_ui1  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalidated through a wire should be converted to a reg.
  //
  // CHECK-LABEL: @InvalidThroughWire
  module @InvalidThroughWire(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %inv = wire  : !firrtl.uint<1>
    %invalid_ui1 = invalidvalue : !firrtl.uint<1>
    connect %inv, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: reg %clock
    %r = regreset %clock, %reset, %inv  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalidated through wires with aggregate types should be
  // converted to a reg.
  //
  // CHECK-LABEL: module @AggregateInvalidThroughWire
  module @AggregateInvalidThroughWire(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.vector<bundle<a: uint<1>>, 2>, out %q: !firrtl.vector<bundle<a: uint<1>>, 2>) {
    %inv = wire : !firrtl.bundle<a: uint<1>>
    %inv_a = subfield %inv[a] : !firrtl.bundle<a: uint<1>>
    %invalid = invalidvalue : !firrtl.uint<1>
    strictconnect %inv_a, %invalid : !firrtl.uint<1>

    %inv1 = wire : !firrtl.vector<bundle<a: uint<1>>, 2>
    %inv1_0 = subindex %inv1[0] : !firrtl.vector<bundle<a: uint<1>>, 2>
    strictconnect %inv1_0, %inv : !firrtl.bundle<a: uint<1>>
    %inv1_1 = subindex %inv1[0] : !firrtl.vector<bundle<a: uint<1>>, 2>
    strictconnect %inv1_1, %inv : !firrtl.bundle<a: uint<1>>

    // CHECK: reg %clock : !firrtl.clock, !firrtl.vector<bundle<a: uint<1>>, 2>
    %r = regreset %clock, %reset, %inv1  : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<bundle<a: uint<1>>, 2>, !firrtl.vector<bundle<a: uint<1>>, 2>
    strictconnect %r, %d : !firrtl.vector<bundle<a: uint<1>>, 2>
    strictconnect %q, %r : !firrtl.vector<bundle<a: uint<1>>, 2>
  }

  // A regreset invalidated via an output port should be converted to a reg.
  //
  // CHECK-LABEL: @InvalidPort
  module @InvalidPort(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>, out %x: !firrtl.uint<1>) {
    %inv = wire  : !firrtl.uint<1>
    %invalid_ui1 = invalidvalue : !firrtl.uint<1>
    connect %inv, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %x, %inv : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: reg %clock
    %r = regreset %clock, %reset, %x  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalidate via an instance input port should be converted to a
  // reg.
  //
  // CHECK-LABEL: @InvalidInstancePort
  module @InvalidInstancePort_Submodule(in %inv: !firrtl.uint<1>) {}
  module @InvalidInstancePort(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %inv = wire  : !firrtl.uint<1>
    %invalid_ui1 = invalidvalue : !firrtl.uint<1>
    connect %inv, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %submodule_inv = instance submodule  @InvalidInstancePort_Submodule(in inv: !firrtl.uint<1>)
    connect %submodule_inv, %inv : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: reg %clock
    %r = regreset %clock, %reset, %submodule_inv  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A primitive operation should block invalid propagation.
  module @InvalidPrimop(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<1>, out %q: !firrtl.uint<1>) {
    %invalid_ui1 = invalidvalue : !firrtl.uint<1>
    %0 = not %invalid_ui1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK: regreset %clock
    %r = regreset %clock, %reset, %0  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    connect %r, %d : !firrtl.uint<1>, !firrtl.uint<1>
    connect %q, %r : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // A regreset invalid value should NOT propagate through a node.
  module @InvalidNode(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %d: !firrtl.uint<8>, out %q: !firrtl.uint<8>) {
    %inv = wire  : !firrtl.uint<8>
    %invalid_ui8 = invalidvalue : !firrtl.uint<8>
    connect %inv, %invalid_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
    %_T = node %inv  : !firrtl.uint<8>
    // CHECK: regreset %clock
    %r = regreset %clock, %reset, %_T  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    connect %r, %d : !firrtl.uint<8>, !firrtl.uint<8>
    connect %q, %r : !firrtl.uint<8>, !firrtl.uint<8>
  }

  module @AggregateInvalid(out %q: !firrtl.bundle<a:uint<1>>) {
    %invalid_ui1 = invalidvalue : !firrtl.bundle<a:uint<1>>
    connect %q, %invalid_ui1 : !firrtl.bundle<a:uint<1>>, !firrtl.bundle<a:uint<1>>
    // CHECK: %c0_ui1 = constant 0
    // CHECK-NEXT: %[[CAST:.+]] = bitcast %c0_ui1
    // CHECK-NEXT: %q, %[[CAST]]
  }

  // All of these should not error as the register is initialzed to a constant
  // reset value while looking through constructs that the SFC allows.  This is
  // testing the following cases:
  //
  //   1. A wire marked don't touch driven to a constant.
  //   2. A node driven to a constant.
  //   3. A wire driven to an invalid.
  //   4. A constant that passes through SFC-approved primops.
  //
  // CHECK-LABEL: module @ConstantAsyncReset
  module @ConstantAsyncReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %r0_init = wire sym @r0_init : !firrtl.uint<1>
    strictconnect %r0_init, %c0_ui1 : !firrtl.uint<1>
    %r0 = regreset %clock, %reset, %r0_init : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>

    %r1_init = node %c0_ui1 : !firrtl.uint<1>
    %r1 = regreset %clock, %reset, %r1_init : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>

    %inv_ui1 = invalidvalue : !firrtl.uint<1>
    %r2_init = wire : !firrtl.uint<1>
    strictconnect %r2_init, %inv_ui1 : !firrtl.uint<1>
    %r2 = regreset %clock, %reset, %r2_init : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>

    %c0_si1 = asSInt %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.sint<1>
    %c0_clock = asClock %c0_si1 : (!firrtl.sint<1>) -> !firrtl.clock
    %c0_asyncreset = asAsyncReset %c0_clock : (!firrtl.clock) -> !firrtl.asyncreset
    %r3_init = asUInt %c0_asyncreset : (!firrtl.asyncreset) -> !firrtl.uint<1>
    %r3 = regreset %clock, %reset, %r3_init : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: module @TailPrimOp
  module @TailPrimOp(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %0 = pad %c0_ui1, 3 : (!firrtl.uint<1>) -> !firrtl.uint<3>
    %1 = tail %0, 2 : (!firrtl.uint<3>) -> !firrtl.uint<1>
    %r0_init = wire sym @r0_init : !firrtl.uint<1>
    strictconnect %r0_init, %1: !firrtl.uint<1>
    %r0 = regreset %clock, %reset, %r0_init : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "NonConstantAsyncReset_Port" {
  // expected-note @below {{reset driver is "x"}}
  module @NonConstantAsyncReset_Port(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %x: !firrtl.uint<1>) {
    // expected-error @below {{register "r0" has an async reset, but its reset value "x" is not driven with a constant value through wires, nodes, or connects}}
    %r0 = regreset %clock, %reset, %x : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "NonConstantAsyncReset_PrimOp" {
  module @NonConstantAsyncReset_PrimOp(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    // expected-note @+1 {{reset driver is here}}
    %c1_ui1 = not %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // expected-error @below {{register "r0" has an async reset, but its reset value is not driven with a constant value through wires, nodes, or connects}}
    %r0 = regreset %clock, %reset, %c1_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "NonConstantAsyncReset_Aggregate0" {
  // expected-note @below {{reset driver is "x"}}
  module @NonConstantAsyncReset_Aggregate0(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %x : !firrtl.vector<uint<1>, 2>) {
    %value = wire : !firrtl.vector<uint<1>, 2>
    strictconnect %value, %x : !firrtl.vector<uint<1>, 2>
    // expected-error @below {{register "r0" has an async reset, but its reset value "value" is not driven with a constant value through wires, nodes, or connects}}
    %r0 = regreset %clock, %reset, %value : !firrtl.clock, !firrtl.asyncreset, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
}

// -----

firrtl.circuit "NonConstantAsyncReset_Aggregate1" {
  // expected-note @below {{reset driver is "x[0].y"}}
  module @NonConstantAsyncReset_Aggregate1(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %x : !firrtl.vector<bundle<y: uint<1>>, 1>) {

    // Aggregate wire used for the reset value.
    %value = wire : !firrtl.vector<uint<1>, 2>

    // Connect a constant 0 to value[0].
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %value_0 = subindex %value[0] : !firrtl.vector<uint<1>, 2>
    strictconnect %value_0, %c0_ui1 : !firrtl.uint<1>

    // Connect a complex chain of operations leading to the port to value[1].
    %subindex = subindex %x[0] : !firrtl.vector<bundle<y : uint<1>>, 1>
    %node = node %subindex : !firrtl.bundle<y : uint<1>>
    %subfield = subfield %node[y] : !firrtl.bundle<y : uint<1>>
    %value_1 = subindex %value[1] : !firrtl.vector<uint<1>, 2>
    strictconnect %value_1, %subfield : !firrtl.uint<1>

    // expected-error @below {{register "r0" has an async reset, but its reset value "value[1]" is not driven with a constant value through wires, nodes, or connects}}
    %r0 = regreset %clock, %reset, %value : !firrtl.clock, !firrtl.asyncreset, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  }
}
