// RUN: circt-opt -pass-pipeline='builtin.module(firrtl-imdeadcodeelim)' --split-input-file -verify-diagnostics %s | FileCheck %s
firrtl.circuit "top" {
  // In `dead_module`, %source is connected to %dest through several dead operations such as
  // node, wire, reg or rgereset. %dest is also dead at any instantiation, so check that
  // all operations are removed by IMDeadCodeElim pass.
  // CHECK-NOT: @dead_module
  module private @dead_module(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>,
                                     in %clock:!firrtl.clock, in %reset:!firrtl.uint<1>) {
    %dead_node = node %source: !firrtl.uint<1>

    %dead_wire = wire : !firrtl.uint<1>
    strictconnect %dead_wire, %dead_node : !firrtl.uint<1>

    %dead_reg = reg %clock : !firrtl.clock, !firrtl.uint<1>
    strictconnect %dead_reg, %dead_wire : !firrtl.uint<1>

    %dead_reg_reset = regreset %clock, %reset, %dead_reg  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    strictconnect %dead_reg_reset, %dead_reg : !firrtl.uint<1>

    %not = not %dead_reg_reset : (!firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %dest, %not : !firrtl.uint<1>
  }

  // `%dontTouch` port has a symbol so it shouldn't be removed. `%sym_wire` also has a
  // symbol so check  that `%source` is preserved too.
  // CHECK-LABEL: module private @dontTouch(in %dontTouch: !firrtl.uint<1> sym @sym, in %source: !firrtl.uint<1>) {
  module private @dontTouch(in %dontTouch: !firrtl.uint<1> sym @sym, in %source: !firrtl.uint<1>, in %dead: !firrtl.uint<1>) {
    // CHECK-NEXT: %sym_wire = wire sym @sym2   : !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %sym_wire, %source : !firrtl.uint<1>
    // CHECK-NEXT: }
    %sym_wire = wire sym @sym2 : !firrtl.uint<1>
    strictconnect %sym_wire, %source : !firrtl.uint<1>

  }

  // CHECK-LABEL: module private @mem(in %source: !firrtl.uint<1>) {
  module private @mem(in %source: !firrtl.uint<1>) {
    // CHECK-NEXT: %ReadMemory_read0 = mem Undefined {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}], depth = 16 : i64, name = "ReadMemory", portNames = ["read0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    %mem = mem Undefined {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}], depth = 16 : i64, name = "ReadMemory", portNames = ["read0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    // CHECK-NEXT: %0 = subfield %ReadMemory_read0[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    // CHECK-NEXT: connect %0, %source : !firrtl.uint<4>, !firrtl.uint<1>
    // CHECK-NEXT: }
    %0 = subfield %mem[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    connect %0, %source : !firrtl.uint<4>, !firrtl.uint<1>
  }

  // Ports of public modules should not be modified.
  // CHECK-LABEL: module @top(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
  module @top(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>,
                     in %clock:!firrtl.clock, in %reset:!firrtl.uint<1>) {
    // CHECK-NEXT: %tmp = node %source
    // CHECK-NEXT: strictconnect %dest, %tmp
    %tmp = node %source: !firrtl.uint<1>
    strictconnect %dest, %tmp : !firrtl.uint<1>

    // CHECK-NOT: @dead_module
    %source1, %dest1, %clock1, %reset1  = instance dead_module @dead_module(in source: !firrtl.uint<1>, out dest: !firrtl.uint<1>, in clock:!firrtl.clock, in reset:!firrtl.uint<1>)
    strictconnect %source1, %source : !firrtl.uint<1>
    strictconnect %clock1, %clock : !firrtl.clock
    strictconnect %reset1, %reset : !firrtl.uint<1>

    // Check that ports with dontTouch are not removed.
    // CHECK-NEXT: %testDontTouch_dontTouch, %testDontTouch_source = instance testDontTouch @dontTouch(in dontTouch: !firrtl.uint<1>, in source: !firrtl.uint<1>)
    // CHECK-NEXT: strictconnect %testDontTouch_dontTouch, %source
    // CHECK-NEXT: strictconnect %testDontTouch_source, %source
    %testDontTouch_dontTouch, %testDontTouch_source,  %dead = instance testDontTouch @dontTouch(in dontTouch: !firrtl.uint<1>, in source: !firrtl.uint<1>, in dead:!firrtl.uint<1>)
    strictconnect %testDontTouch_dontTouch, %source : !firrtl.uint<1>
    strictconnect %testDontTouch_source, %source : !firrtl.uint<1>
    strictconnect %dead, %source : !firrtl.uint<1>

    // CHECK-NEXT: %mem_source = instance mem @mem(in source: !firrtl.uint<1>)
    // CHECK-NEXT: strictconnect %mem_source, %source : !firrtl.uint<1>
    %mem_source  = instance mem @mem(in source: !firrtl.uint<1>)
    strictconnect %mem_source, %source : !firrtl.uint<1>
    // CHECK-NEXT: }
  }
}

// -----

// Check that it's possible to analyze complex dependency across different modules.
firrtl.circuit "top"  {
  // CHECK-NOT: @Child1
  module private @Child1(in %input: !firrtl.uint<1>, out %output: !firrtl.uint<1>) {
    strictconnect %output, %input : !firrtl.uint<1>
  }
  // CHECK-NOT: @Child2
  module private @Child2(in %input: !firrtl.uint<1>, in %clock: !firrtl.clock, out %output: !firrtl.uint<1>) {
    %r = reg %clock  : !firrtl.clock, !firrtl.uint<1>
    strictconnect %r, %input : !firrtl.uint<1>
    strictconnect %output, %r : !firrtl.uint<1>
  }

  // CHECK-LABEL: module @top(in %clock: !firrtl.clock, in %input: !firrtl.uint<1>) {
  // CHECK-NEXT:  }
  // expected-warning @+1 {{module `top` is empty but cannot be removed because the module is public}}
  module @top(in %clock: !firrtl.clock, in %input: !firrtl.uint<1>) {
    %tile_input, %tile_output = instance tile  @Child1(in input: !firrtl.uint<1>, out output: !firrtl.uint<1>)
    strictconnect %tile_input, %input : !firrtl.uint<1>
    %named = node  %tile_output  : !firrtl.uint<1>
    %bar_input, %bar_clock, %bar_output = instance bar  @Child2(in input: !firrtl.uint<1>, in clock: !firrtl.clock, out output: !firrtl.uint<1>)
    strictconnect %bar_clock, %clock : !firrtl.clock
    strictconnect %bar_input, %named : !firrtl.uint<1>
  }
}

// -----

// CHECK-LABEL: "UnusedOutput"
firrtl.circuit "UnusedOutput"  {
  // CHECK: module {{.+}}@SingleDriver
  // CHECK-NOT:     out %c
  module private @SingleDriver(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
    // CHECK-NEXT: %[[c_wire:.+]] = wire
    // CHECK-NEXT: strictconnect %b, %[[c_wire]]
    strictconnect %b, %c : !firrtl.uint<1>
    // CHECK-NEXT: %[[not_a:.+]] = not %a
    %0 = not %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %[[c_wire]], %[[not_a]]
    strictconnect %c, %0 : !firrtl.uint<1>
  }
  // CHECK-LABEL: @UnusedOutput
  module @UnusedOutput(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    // CHECK: %singleDriver_a, %singleDriver_b = instance singleDriver
    %singleDriver_a, %singleDriver_b, %singleDriver_c = instance singleDriver @SingleDriver(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    strictconnect %singleDriver_a, %a : !firrtl.uint<1>
    strictconnect %b, %singleDriver_b : !firrtl.uint<1>
  }
}

// -----

// Ensure that the "output_file" attribute isn't destroyed by IMDeadCodeElim.
// This matters for interactions between Grand Central (which sets these) and
// IMDeadCodeElim which may clone modules with stripped ports.
//
// CHECK-LABEL: "PreserveOutputFile"
firrtl.circuit "PreserveOutputFile" {
  // CHECK-NEXT: module {{.+}}@Sub
  // CHECK-NOT:    %a
  // CHECK-SAME:   output_file
  // expected-warning @+1{{module `Sub` is empty but cannot be removed because the module has ports "b" are referenced by name or dontTouched}}
  module private @Sub(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1> sym @sym) attributes {output_file = #hw.output_file<"hello">} {}
  // CHECK: module @PreserveOutputFile
  module @PreserveOutputFile() {
    // CHECK-NEXT: instance sub
    // CHECK-SAME: output_file
    instance sub {output_file = #hw.output_file<"hello">} @Sub(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>)
  }
}

// -----

// CHECK-LABEL: "DeleteEmptyModule"
firrtl.circuit "DeleteEmptyModule" {
  // CHECK: module private @empty
  // expected-warning @+1{{module `empty` is empty but cannot be removed because the module has annotations [{class = "foo"}]}}
  module private @empty() attributes {annotations = [{class = "foo"}]}  {}
  // CHECK-NOT: module private @Sub
  module private @Sub(in %a: !firrtl.uint<1>)  {}
  // CHECK: module @DeleteEmptyModule
  module @DeleteEmptyModule() {
    // CHECK-NOT: instance sub1
    instance sub1 sym @Foo @Sub(in a: !firrtl.uint<1>)
    // CHECK-NOT: sub2
    instance sub2 @Sub(in a: !firrtl.uint<1>)
    // CHECK: empty
    instance empty @empty()
  }
}

// -----

// CHECK-LABEL: "ForwardConstant"
firrtl.circuit "ForwardConstant" {
  // CHECK-NOT: Zero
  module private @Zero(out %zero: !firrtl.uint<1>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    strictconnect %zero, %c0_ui1 : !firrtl.uint<1>
  }
  // CHECK-LABEL: @ForwardConstant
  module @ForwardConstant(out %zero: !firrtl.uint<1>) {
    // CHECK: %c0_ui1 = constant 0
    %sub_zero = instance sub @Zero(out zero: !firrtl.uint<1>)
    // CHECK-NEXT: strictconnect %zero, %c0_ui1
    strictconnect %zero, %sub_zero : !firrtl.uint<1>
  }
}

// -----

// Test handling of ref ports and ops.

// CHECK-LABEL: "RefPorts"
firrtl.circuit "RefPorts" {
  // CHECK-NOT: @dead_ref_send
  module private @dead_ref_send(in %source: !firrtl.uint<1>, out %dest: !firrtl.probe<uint<1>>) {
    %ref = ref.send %source: !firrtl.uint<1>
    ref.define %dest, %ref : !firrtl.probe<uint<1>>
  }

  // CHECK-LABEL: @dead_ref_port
  // CHECK-NOT: ref
  module private @dead_ref_port(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>, out %ref_dest: !firrtl.probe<uint<1>>) {
    %ref_not = ref.send %source: !firrtl.uint<1>
    ref.define %ref_dest, %ref_not : !firrtl.probe<uint<1>>
    strictconnect %dest, %source : !firrtl.uint<1>
  }

  // CHECK: @live_ref
  module private @live_ref(in %source: !firrtl.uint<1>, out %dest: !firrtl.probe<uint<1>>) {
    %ref_source = ref.send %source: !firrtl.uint<1>
    ref.define %dest, %ref_source : !firrtl.probe<uint<1>>
  }

  // CHECK-LABEL: @RefPorts
  module @RefPorts(in %source : !firrtl.uint<1>, out %dest : !firrtl.uint<1>) {
    // Delete send's that aren't resolved, and check deletion of modules with ref ops + ports.
    // CHECK-NOT: @dead_ref_send
    %source1, %dest1 = instance dead_ref_send @dead_ref_send(in source: !firrtl.uint<1>, out dest: !firrtl.probe<uint<1>>)
    strictconnect %source1, %source : !firrtl.uint<1>

    // Check that an unused resolve doesn't keep send alive, and test ref port removal.
    // CHECK: @dead_ref_port
    // CHECK-NOT: ref
    %source2, %dest2, %ref_dest2 = instance dead_ref_port @dead_ref_port(in source: !firrtl.uint<1>, out dest: !firrtl.uint<1>, out ref_dest: !firrtl.probe<uint<1>>)
    strictconnect %source2, %source : !firrtl.uint<1>
    %unused = ref.resolve %ref_dest2 : !firrtl.probe<uint<1>>
    strictconnect %dest, %dest2 : !firrtl.uint<1>

    // Check not deleted if live.
    // CHECK: @live_ref
    %source3, %dest3 = instance live_ref @live_ref(in source: !firrtl.uint<1>, out dest: !firrtl.probe<uint<1>>)
    strictconnect %source3, %source : !firrtl.uint<1>
    // CHECK: ref.resolve
    %dest3_resolved = ref.resolve %dest3 : !firrtl.probe<uint<1>>
    strictconnect %dest, %dest3_resolved : !firrtl.uint<1>

    // Check dead resolve is deleted.
    // CHECK-NOT: dead_instance
    %source4, %dest4 = instance dead_instance @live_ref(in source: !firrtl.uint<1>, out dest: !firrtl.probe<uint<1>>)
    strictconnect %source4, %source : !firrtl.uint<1>
    // CHECK-NOT: ref.resolve
    %unused5 = ref.resolve %dest4 : !firrtl.probe<uint<1>>
  }
}

// -----

// Test the removal of memories in dead cycles

firrtl.circuit "MemoryInDeadCycle" {
  // CHECK-LABEL: module public @MemoryInDeadCycle
  // expected-warning @+1{{module `MemoryInDeadCycle` is empty but cannot be removed because the module is public}}
  module public @MemoryInDeadCycle(in %clock: !firrtl.clock, in %addr: !firrtl.uint<4>) {

    // CHECK-NOT: mem
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %Memory_r = mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["read"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>

    %r_addr = subfield %Memory_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %r_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %r_en = subfield %Memory_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %r_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %r_clk = subfield %Memory_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %r_clk, %clock : !firrtl.clock, !firrtl.clock

    // CHECK-NOT: mem
    %Memory_w = mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["w"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>

    %w_addr = subfield %Memory_w[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %w_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %w_en = subfield %Memory_w[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %w_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %w_clk = subfield %Memory_w[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %w_clk, %clock : !firrtl.clock, !firrtl.clock
    %w_mask = subfield %Memory_w[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    connect %w_mask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    %w_data = subfield %Memory_w[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    %r_data = subfield %Memory_r[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    connect %w_data, %r_data : !firrtl.uint<42>, !firrtl.uint<42>
  }
}

// -----
// CHECK-LABEL: circuit "DeadInputPort"
firrtl.circuit "DeadInputPort"  {
  // CHECK-NOT: module private @Bar
  module private @Bar(in %a: !firrtl.uint<1>) {
  }

  // CHECK-LABEL: module @DeadInputPort
  module @DeadInputPort(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    // CHECK-NEXT: %0 = wire
    // CHECK-NEXT: strictconnect %0, %a
    // CHECK-NEXT: strictconnect %b, %0
    %bar_a = instance bar  @Bar(in a: !firrtl.uint<1>)
    strictconnect %bar_a, %a : !firrtl.uint<1>
    strictconnect %b, %bar_a : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "DeleteInstance" {
  // CHECK-NOT: @InvalidValue
  module private @InvalidValue() {
      %invalid_ui289 = invalidvalue : !firrtl.uint<289>
  }
  module private @SideEffect1(in %a: !firrtl.uint<1>, in %clock: !firrtl.clock) {
    printf %clock, %a, "foo"  : !firrtl.clock, !firrtl.uint<1>
  }
  module private @SideEffect2(in %a: !firrtl.uint<1>, in %clock: !firrtl.clock) {
    %s1_a, %s1_clock = instance s1 @SideEffect1(in a: !firrtl.uint<1>, in clock: !firrtl.clock)
    strictconnect %s1_a, %a : !firrtl.uint<1>
    strictconnect %s1_clock, %clock : !firrtl.clock
  }
  module private @PassThrough(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    strictconnect %b, %a : !firrtl.uint<1>
  }
  // CHECK-LABEL: DeleteInstance
  module @DeleteInstance(in %a: !firrtl.uint<1>, in %clock: !firrtl.clock, out %b: !firrtl.uint<1>) {
    // CHECK-NOT: inv
    instance inv @InvalidValue()
    // CHECK-NOT: p1
    // CHECK: instance p2 @PassThrough
    // CHECK-NEXT: instance s @SideEffect2
    %p1_a, %p1_b = instance p1 @PassThrough(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    %p2_a, %p2_b = instance p2 @PassThrough(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    %s_a, %s_clock = instance s @SideEffect2(in a: !firrtl.uint<1>, in clock: !firrtl.clock)
    // CHECK-NEXT: strictconnect %s_a, %a : !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %s_clock, %clock : !firrtl.clock
    // CHECK-NEXT: strictconnect %p2_a, %a : !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %b, %p2_b : !firrtl.uint<1>
    strictconnect %s_a, %a : !firrtl.uint<1>
    strictconnect %s_clock, %clock : !firrtl.clock
    strictconnect %p1_a, %a : !firrtl.uint<1>
    strictconnect %p2_a, %a : !firrtl.uint<1>
    strictconnect %b, %p2_b : !firrtl.uint<1>
  }
}

// -----
firrtl.circuit "Top" {
  // CHECK-NOT: @nla_1
  // CHECK: @nla_2
  hw.hierpath private @nla_1 [@Foo1::@dead, @EncodingModule]
  hw.hierpath private @nla_2 [@Foo2::@live, @EncodingModule]
  // CHECK-LABEL private @EncodingModule
  // CHECK-NOT: @nla_1
  // CHECK-SAME @nla_2
  module private @EncodingModule(in %in: !firrtl.uint<1>, out %a: !firrtl.uint<1> [{circt.nonlocal = @nla_1, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0 : i64, type = "OMReferenceTarget"}, {circt.nonlocal = @nla_2, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 1 : i64, type = "OMReferenceTarget"}]) {
    strictconnect %a, %in : !firrtl.uint<1>
  }
  // CHECK-NOT: @Foo1
  module private @Foo1(in %in: !firrtl.uint<1>) {
    %c_in, %c_a = instance c sym @dead @EncodingModule(in in: !firrtl.uint<1>, out a: !firrtl.uint<1>)
    strictconnect %c_in, %in : !firrtl.uint<1>
  }
  // CHECK-LABEL: @Foo2
  module private @Foo2(in %in: !firrtl.uint<1>, out %a: !firrtl.uint<1>) {
    %c_in, %c_a = instance c sym @live @EncodingModule(in in: !firrtl.uint<1>, out a: !firrtl.uint<1>)
    strictconnect %a, %c_a : !firrtl.uint<1>
    strictconnect %c_in, %in : !firrtl.uint<1>
  }
  // CHECK-LABEL: @Top
  // CHECK-NOT: @Foo1
  // CHECK-NOT: strictconnect %foo1_in, %in
  // CHECK: @Foo2
  module @Top(in %in: !firrtl.uint<1>, out %a: !firrtl.uint<1>) {
    %foo1_in = instance foo1 @Foo1(in in: !firrtl.uint<1>)
    strictconnect %foo1_in, %in : !firrtl.uint<1>
    %foo2_in, %foo2_a = instance foo2 @Foo2(in in: !firrtl.uint<1>, out a: !firrtl.uint<1>)
    strictconnect %a, %foo2_a : !firrtl.uint<1>
    strictconnect %foo2_in, %in : !firrtl.uint<1>

  }
}

// -----

firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @nla_1
  hw.hierpath private @nla_1 [@Top::@foo1, @Bar::@w]
  // CHECK-NEXT: hw.hierpath private @nla_2
  hw.hierpath private @nla_2 [@Top::@foo1, @Bar]
  // CHECK-NEXT: sv.verbatim "foo" {some = [@nla_2]}
  sv.verbatim "foo" {some = [@nla_2]}
  // CHECK-LABEL: module private @Bar
  // CHECK: %in1{{.*}}sym @w
  // CHECK-SAME: %in2
  // CHECK-NOT: %in3
  // expected-warning @+1 {{module `Bar` is empty but cannot be removed because the module has ports "in1", "in2" are referenced by name or dontTouched}}
  module private @Bar(in %in1 : !firrtl.uint<1> sym @w, in %in2: !firrtl.uint<1> [{class = "foo"}], in %in3: !firrtl.uint<1>) {}
  // CHECK-LABEL: module private @Baz
  // expected-warning @+1 {{module `Baz` is empty but cannot be removed because an instance is referenced by nam}}
  module private @Baz() {}

  // CHECK-LABEL: module @Top
  module @Top(in %in: !firrtl.uint<1>) {
    %c_in1, %c_in2, %c_in3 = instance c sym @foo1 @Bar(in in1: !firrtl.uint<1>, in in2: !firrtl.uint<1>, in in3: !firrtl.uint<1>)
    strictconnect %c_in1, %in : !firrtl.uint<1>
    strictconnect %c_in2, %in : !firrtl.uint<1>
    strictconnect %c_in3, %in : !firrtl.uint<1>
    // CHECK: sv.verbatim "foo" {some = #hw.innerNameRef<@Top::@baz1>}
    sv.verbatim "foo" {some = #hw.innerNameRef<@Top::@baz1>}
    // Don't remove the instance if there is an unknown use of inner reference.
    // CHECK: baz1
    // expected-note @+1 {{these are instances with symbols}}
    instance baz1 sym @baz1 @Baz()
    // Remove a dead instance otherwise.
    // CHECK-NOT: baz2
    instance baz2 sym @baz2 @Baz()
  }
}
