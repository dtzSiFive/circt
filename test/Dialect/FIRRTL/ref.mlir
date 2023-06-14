// RUN: circt-opt %s -split-input-file | circt-opt -split-input-file
// RUN: firtool %s -split-input-file
// These tests are just for demonstrating RefOps, and expected to not error.

// Simple 1 level read from wire.
firrtl.circuit "xmr" {
  module private @Test(out %x: !firrtl.probe<uint<2>>) {
    %w = wire : !firrtl.uint<2>
    %zero = constant 0 : !firrtl.uint<2>
    strictconnect %w, %zero : !firrtl.uint<2>
    %1 = ref.send %w : !firrtl.uint<2>
    ref.define %x, %1 : !firrtl.probe<uint<2>>
  }
  module @xmr() {
    %test_x = instance test @Test(out x: !firrtl.probe<uint<2>>)
    %x = ref.resolve %test_x : !firrtl.probe<uint<2>>
  }
}

// -----

// Simple 1 level read from constant.
firrtl.circuit "SimpleRead" {
  module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @SimpleRead() {
    %bar_a = instance bar @Bar(out _a: !firrtl.probe<uint<1>>)
    %a = wire : !firrtl.uint<1>
    %0 = ref.resolve %bar_a : !firrtl.probe<uint<1>>
    strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// Forward module port to instance
firrtl.circuit "ForwardToInstance" {
  module @Bar2(out %_a: !firrtl.probe<uint<1>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %bar_2 = instance bar @Bar2(out _a: !firrtl.probe<uint<1>>)
    ref.define %_a, %bar_2 : !firrtl.probe<uint<1>>
  }
  module @ForwardToInstance() {
    %bar_a = instance bar @Bar(out _a: !firrtl.probe<uint<1>>)
    %a = wire : !firrtl.uint<1>
    %0 = ref.resolve %bar_a : !firrtl.probe<uint<1>>
    strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// Multiple readers, for a single remote value.
firrtl.circuit "ForwardToInstance" {
  module @Bar2(out %_a: !firrtl.probe<uint<1>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1    : !firrtl.probe<uint<1>>
  }
  module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %bar_2 = instance bar @Bar2(out _a: !firrtl.probe<uint<1>>)
    ref.define %_a, %bar_2 : !firrtl.probe<uint<1>>
    // Reader 1
    %0 = ref.resolve %bar_2 : !firrtl.probe<uint<1>>
    %a = wire : !firrtl.uint<1>
    strictconnect %a, %0 : !firrtl.uint<1>
  }
  module @ForwardToInstance() {
    %bar_a = instance bar @Bar(out _a: !firrtl.probe<uint<1>>)
    %a = wire : !firrtl.uint<1>
    // Reader 2
    %0 = ref.resolve %bar_a : !firrtl.probe<uint<1>>
    strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// Two references passed by value.
firrtl.circuit "DUT" {
  module private @Submodule (out %ref_out1: !firrtl.probe<uint<1>>, out %ref_out2: !firrtl.probe<uint<4>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %w_data1 = wire : !firrtl.uint<1>
    strictconnect %w_data1, %zero : !firrtl.uint<1>
    %1 = ref.send %w_data1 : !firrtl.uint<1>
    ref.define %ref_out1, %1 : !firrtl.probe<uint<1>>
    %w_data2 = wire : !firrtl.uint<4>
    %zero4 = constant 0 : !firrtl.uint<4>
    strictconnect %w_data2, %zero4 : !firrtl.uint<4>
    %2 = ref.send %w_data2 : !firrtl.uint<4>
    ref.define %ref_out2, %2 : !firrtl.probe<uint<4>>
  }
  module @DUT() {
    %view_out1, %view_out2 = instance sub @Submodule(out ref_out1: !firrtl.probe<uint<1>>, out ref_out2: !firrtl.probe<uint<4>>)
    %view_in1, %view_in2 = instance MyView_companion @MyView_companion(in ref_in1: !firrtl.uint<1>, in ref_in2: !firrtl.uint<4>)

    %1 = ref.resolve %view_out1 : !firrtl.probe<uint<1>>
    %2 = ref.resolve %view_out2 : !firrtl.probe<uint<4>>
    strictconnect %view_in1, %1 : !firrtl.uint<1>
    strictconnect %view_in2, %2 : !firrtl.uint<4>
  }

  module private @MyView_companion (in %ref_in1: !firrtl.uint<1>, in %ref_in2: !firrtl.uint<4>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %_WIRE = wire sym @_WIRE : !firrtl.uint<1>
    strictconnect %_WIRE, %c0_ui1 : !firrtl.uint<1>
    %iface = sv.interface.instance sym @__MyView_MyInterface__  : !sv.interface<@MyInterface>
  }

  sv.interface @MyInterface {
    sv.verbatim "// a wire called 'bool'" {symbols = []}
    sv.interface.signal @bool : i1
  }
}

// -----

// RefType of aggregates and RefSub. 
firrtl.circuit "RefTypeVector" {
  module @RefTypeVector(in %bundle : !firrtl.bundle<a: uint<1>, b flip: uint<2>>) {
    %zero = constant 0 : !firrtl.uint<4>
    %z = bitcast %zero : (!firrtl.uint<4>) -> !firrtl.vector<uint<1>,4>
    %1 = ref.send %z : !firrtl.vector<uint<1>,4>
    %10 = ref.sub %1[0] : !firrtl.probe<vector<uint<1>,4>>
    %11 = ref.sub %1[1] : !firrtl.probe<vector<uint<1>,4>>
    %a = ref.resolve %10 : !firrtl.probe<uint<1>>
    %b = ref.resolve %11 : !firrtl.probe<uint<1>>
    %b1 = ref.send %bundle : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
    %12 = ref.sub %b1[1] : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>
    %rb = ref.resolve %12 : !firrtl.probe<uint<2>>
    %bundle_b = subfield %bundle[b] : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
    %zero2 = constant 0 : !firrtl.uint<2>
    strictconnect %bundle_b, %zero2 : !firrtl.uint<2>
  }
}

// -----

// https://github.com/llvm/circt/issues/3715
firrtl.circuit "Issue3715" {
  module private @Test(in %p: !firrtl.uint<1>, out %x: !firrtl.probe<uint<2>>) {
    when %p : !firrtl.uint<1> {
      %zero = constant 1 : !firrtl.uint<2>
      %w = wire : !firrtl.uint<2>
      %1 = ref.send %w : !firrtl.uint<2>
      ref.define %x, %1 : !firrtl.probe<uint<2>>
      strictconnect %w, %zero : !firrtl.uint<2>
    }
  }
  module @Issue3715(in %p: !firrtl.uint<1>) {
    %test_in, %test_x = instance test @Test(in p: !firrtl.uint<1>, out x: !firrtl.probe<uint<2>>)
    strictconnect %test_in, %p : !firrtl.uint<1>
    %x = ref.resolve %test_x : !firrtl.probe<uint<2>>
  }
}

// -----

// Support using output port reference locally.
// https://github.com/llvm/circt/issues/3713

firrtl.circuit "UseRefsWithSinkFlow" {
  module private @InChild(in %p: !firrtl.probe<uint<1>>) {
  }
  module private @OutChild(in %x: !firrtl.uint, out %y: !firrtl.uint, out %p: !firrtl.probe<uint>) {
    %0 = ref.send %x : !firrtl.uint
    ref.define %p, %0 : !firrtl.probe<uint>
    %1 = ref.resolve %p : !firrtl.probe<uint>
    connect %y, %1 : !firrtl.uint, !firrtl.uint
  }
  module @UseRefsWithSinkFlow(in %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>, out %z: !firrtl.uint<1>, out %zz: !firrtl.uint<1>, out %p: !firrtl.probe<uint<1>>) {
    %0 = ref.send %x : !firrtl.uint<1>
    ref.define %p, %0 : !firrtl.probe<uint<1>>
    %1 = ref.resolve %p : !firrtl.probe<uint<1>>
    strictconnect %y, %1 : !firrtl.uint<1>
    %ic_p = instance ic interesting_name @InChild(in p: !firrtl.probe<uint<1>>)
    %2 = ref.send %x : !firrtl.uint<1>
    ref.define %ic_p, %2 : !firrtl.probe<uint<1>>
    %3 = ref.resolve %ic_p : !firrtl.probe<uint<1>>
    strictconnect %z, %3 : !firrtl.uint<1>
    %oc_x, %oc_y, %oc_p = instance oc interesting_name @OutChild(in x: !firrtl.uint, out y: !firrtl.uint, out p: !firrtl.probe<uint>)
    connect %oc_x, %x : !firrtl.uint, !firrtl.uint<1>
    connect %zz, %oc_y : !firrtl.uint<1>, !firrtl.uint
  }
}

// -----

firrtl.circuit "ProbeAndRWProbe" {
  // Dead, just check it parses.
  module private @Probes(in %ro : !firrtl.probe<uint<1>>, in %rw : !firrtl.rwprobe<uint<2>>) { }
  module @ProbeAndRWProbe() {
  }
}

// -----

firrtl.circuit "Forceable" {
  // Check forceable declarations
  module @Forceable(
    in %clock : !firrtl.clock,
    in %reset : !firrtl.uint<1>,
    in %value : !firrtl.uint<2>,
    out %node_ref : !firrtl.rwprobe<uint<2>>,
    out %wire_ref : !firrtl.rwprobe<uint>,
    out %reg_ref : !firrtl.rwprobe<uint<2>>,
    out %regreset_ref : !firrtl.rwprobe<uint<2>>) {

    %n, %n_f = node %value forceable : !firrtl.uint<2>
    ref.define %node_ref, %n_f : !firrtl.rwprobe<uint<2>>

    // TODO: infer ref result existence + type based on "forceable" or other ref-kind(s) indicator.
    %w, %w_f = wire forceable : !firrtl.uint, !firrtl.rwprobe<uint>
    ref.define %wire_ref, %w_f : !firrtl.rwprobe<uint>
    connect %w, %value : !firrtl.uint, !firrtl.uint<2>

    %reg, %reg_f = reg %clock forceable : !firrtl.clock, !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    ref.define %reg_ref, %reg_f : !firrtl.rwprobe<uint<2>>

    %regreset, %regreset_f = regreset %clock, %reset, %value forceable : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    ref.define %regreset_ref, %regreset_f : !firrtl.rwprobe<uint<2>>
  }
}

