// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-grand-central))' -split-input-file -verify-diagnostics %s

// expected-error @+1 {{more than one 'ExtractGrandCentralAnnotation' was found, but exactly one must be provided}}
firrtl.circuit "MoreThanOneExtractGrandCentralAnnotation" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {name = "foo",
        tpe = "sifive.enterprise.grandcentral.AugmentedGroundType"}]},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}] } {
  module @MoreThanOneExtractGrandCentralAnnotation() {}
}

// -----

firrtl.circuit "NonGroundType" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        id = 1 : i64,
        name = "foo"}],
     id = 0 : i64},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]} {
  module private @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
       defName = "Foo",
       id = 0 : i64,
       name = "View"}]} {
    %_vector = verbatim.expr "???" : () -> !firrtl.vector<uint<2>, 1>
    %ref_vector = ref.send %_vector : !firrtl.vector<uint<2>, 1>
    %vector = ref.resolve %ref_vector : !firrtl.probe<vector<uint<2>, 1>>
    // expected-error @+1 {{'firrtl.node' op cannot be added to interface with id '0' because it is not a ground type}}
    %a = node %vector {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 1 : i64
        }
      ]
    } : !firrtl.vector<uint<2>, 1>
  }
  module private @DUT() {
    instance View_companion @View_companion()
  }
  module @NonGroundType() {
    instance dut @DUT()
  }
}

// -----

// expected-error @+1 {{missing 'id' in root-level BundleType}}
firrtl.circuit "NonGroundType" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]} {
  module @NonGroundType() {}
}

// -----

firrtl.circuit "Foo" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "View",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        id = 1 : i64}],
     id = 0 : i64},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]}  {
  module private @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
       defName = "Foo",
       id = 0 : i64,
       name = "View"}]} {}
  module private @Bar(in %a: !firrtl.uint<1>) {}
  module private @DUT(in %a: !firrtl.uint<1>) {
    // expected-error @+1 {{'firrtl.instance' op is marked as an interface element, but this should be impossible due to how the Chisel Grand Central API works}}
    %bar_a = instance bar @Bar(in a: !firrtl.uint<1> [
        {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
         id = 1 : i64}])
    connect %bar_a, %a : !firrtl.uint<1>, !firrtl.uint<1>
    instance View_companion @View_companion()
  }
  module @Foo() {
    %dut_a = instance dut @DUT(in a: !firrtl.uint<1>)
  }
}

// -----

firrtl.circuit "Foo" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "View",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        id = 1 : i64,
        name = "foo"}],
     id = 0 : i64},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]}  {
  module private @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
       defName = "Foo",
       id = 0 : i64,
       name = "View"}]} {}
  module private @DUT(in %a: !firrtl.uint<1>) {
    // expected-error @+1 {{'firrtl.mem' op is marked as an interface element, but this does not make sense (is there a scattering bug or do you have a malformed hand-crafted MLIR circuit?)}}
    %memory_b_r = mem Undefined {
      annotations = [
        {a},
        {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
         id = 1 : i64}],
      depth = 16 : i64,
      name = "memory_b",
      portNames = ["r"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
    instance View_companion @View_companion()
  }
  module @Foo() {
    %dut_a = instance dut @DUT(in a: !firrtl.uint<1>)
  }
}

// -----

// expected-error @+1 {{'firrtl.circuit' op has an AugmentedGroundType with 'id == 42' that does not have a scattered leaf to connect to in the circuit}}
firrtl.circuit "Foo" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Bar",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        id = 42 : i64,
        name = "baz"}],
     id = 0 : i64},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]}  {
  module private @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
       defName = "Foo",
       id = 0 : i64,
       name = "View"}]} {}
  module private @DUT() {
    instance View_companion @View_companion()
  }
  module @Foo() {
    instance dut @DUT()
  }
}

// -----

firrtl.circuit "FieldNotInCompanion" attributes {
  annotations = [
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "Foo",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          description = "description of foo",
          name = "foo",
          id = 1 : i64
        }
      ],
      id = 0 : i64,
      name = "Foo"
    }
  ]
} {
  // expected-error @+1 {{Grand Central View "Foo" is invalid because a leaf is not inside the companion module}}
  module @Companion() attributes {
    annotations = [
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "Foo",
        id = 0 : i64,
        name = "Foo"
      }
    ]
  } {}
  // expected-note @+1 {{the leaf value is inside this module}}
  module @FieldNotInCompanion() {

    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %c-1_si2 = constant -1 : !firrtl.sint<2>

    // expected-note @+1 {{the leaf value is declared here}}
    %node_c0_ui1 = node %c0_ui1 {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 1 : i64
        }
      ]
    } : !firrtl.uint<1>

    instance companion @Companion()
  }
}

// -----

firrtl.circuit "InvalidField" attributes {
  annotations = [
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "Foo",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          description = "description of foo",
          name = "foo",
          id = 1 : i64
        }
      ],
      id = 0 : i64,
      name = "Foo"
    }
  ]
} {
  module @Companion() attributes {
    annotations = [
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "Foo",
        id = 0 : i64,
        name = "Foo"
      }
    ]
  } {
    // expected-error @+1 {{Grand Central View "Foo" has an invalid leaf value}}
    %node = wire {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 1 : i64
        }
      ]
    } : !firrtl.uint<1>
  }
  module @InvalidField() {
    instance companion @Companion()
  }
}

// -----

firrtl.circuit "MultiplyInstantiated" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Bar",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        id = 42 : i64,
        name = "baz"}],
     id = 0 : i64},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]}  {
  // expected-error @below {{'firrtl.module' op is marked as a GrandCentral 'companion', but it is instantiated more than once}}
  module private @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
       defName = "Companion",
       id = 0 : i64,
       name = "View"}]} {
    %0 = constant 0 :!firrtl.uint<1>
    %zero = node  %0  {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 42 : i64
        }
      ]
    } : !firrtl.uint<1>
  }
  module private @DUT() {
    // expected-note @below {{it is instantiated here}}
    instance View_companion @View_companion()
    // expected-note @below {{it is instantiated here}}
    instance View_companion @View_companion()
  }
  module @MultiplyInstantiated() {
    instance dut @DUT()
  }
}

// -----

firrtl.circuit "NotInstantiated" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Bar",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        id = 42 : i64,
        name = "baz"}],
     id = 0 : i64},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]}  {
  // expected-error @below {{'firrtl.module' op is marked as a GrandCentral 'companion', but is never instantiated}}
  module private @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
       defName = "Companion",
       id = 0 : i64,
       name = "View"}]} {
    %0 = constant 0 :!firrtl.uint<1>
    %zero = node  %0  {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 42 : i64
        }
      ]
    } : !firrtl.uint<1>
  }
  module private @DUT() {
  }
  module @NotInstantiated() {
    instance dut @DUT()
  }
}
