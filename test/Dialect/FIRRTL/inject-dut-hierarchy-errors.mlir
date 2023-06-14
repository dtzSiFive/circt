// RUN: circt-opt -verify-diagnostics -pass-pipeline='builtin.module(firrtl.circuit(firrtl-inject-dut-hier))' --split-input-file %s

// expected-error @+1 {{contained multiple 'sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation' annotations when at most one is allowed}}
firrtl.circuit "MultipleHierarchyAnnotations" attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "Foo"},
      {class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "Bar"}
    ]
  } {
  module private @DUT() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {}
  module @MultipleHierarchyAnnotations() {
    instance dut @DUT()
  }
}

// -----

// expected-error @+1 {{contained a 'sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation', but no 'sifive.enterprise.firrtl.MarkDUTAnnotation' was provided}}
firrtl.circuit "HierarchyAnnotationWithoutMarkDUT" attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "Foo"}
    ]
  } {
  module private @DUT() {}
  module @HierarchyAnnotationWithoutMarkDUT() {
    instance dut @DUT()
  }
}
