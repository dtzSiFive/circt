// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-inliner))' -allow-unregistered-dialect %s | FileCheck %s
// XFAIL: *

// When a private inline module that is the root of a non-local annotation is
// instantiated multiple times, the inliner correctly emits one HierPathOp
// per instantiation context.  However, the leaf annotation on the
// FExtModuleOp is NOT duplicated to reference each new HierPathOp -- only
// the original sym is referenced.  This means downstream passes (e.g.,
// SymbolDCE) drop the additional HierPathOps as unused, and the
// corresponding instance loses its inner sym, breaking the non-local
// reference.
//
// The post-inliner annotation duplication loop in run() walks FModuleOp
// ops only; it does not handle annotations attached to FExtModuleOp
// definitions.

// CHECK-LABEL: firrtl.circuit "InlineRetopExtModuleAnnoDup"
firrtl.circuit "InlineRetopExtModuleAnnoDup" {
  // CHECK-DAG: hw.hierpath private @nla [@InlineRetopExtModuleAnnoDup::@{{[_a-zA-Z0-9]+}}, @A]
  // CHECK-DAG: hw.hierpath private @nla_0 [@InlineRetopExtModuleAnnoDup::@{{[_a-zA-Z0-9]+}}, @A]
  hw.hierpath private @nla [@X::@sym, @A]
  // The annotation on @A should be duplicated, one with @nla and one with
  // @nla_0, so each context's HierPathOp has a consumer.
  // CHECK-DAG: firrtl.extmodule private @A() attributes {annotations = [{{.*}}circt.nonlocal = @nla, class = "test"{{.*}}{{.*}}circt.nonlocal = @nla_0, class = "test"{{.*}}]}
  firrtl.extmodule private @A() attributes {
    annotations = [{circt.nonlocal = @nla, class = "test"}]
  }
  firrtl.module private @X() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance a sym @sym @A()
  }
  firrtl.module @InlineRetopExtModuleAnnoDup() {
    firrtl.instance x1 @X()
    firrtl.instance x2 @X()
  }
}
