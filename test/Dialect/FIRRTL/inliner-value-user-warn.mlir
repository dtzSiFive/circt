// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-inliner{error-on-unresolved-value-user=false}))' -allow-unregistered-dialect -verify-diagnostics --split-input-file %s | FileCheck %s

// With error-on-unresolved-value-user=false, an unresolvable value user of a
// forked hierpath is a warning and the pass still succeeds, leaving the
// reference naming one arbitrary copy (opt-in, for incremental rollout). The
// default (error) path is covered in inliner-errors.mlir.
// CHECK-LABEL: firrtl.circuit "Top"
firrtl.circuit "Top" {
  hw.hierpath private @p [@Mid::@w]
  firrtl.module private @Sib() {
    // expected-warning @below {{value user of hierpath @p cannot be resolved after inlining}}
    sv.verbatim "mref {{0}}" {symbols = [@p]}
  }
  firrtl.module private @Mid() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %w = firrtl.wire sym @w : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Top()
  firrtl.module @Top() {
    firrtl.instance m1 @Mid()
    firrtl.instance m2 @Mid()
    firrtl.instance sib @Sib()
  }
}

// -----

// A value user of an erased (dead-rooted) hierpath in warn mode.
// The pass warns and completes, keeping the reference; it now dangles.
// That is invalid IR: the warning is the pass's, the error the verifier's.
firrtl.circuit "DeadRootedWarn" {
  hw.hierpath private @p [@Unreachable::@w]
  firrtl.module private @Unreachable() {
    %w = firrtl.wire sym @w : !firrtl.uint<1>
  }
  firrtl.module @DeadRootedWarn() {
    // expected-warning @below {{value user of hierpath @p cannot be resolved after inlining: the hierpath has no surviving target (dead-rooted) and was erased, so this reference would dangle}}
    // expected-error @below {{Referenced path doesn't exist @p}}
    %x = sv.xmr.ref @p : !hw.inout<i1>
  }
}
