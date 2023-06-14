// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-resets))' --verify-diagnostics --split-input-file %s | FileCheck %s

// Tests extracted from:
// - github.com/chipsalliance/firrtl:
//   - test/scala/firrtlTests/InferResetsSpec.scala
// - github.com/sifive/$internal:
//   - test/scala/firrtl/FullAsyncResetTransform.scala

firrtl.circuit "Foo" {
firrtl.module @Foo() {}


//===----------------------------------------------------------------------===//
// Reset Inference
//===----------------------------------------------------------------------===//

// Provoke two existing reset networks being merged.
// CHECK-LABEL: module @MergeNetsChild1
// CHECK-SAME: in %reset: !firrtl.asyncreset
firrtl.module @MergeNetsChild1(in %reset: !firrtl.reset) {
  // CHECK: %localReset = wire : !firrtl.asyncreset
  %localReset = wire : !firrtl.reset
  strictconnect %localReset, %reset : !firrtl.reset
}
// CHECK-LABEL: module @MergeNetsChild2
// CHECK-SAME: in %reset: !firrtl.asyncreset
firrtl.module @MergeNetsChild2(in %reset: !firrtl.reset) {
  // CHECK: %localReset = wire : !firrtl.asyncreset
  %localReset = wire : !firrtl.reset
  strictconnect %localReset, %reset : !firrtl.reset
}
// CHECK-LABEL: module @MergeNetsTop
firrtl.module @MergeNetsTop(in %reset: !firrtl.asyncreset) {
  // CHECK: %localReset = wire : !firrtl.asyncreset
  %localReset = wire : !firrtl.reset
  %t = resetCast %reset : (!firrtl.asyncreset) -> !firrtl.reset
  strictconnect %localReset, %t : !firrtl.reset
  // CHECK: %c1_reset = instance c1 @MergeNetsChild1(in reset: !firrtl.asyncreset)
  // CHECK: %c2_reset = instance c2 @MergeNetsChild2(in reset: !firrtl.asyncreset)
  %c1_reset = instance c1 @MergeNetsChild1(in reset: !firrtl.reset)
  %c2_reset = instance c2 @MergeNetsChild2(in reset: !firrtl.reset)
  strictconnect %c1_reset, %localReset : !firrtl.reset
  strictconnect %c2_reset, %localReset : !firrtl.reset
}

// Should support casting to other types
// CHECK-LABEL: module @CastingToOtherTypes
firrtl.module @CastingToOtherTypes(in %a: !firrtl.uint<1>, out %v: !firrtl.uint<1>, out %w: !firrtl.sint<1>, out %x: !firrtl.clock, out %y: !firrtl.asyncreset) {
  // CHECK: %r = wire : !firrtl.uint<1>
  %r = wire : !firrtl.reset
  %0 = asUInt %r : (!firrtl.reset) -> !firrtl.uint<1>
  %1 = asSInt %r : (!firrtl.reset) -> !firrtl.sint<1>
  %2 = asClock %r : (!firrtl.reset) -> !firrtl.clock
  %3 = asAsyncReset %r : (!firrtl.reset) -> !firrtl.asyncreset
  %4 = resetCast %a : (!firrtl.uint<1>) -> !firrtl.reset
  strictconnect %r, %4 : !firrtl.reset
  strictconnect %v, %0 : !firrtl.uint<1>
  strictconnect %w, %1 : !firrtl.sint<1>
  strictconnect %x, %2 : !firrtl.clock
  strictconnect %y, %3 : !firrtl.asyncreset
}

// Should support const-casts
// CHECK-LABEL: module @ConstCast
firrtl.module @ConstCast(in %a: !firrtl.const.uint<1>) {
  // CHECK: %r = wire : !firrtl.uint<1>
  %r = wire : !firrtl.reset
  %0 = resetCast %a : (!firrtl.const.uint<1>) -> !firrtl.const.reset
  %1 = constCast %0 : (!firrtl.const.reset) -> !firrtl.reset
  strictconnect %r, %1 : !firrtl.reset
}

// Should work across Module boundaries
// CHECK-LABEL: module @ModuleBoundariesChild
// CHECK-SAME: in %childReset: !firrtl.uint<1>
firrtl.module @ModuleBoundariesChild(in %clock: !firrtl.clock, in %childReset: !firrtl.reset, in %x: !firrtl.uint<8>, out %z: !firrtl.uint<8>) {
  %c123_ui = constant 123 : !firrtl.uint
  // CHECK: %r = regreset %clock, %childReset, %c123_ui : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint, !firrtl.uint<8>
  %r = regreset %clock, %childReset, %c123_ui : !firrtl.clock, !firrtl.reset, !firrtl.uint, !firrtl.uint<8>
  strictconnect %r, %x : !firrtl.uint<8>
  strictconnect %z, %r : !firrtl.uint<8>
}
// CHECK-LABEL: module @ModuleBoundariesTop
firrtl.module @ModuleBoundariesTop(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %x: !firrtl.uint<8>, out %z: !firrtl.uint<8>) {
  // CHECK: {{.*}} = instance c @ModuleBoundariesChild(in clock: !firrtl.clock, in childReset: !firrtl.uint<1>, in x: !firrtl.uint<8>, out z: !firrtl.uint<8>)
  %c_clock, %c_childReset, %c_x, %c_z = instance c @ModuleBoundariesChild(in clock: !firrtl.clock, in childReset: !firrtl.reset, in x: !firrtl.uint<8>, out z: !firrtl.uint<8>)
  strictconnect %c_clock, %clock : !firrtl.clock
  connect %c_childReset, %reset : !firrtl.reset, !firrtl.uint<1>
  strictconnect %c_x, %x : !firrtl.uint<8>
  strictconnect %z, %c_z : !firrtl.uint<8>
}

// Should work across multiple Module boundaries
// CHECK-LABEL: module @MultipleModuleBoundariesChild
// CHECK-SAME: in %resetIn: !firrtl.uint<1>
// CHECK-SAME: out %resetOut: !firrtl.uint<1>
firrtl.module @MultipleModuleBoundariesChild(in %resetIn: !firrtl.reset, out %resetOut: !firrtl.reset) {
  strictconnect %resetOut, %resetIn : !firrtl.reset
}
// CHECK-LABEL: module @MultipleModuleBoundariesTop
firrtl.module @MultipleModuleBoundariesTop(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %x: !firrtl.uint<8>, out %z: !firrtl.uint<8>) {
  // CHECK: {{.*}} = instance c @MultipleModuleBoundariesChild(in resetIn: !firrtl.uint<1>, out resetOut: !firrtl.uint<1>)
  %c_resetIn, %c_resetOut = instance c @MultipleModuleBoundariesChild(in resetIn: !firrtl.reset, out resetOut: !firrtl.reset)
  connect %c_resetIn, %reset : !firrtl.reset, !firrtl.uint<1>
  %c123_ui = constant 123 : !firrtl.uint
  // CHECK: %r = regreset %clock, %c_resetOut, %c123_ui : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint, !firrtl.uint<8>
  %r = regreset %clock, %c_resetOut, %c123_ui : !firrtl.clock, !firrtl.reset, !firrtl.uint, !firrtl.uint<8>
  strictconnect %r, %x : !firrtl.uint<8>
  strictconnect %z, %r : !firrtl.uint<8>
}

// Should work in nested and flipped aggregates with connect
// CHECK-LABEL: module @NestedAggregates
// CHECK-SAME: out %buzz: !firrtl.bundle<foo flip: vector<bundle<a: asyncreset, b flip: asyncreset, c: uint<1>>, 2>, bar: vector<bundle<a: asyncreset, b flip: asyncreset, c: uint<8>>, 2>>
firrtl.module @NestedAggregates(out %buzz: !firrtl.bundle<foo flip: vector<bundle<a: asyncreset, b flip: reset, c: uint<1>>, 2>, bar: vector<bundle<a: reset, b flip: asyncreset, c: uint<8>>, 2>>) {
  %0 = subfield %buzz[bar] : !firrtl.bundle<foo flip: vector<bundle<a: asyncreset, b flip: reset, c: uint<1>>, 2>, bar: vector<bundle<a: reset, b flip: asyncreset, c: uint<8>>, 2>>
  %1 = subfield %buzz[foo] : !firrtl.bundle<foo flip: vector<bundle<a: asyncreset, b flip: reset, c: uint<1>>, 2>, bar: vector<bundle<a: reset, b flip: asyncreset, c: uint<8>>, 2>>
  connect %0, %1 :  !firrtl.vector<bundle<a: reset, b flip: asyncreset, c: uint<8>>, 2>, !firrtl.vector<bundle<a: asyncreset, b flip: reset, c: uint<1>>, 2>
}

// Should work with deeply nested aggregates.
// CHECK-LABEL: module @DeeplyNestedAggregates(in %reset: !firrtl.uint<1>, out %buzz: !firrtl.bundle<a: bundle<b: uint<1>>>) {
firrtl.module @DeeplyNestedAggregates(in %reset: !firrtl.uint<1>, out %buzz: !firrtl.bundle<a: bundle<b: reset>>) {
  %0 = subfield %buzz[a] : !firrtl.bundle<a: bundle<b : reset>>
  %1 = subfield %0[b] : !firrtl.bundle<b: reset>
  // CHECK: connect %1, %reset : !firrtl.uint<1>
  connect %1, %reset : !firrtl.reset, !firrtl.uint<1>
}


// Should not crash if a ResetType has no drivers
// CHECK-LABEL: module @DontCrashIfNoDrivers
// CHECK-SAME: out %out: !firrtl.uint<1>
firrtl.module @DontCrashIfNoDrivers(out %out: !firrtl.reset) {
  %c1_ui = constant 1 : !firrtl.uint
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  // CHECK: %w = wire : !firrtl.uint<1>
  %w = wire : !firrtl.reset
  strictconnect %out, %w : !firrtl.reset
  // TODO: Enable the following once #1303 is fixed.
  // connect %out, %c1_ui : !firrtl.reset, !firrtl.uint
  connect %out, %c1_ui1 : !firrtl.reset, !firrtl.uint<1>
}

// Should allow concrete reset types to overrule invalidation
// CHECK-LABEL: module @ConcreteResetOverruleInvalid
// CHECK-SAME: out %out: !firrtl.asyncreset
firrtl.module @ConcreteResetOverruleInvalid(in %in: !firrtl.asyncreset, out %out: !firrtl.reset) {
  // CHECK: %invalid_asyncreset = invalidvalue : !firrtl.asyncreset
  %invalid_reset = invalidvalue : !firrtl.reset
  strictconnect %out, %invalid_reset : !firrtl.reset
  connect %out, %in : !firrtl.reset, !firrtl.asyncreset
}

// Should default to BoolType for Resets that are only invalidated
// CHECK-LABEL: module @DefaultToBool
// CHECK-SAME: out %out: !firrtl.uint<1>
firrtl.module @DefaultToBool(out %out: !firrtl.reset) {
  // CHECK: %invalid_ui1 = invalidvalue : !firrtl.uint<1>
  %invalid_reset = invalidvalue : !firrtl.reset
  strictconnect %out, %invalid_reset : !firrtl.reset
}

// Should not error if component of ResetType is invalidated and connected to an AsyncResetType
// CHECK-LABEL: module @OverrideInvalidWithDifferentResetType
// CHECK-SAME: out %out: !firrtl.asyncreset
firrtl.module @OverrideInvalidWithDifferentResetType(in %cond: !firrtl.uint<1>, in %in: !firrtl.asyncreset, out %out: !firrtl.reset) {
  // CHECK: %invalid_asyncreset = invalidvalue : !firrtl.asyncreset
  %invalid_reset = invalidvalue : !firrtl.reset
  strictconnect %out, %invalid_reset : !firrtl.reset
  when %cond : !firrtl.uint<1>  {
    connect %out, %in : !firrtl.reset, !firrtl.asyncreset
  }
}

// Should allow ResetType to drive AsyncResets or UInt<1>
// CHECK-LABEL: module @ResetDrivesAsyncResetOrBool1
firrtl.module @ResetDrivesAsyncResetOrBool1(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  // CHECK: %w = wire : !firrtl.uint<1>
  %w = wire : !firrtl.reset
  connect %w, %in : !firrtl.reset, !firrtl.uint<1>
  connect %out, %w : !firrtl.uint<1>, !firrtl.reset
}
// CHECK-LABEL: module @ResetDrivesAsyncResetOrBool2
firrtl.module @ResetDrivesAsyncResetOrBool2(out %foo: !firrtl.bundle<a flip: uint<1>>, in %bar: !firrtl.bundle<a flip: uint<1>>) {
  // CHECK: %w = wire : !firrtl.bundle<a flip: uint<1>>
  %w = wire : !firrtl.bundle<a flip: reset>
  connect %foo, %w : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: reset>
  connect %w, %bar : !firrtl.bundle<a flip: reset>, !firrtl.bundle<a flip: uint<1>>
}
// CHECK-LABEL: module @ResetDrivesAsyncResetOrBool3
firrtl.module @ResetDrivesAsyncResetOrBool3(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  // CHECK: %w = wire : !firrtl.uint<1>
  %w = wire : !firrtl.reset
  connect %w, %in : !firrtl.reset, !firrtl.uint<1>
  connect %out, %w : !firrtl.uint<1>, !firrtl.reset
}

// Should support inferring modules that would dedup differently
// CHECK-LABEL: module @DedupDifferentlyChild1
// CHECK-SAME: in %childReset: !firrtl.uint<1>
firrtl.module @DedupDifferentlyChild1(in %clock: !firrtl.clock, in %childReset: !firrtl.reset, in %x: !firrtl.uint<8>, out %z: !firrtl.uint<8>) {
  %c123_ui = constant 123 : !firrtl.uint
  // CHECK: %r = regreset %clock, %childReset, %c123_ui : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint, !firrtl.uint<8>
  %r = regreset %clock, %childReset, %c123_ui : !firrtl.clock, !firrtl.reset, !firrtl.uint, !firrtl.uint<8>
  strictconnect %r, %x : !firrtl.uint<8>
  strictconnect %z, %r : !firrtl.uint<8>
}
// CHECK-LABEL: module @DedupDifferentlyChild2
// CHECK-SAME: in %childReset: !firrtl.asyncreset
firrtl.module @DedupDifferentlyChild2(in %clock: !firrtl.clock, in %childReset: !firrtl.reset, in %x: !firrtl.uint<8>, out %z: !firrtl.uint<8>) {
  %c123_ui = constant 123 : !firrtl.uint
  // CHECK: %r = regreset %clock, %childReset, %c123_ui : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint, !firrtl.uint<8>
  %r = regreset %clock, %childReset, %c123_ui : !firrtl.clock, !firrtl.reset, !firrtl.uint, !firrtl.uint<8>
  strictconnect %r, %x : !firrtl.uint<8>
  strictconnect %z, %r : !firrtl.uint<8>
}
// CHECK-LABEL: module @DedupDifferentlyTop
firrtl.module @DedupDifferentlyTop(in %clock: !firrtl.clock, in %reset1: !firrtl.uint<1>, in %reset2: !firrtl.asyncreset, in %x: !firrtl.vector<uint<8>, 2>, out %z: !firrtl.vector<uint<8>, 2>) {
  // CHECK: {{.*}} = instance c1 @DedupDifferentlyChild1(in clock: !firrtl.clock, in childReset: !firrtl.uint<1>
  %c1_clock, %c1_childReset, %c1_x, %c1_z = instance c1 @DedupDifferentlyChild1(in clock: !firrtl.clock, in childReset: !firrtl.reset, in x: !firrtl.uint<8>, out z: !firrtl.uint<8>)
  strictconnect %c1_clock, %clock : !firrtl.clock
  connect %c1_childReset, %reset1 : !firrtl.reset, !firrtl.uint<1>
  %0 = subindex %x[0] : !firrtl.vector<uint<8>, 2>
  strictconnect %c1_x, %0 : !firrtl.uint<8>
  %1 = subindex %z[0] : !firrtl.vector<uint<8>, 2>
  strictconnect %1, %c1_z : !firrtl.uint<8>
  // CHECK: {{.*}} = instance c2 @DedupDifferentlyChild2(in clock: !firrtl.clock, in childReset: !firrtl.asyncreset
  %c2_clock, %c2_childReset, %c2_x, %c2_z = instance c2 @DedupDifferentlyChild2(in clock: !firrtl.clock, in childReset: !firrtl.reset, in x: !firrtl.uint<8>, out z: !firrtl.uint<8>)
  strictconnect %c2_clock, %clock : !firrtl.clock
  connect %c2_childReset, %reset2 : !firrtl.reset, !firrtl.asyncreset
  %2 = subindex %x[1] : !firrtl.vector<uint<8>, 2>
  strictconnect %c2_x, %2 : !firrtl.uint<8>
  %3 = subindex %z[1] : !firrtl.vector<uint<8>, 2>
  strictconnect %3, %c2_z : !firrtl.uint<8>
}

// Should infer based on what a component *drives* not just what drives it
// CHECK-LABEL: module @InferBasedOnDriven
// CHECK-SAME: out %out: !firrtl.asyncreset
firrtl.module @InferBasedOnDriven(in %in: !firrtl.asyncreset, out %out: !firrtl.reset) {
  // CHECK: %w = wire : !firrtl.asyncreset
  // CHECK: %invalid_asyncreset = invalidvalue : !firrtl.asyncreset
  %w = wire : !firrtl.reset
  %invalid_reset = invalidvalue : !firrtl.reset
  strictconnect %w, %invalid_reset : !firrtl.reset
  strictconnect %out, %w : !firrtl.reset
  connect %out, %in : !firrtl.reset, !firrtl.asyncreset
}

// Should infer from connections, ignoring the fact that the invalidation wins
// CHECK-LABEL: module @InferIgnoreInvalidation
// CHECK-SAME: out %out: !firrtl.asyncreset
firrtl.module @InferIgnoreInvalidation(in %in: !firrtl.asyncreset, out %out: !firrtl.reset) {
  // CHECK: %invalid_asyncreset = invalidvalue : !firrtl.asyncreset
  %invalid_reset = invalidvalue : !firrtl.reset
  connect %out, %in : !firrtl.reset, !firrtl.asyncreset
  strictconnect %out, %invalid_reset : !firrtl.reset
}

// Should not propagate type info from downstream across a cast
// CHECK-LABEL: module @DontPropagateUpstreamAcrossCast
// CHECK-SAME: out %out0: !firrtl.asyncreset
// CHECK-SAME: out %out1: !firrtl.uint<1>
firrtl.module @DontPropagateUpstreamAcrossCast(in %in0: !firrtl.asyncreset, in %in1: !firrtl.uint<1>, out %out0: !firrtl.reset, out %out1: !firrtl.reset) {
  // CHECK: %w = wire : !firrtl.uint<1>
  %w = wire : !firrtl.reset
  %invalid_reset = invalidvalue : !firrtl.reset
  strictconnect %w, %invalid_reset : !firrtl.reset
  %0 = asAsyncReset %w : (!firrtl.reset) -> !firrtl.asyncreset
  connect %out0, %0 : !firrtl.reset, !firrtl.asyncreset
  strictconnect %out1, %w : !firrtl.reset
  connect %out0, %in0 : !firrtl.reset, !firrtl.asyncreset
  connect %out1, %in1 : !firrtl.reset, !firrtl.uint<1>
}

// Should take into account both internal and external constraints on Module port types
// CHECK-LABEL: module @InternalAndExternalChild
// CHECK-SAME: out %o: !firrtl.asyncreset
firrtl.module @InternalAndExternalChild(in %i: !firrtl.asyncreset, out %o: !firrtl.reset) {
  connect %o, %i : !firrtl.reset, !firrtl.asyncreset
}
// CHECK-LABEL: module @InternalAndExternalTop
firrtl.module @InternalAndExternalTop(in %in: !firrtl.asyncreset, out %out: !firrtl.asyncreset) {
  // CHECK: {{.*}} = instance c @InternalAndExternalChild(in i: !firrtl.asyncreset, out o: !firrtl.asyncreset)
  %c_i, %c_o = instance c @InternalAndExternalChild(in i: !firrtl.asyncreset, out o: !firrtl.reset)
  strictconnect %c_i, %in : !firrtl.asyncreset
  connect %out, %c_o : !firrtl.asyncreset, !firrtl.reset
}

// Should not crash on combinational loops
// CHECK-LABEL: module @NoCrashOnCombLoop
// CHECK-SAME: out %out: !firrtl.asyncreset
firrtl.module @NoCrashOnCombLoop(in %in: !firrtl.asyncreset, out %out: !firrtl.reset) {
  %w0 = wire : !firrtl.reset
  %w1 = wire : !firrtl.reset
  connect %w0, %in : !firrtl.reset, !firrtl.asyncreset
  strictconnect %w0, %w1 : !firrtl.reset
  strictconnect %w1, %w0 : !firrtl.reset
  connect %out, %in : !firrtl.reset, !firrtl.asyncreset
}

// Should not treat a single `invalidvalue` connected to different resets as a
// connection of the resets themselves.
// CHECK-LABEL: module @InvalidValueShouldNotConnect
// CHECK-SAME: out %r0: !firrtl.asyncreset
// CHECK-SAME: out %r1: !firrtl.asyncreset
// CHECK-SAME: out %r2: !firrtl.uint<1>
// CHECK-SAME: out %r3: !firrtl.uint<1>
firrtl.module @InvalidValueShouldNotConnect(
  in %ar: !firrtl.asyncreset,
  in %sr: !firrtl.uint<1>,
  out %r0: !firrtl.reset,
  out %r1: !firrtl.reset,
  out %r2: !firrtl.reset,
  out %r3: !firrtl.reset
) {
  %invalid_reset = invalidvalue : !firrtl.reset
  strictconnect %r0, %invalid_reset : !firrtl.reset
  strictconnect %r1, %invalid_reset : !firrtl.reset
  strictconnect %r2, %invalid_reset : !firrtl.reset
  strictconnect %r3, %invalid_reset : !firrtl.reset
  connect %r0, %ar : !firrtl.reset, !firrtl.asyncreset
  connect %r1, %ar : !firrtl.reset, !firrtl.asyncreset
  connect %r2, %sr : !firrtl.reset, !firrtl.uint<1>
  connect %r3, %sr : !firrtl.reset, !firrtl.uint<1>
}

// Should properly adjust the type of external modules.
// CHECK-LABEL: extmodule @ShouldAdjustExtModule1
// CHECK-SAME: in reset: !firrtl.uint<1>
firrtl.extmodule @ShouldAdjustExtModule1(in reset: !firrtl.reset)
// CHECK-LABEL: module @ShouldAdjustExtModule2
// CHECK: %x_reset = instance x @ShouldAdjustExtModule1(in reset: !firrtl.uint<1>)
firrtl.module @ShouldAdjustExtModule2() {
  %x_reset = instance x @ShouldAdjustExtModule1(in reset: !firrtl.reset)
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  connect %x_reset, %c1_ui1 : !firrtl.reset, !firrtl.uint<1>
}

// Should not crash if there are connects with foreign types.
// CHECK-LABEL: module @ForeignTypes
firrtl.module @ForeignTypes(out %out: !firrtl.reset) {
  %0 = wire : index
  %1 = wire : index
  connect %0, %1 : index, index
  // CHECK-NEXT: [[W0:%.+]] = wire : index
  // CHECK-NEXT: [[W1:%.+]] = wire : index
  // CHECK-NEXT: connect [[W0]], [[W1]] : index
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  connect %out, %c1_ui1 : !firrtl.reset, !firrtl.uint<1>
}


//===----------------------------------------------------------------------===//
// Full Async Reset
//===----------------------------------------------------------------------===//


// CHECK-LABEL: module @ConsumeIgnoreAnno
// CHECK-NOT: IgnoreFullAsyncResetAnnotation
firrtl.module @ConsumeIgnoreAnno() attributes {annotations = [{class = "sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation"}]} {
}

// CHECK-LABEL: module @ConsumeResetAnnoPort
// CHECK-NOT: FullAsyncResetAnnotation
firrtl.module @ConsumeResetAnnoPort(in %outerReset: !firrtl.asyncreset) attributes {portAnnotations = [[{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
}

// CHECK-LABEL: module @ConsumeResetAnnoWire
firrtl.module @ConsumeResetAnnoWire(in %outerReset: !firrtl.asyncreset) {
  // CHECK: %innerReset = wire
  // CHECK-NOT: FullAsyncResetAnnotation
  %innerReset = wire {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
}

} // circuit

// -----
// Reset-less registers should inherit the annotated async reset signal.
firrtl.circuit "Top" {
  // CHECK-LABEL: module @Top
  module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %init: !firrtl.uint<1>, in %in: !firrtl.uint<8>, in %extraReset: !firrtl.asyncreset ) attributes {
    portAnnotations = [[],[],[],[],[{class = "firrtl.transforms.DontTouchAnnotation"}, {class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
    %c1_ui8 = constant 1 : !firrtl.uint<8>
    // CHECK: %reg1 = regreset sym @reg1 %clock, %extraReset, %c0_ui8
    %reg1 = reg sym @reg1 %clock : !firrtl.clock, !firrtl.uint<8>
    strictconnect %reg1, %in : !firrtl.uint<8>

    // Existing async reset remains untouched.
    // CHECK: %reg2 = regreset %clock, %reset, %c1_ui8
    %reg2 = regreset %clock, %reset, %c1_ui8 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
    strictconnect %reg2, %in : !firrtl.uint<8>

    // Existing sync reset is moved to mux.
    // CHECK: %reg3 = regreset %clock, %extraReset, %c0_ui8
    // CHECK: %0 = mux(%init, %c1_ui8, %reg3)
    // CHECK: %1 = mux(%init, %c1_ui8, %in)
    // CHECK: strictconnect %reg3, %1
    %reg3 = regreset %clock, %init, %c1_ui8 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    strictconnect %reg3, %in : !firrtl.uint<8>

    // Factoring of sync reset into mux works through subfield op.
    // CHECK: %reg4 = regreset %clock, %extraReset, %2
    // CHECK: %4 = mux(%init, %reset4, %reg4)
    // CHECK: %5 = subfield %reset4[a]
    // CHECK: %6 = subfield %reg4[a]
    // CHECK: %7 = mux(%init, %5, %in)
    // CHECK: strictconnect %6, %7
    %reset4 = wire : !firrtl.bundle<a: uint<8>>
    %reg4 = regreset %clock, %init, %reset4 : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>
    %0 = subfield %reg4[a] : !firrtl.bundle<a: uint<8>>
    strictconnect %0, %in : !firrtl.uint<8>

    // Factoring of sync reset into mux works through subindex op.
    // CHECK: %reg5 = regreset %clock, %extraReset, %8
    // CHECK: %10 = mux(%init, %reset5, %reg5)
    // CHECK: strictconnect %reg5, %10
    // CHECK: %11 = subindex %reset5[0]
    // CHECK: %12 = subindex %reg5[0]
    // CHECK: %13 = mux(%init, %11, %in)
    // CHECK: strictconnect %12, %13
    %reset5 = wire : !firrtl.vector<uint<8>, 1>
    %reg5 = regreset %clock, %init, %reset5 : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>
    %1 = subindex %reg5[0] : !firrtl.vector<uint<8>, 1>
    strictconnect %1, %in : !firrtl.uint<8>

    // Factoring of sync reset into mux works through subaccess op.
    // CHECK: %reg6 = regreset %clock, %extraReset, %14 
    // CHECK: %16 = mux(%init, %reset6, %reg6)
    // CHECK: strictconnect %reg6, %16
    // CHECK: %17 = subaccess %reset6[%in]
    // CHECK: %18 = subaccess %reg6[%in]
    // CHECK: %19 = mux(%init, %17, %in)
    // CHECK: strictconnect %18, %19
    %reset6 = wire : !firrtl.vector<uint<8>, 1>
    %reg6 = regreset %clock, %init, %reset6 : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>
    %2 = subaccess %reg6[%in] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<8>
    strictconnect %2, %in : !firrtl.uint<8>

    // Subfields that are never assigned to should not leave unused reset
    // subfields behind.
    // CHECK-NOT: subfield %reset4[a]
    // CHECK: %20 = subfield %reg4[a]
    %3 = subfield %reg4[a] : !firrtl.bundle<a: uint<8>>
  }
}

// -----
// Async reset inference should be able to construct reset values for aggregate
// types.
firrtl.circuit "Top" {
  // CHECK-LABEL: module @Top
  module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) attributes {
    portAnnotations = [[],[{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
    // CHECK: %c0_ui = constant 0 : !firrtl.uint
    // CHECK: %reg_uint = regreset %clock, %reset, %c0_ui
    %reg_uint = reg %clock : !firrtl.clock, !firrtl.uint
    // CHECK: %c0_si = constant 0 : !firrtl.sint
    // CHECK: %reg_sint = regreset %clock, %reset, %c0_si
    %reg_sint = reg %clock : !firrtl.clock, !firrtl.sint
    // CHECK: %0 = wire : !firrtl.bundle<a: uint<8>, b: bundle<x: uint<8>, y: uint<8>>>
    // CHECK: %c0_ui8 = constant 0 : !firrtl.uint<8>
    // CHECK: %1 = subfield %0[a]
    // CHECK: strictconnect %1, %c0_ui8
    // CHECK: %2 = wire : !firrtl.bundle<x: uint<8>, y: uint<8>>
    // CHECK: %3 = subfield %2[x]
    // CHECK: strictconnect %3, %c0_ui8
    // CHECK: %4 = subfield %2[y]
    // CHECK: strictconnect %4, %c0_ui8
    // CHECK: %5 = subfield %0[b]
    // CHECK: strictconnect %5, %2
    // CHECK: %reg_bundle = regreset %clock, %reset, %0
    %reg_bundle = reg %clock : !firrtl.clock, !firrtl.bundle<a: uint<8>, b: bundle<x: uint<8>, y: uint<8>>>
    // CHECK: %6 = wire : !firrtl.vector<uint<8>, 4>
    // CHECK: %c0_ui8_0 = constant 0 : !firrtl.uint<8>
    // CHECK: %7 = subindex %6[0]
    // CHECK: strictconnect %7, %c0_ui8_0
    // CHECK: %8 = subindex %6[1]
    // CHECK: strictconnect %8, %c0_ui8_0
    // CHECK: %9 = subindex %6[2]
    // CHECK: strictconnect %9, %c0_ui8_0
    // CHECK: %10 = subindex %6[3]
    // CHECK: strictconnect %10, %c0_ui8_0
    // CHECK: %reg_vector = regreset %clock, %reset, %6
    %reg_vector = reg %clock : !firrtl.clock, !firrtl.vector<uint<8>, 4>
  }
}

// -----
// Reset should reuse ports if name and type matches.
firrtl.circuit "ReusePorts" {
  // CHECK-LABEL: module @Child
  // CHECK-SAME: in %clock: !firrtl.clock
  // CHECK-SAME: in %reset: !firrtl.asyncreset
  // CHECK: %reg = regreset %clock, %reset, %c0_ui8
  module @Child(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %reg = reg %clock : !firrtl.clock, !firrtl.uint<8>
  }
  // CHECK-LABEL: module @BadName
  // CHECK-SAME: in %reset: !firrtl.asyncreset,
  // CHECK-SAME: in %clock: !firrtl.clock
  // CHECK-SAME: in %existingReset: !firrtl.asyncreset
  // CHECK: %reg = regreset %clock, %reset, %c0_ui8
  module @BadName(in %clock: !firrtl.clock, in %existingReset: !firrtl.asyncreset) {
    %reg = reg %clock : !firrtl.clock, !firrtl.uint<8>
  }
  // CHECK-LABEL: module @BadType
  // CHECK-SAME: in %reset_0: !firrtl.asyncreset,
  // CHECK-SAME: in %clock: !firrtl.clock
  // CHECK-SAME: in %reset: !firrtl.uint<1>
  // CHECK: %reg = regreset %clock, %reset_0, %c0_ui8
  module @BadType(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %reg = reg %clock : !firrtl.clock, !firrtl.uint<8>
  }
  // CHECK-LABEL: module @ReusePorts
  module @ReusePorts(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) attributes {
    portAnnotations = [[],[{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
    // CHECK: %child_clock, %child_reset = instance child
    // CHECK: strictconnect %child_reset, %reset
    // CHECK: %badName_reset, %badName_clock, %badName_existingReset = instance badName
    // CHECK: strictconnect %badName_reset, %reset
    // CHECK: %badType_reset_0, %badType_clock, %badType_reset = instance badType
    // CHECK: strictconnect %badType_reset_0, %reset
    %child_clock, %child_reset = instance child @Child(in clock: !firrtl.clock, in reset: !firrtl.asyncreset)
    %badName_clock, %badName_existingReset = instance badName @BadName(in clock: !firrtl.clock, in existingReset: !firrtl.asyncreset)
    %badType_clock, %badType_reset = instance badType @BadType(in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
  }
}

// -----
// Infer async reset: nested
firrtl.circuit "FullAsyncNested" {
  // CHECK-LABEL: module @FullAsyncNestedDeeper
  module @FullAsyncNestedDeeper(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>) {
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    // CHECK: %io_out_REG = regreset %clock, %reset, %c1_ui1
    %io_out_REG = regreset %clock, %reset, %c1_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<8>
    strictconnect %io_out_REG, %io_in : !firrtl.uint<8>
    strictconnect %io_out, %io_out_REG : !firrtl.uint<8>
  }
  // CHECK-LABEL: module @FullAsyncNestedChild
  module @FullAsyncNestedChild(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>) {
    %inst_clock, %inst_reset, %inst_io_in, %inst_io_out = instance inst @FullAsyncNestedDeeper(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, in io_in: !firrtl.uint<8>, out io_out: !firrtl.uint<8>)
    strictconnect %inst_clock, %clock : !firrtl.clock
    strictconnect %inst_reset, %reset : !firrtl.asyncreset
    strictconnect %inst_io_in, %io_in : !firrtl.uint<8>
    // CHECK: %io_out_REG = regreset %clock, %reset, %c0_ui8
    %io_out_REG = reg %clock : !firrtl.clock, !firrtl.uint<8>
    // CHECK: %io_out_REG_NO = reg %clock : !firrtl.clock, !firrtl.uint<8>
    %io_out_REG_NO = reg %clock {annotations = [{class = "sifive.enterprise.firrtl.ExcludeMemFromMemToRegOfVec"}]}: !firrtl.clock, !firrtl.uint<8>
    strictconnect %io_out_REG, %io_in : !firrtl.uint<8>
    %0 = add %io_out_REG, %inst_io_out : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<9>
    %1 = bits %0 7 to 0 : (!firrtl.uint<9>) -> !firrtl.uint<8>
    strictconnect %io_out, %1 : !firrtl.uint<8>
  }
  // CHECK-LABEL: module @FullAsyncNested
  module @FullAsyncNested(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>) attributes {
    portAnnotations=[[],[{class = "firrtl.transforms.DontTouchAnnotation"}, {class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}], [], []] } {
    %inst_clock, %inst_reset, %inst_io_in, %inst_io_out = instance inst @FullAsyncNestedChild(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, in io_in: !firrtl.uint<8>, out io_out: !firrtl.uint<8>)
    strictconnect %inst_clock, %clock : !firrtl.clock
    strictconnect %inst_reset, %reset : !firrtl.asyncreset
    strictconnect %io_out, %inst_io_out : !firrtl.uint<8>
    strictconnect %inst_io_in, %io_in : !firrtl.uint<8>
  }
}


// -----
// Infer async reset: excluded
// TODO: Check that no extraReset port present
firrtl.circuit "FullAsyncExcluded" {
  // CHECK-LABEL: module @FullAsyncExcludedChild
  // CHECK-SAME: (in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>)
  module @FullAsyncExcludedChild(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>) attributes {annotations = [{class = "sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation"}]} {
    // CHECK: %io_out_REG = reg %clock
    %io_out_REG = reg %clock : !firrtl.clock, !firrtl.uint<8>
    strictconnect %io_out_REG, %io_in : !firrtl.uint<8>
    strictconnect %io_out, %io_out_REG : !firrtl.uint<8>
  }
  // CHECK-LABEL: module @FullAsyncExcluded
  module @FullAsyncExcluded(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>, in %extraReset: !firrtl.asyncreset) attributes {
     portAnnotations = [[],[],[],[],[{class = "firrtl.transforms.DontTouchAnnotation"}, {class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
    // CHECK: %inst_clock, %inst_reset, %inst_io_in, %inst_io_out = instance inst @FullAsyncExcludedChild
    %inst_clock, %inst_reset, %inst_io_in, %inst_io_out = instance inst @FullAsyncExcludedChild(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, in io_in: !firrtl.uint<8>, out io_out: !firrtl.uint<8>)
    strictconnect %inst_clock, %clock : !firrtl.clock
    strictconnect %inst_reset, %reset : !firrtl.asyncreset
    strictconnect %io_out, %inst_io_out : !firrtl.uint<8>
    strictconnect %inst_io_in, %io_in : !firrtl.uint<8>
  }
}


// -----

// Local wire as async reset should be moved before all its uses.
firrtl.circuit "WireShouldDominate" {
  // CHECK-LABEL: module @WireShouldDominate
  module @WireShouldDominate(in %clock: !firrtl.clock) {
    %reg = reg %clock : !firrtl.clock, !firrtl.uint<8> // gets wired to localReset
    %localReset = wire {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    // CHECK-NEXT: %localReset = wire
    // CHECK-NEXT: [[RV:%.+]] = constant 0
    // CHECK-NEXT: %reg = regreset %clock, %localReset, [[RV]]
  }
}

// -----

// Local node as async reset should be moved before all its uses if its input
// value dominates the target location in the module.
firrtl.circuit "MovableNodeShouldDominate" {
  // CHECK-LABEL: module @MovableNodeShouldDominate
  module @MovableNodeShouldDominate(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    %0 = asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // does not block move of node
    %reg = reg %clock : !firrtl.clock, !firrtl.uint<8> // gets wired to localReset
    %localReset = node sym @theReset %0 {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    // CHECK-NEXT: %0 = asAsyncReset %ui1
    // CHECK-NEXT: %localReset = node sym @theReset %0
    // CHECK-NEXT: [[RV:%.+]] = constant 0
    // CHECK-NEXT: %reg = regreset %clock, %localReset, [[RV]]
  }
}

// -----

// Local node as async reset should be replaced by a wire and moved before all
// its uses if its input value does not dominate the target location in the
// module.
firrtl.circuit "UnmovableNodeShouldDominate" {
  // CHECK-LABEL: module @UnmovableNodeShouldDominate
  module @UnmovableNodeShouldDominate(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    %reg = reg %clock : !firrtl.clock, !firrtl.uint<8> // gets wired to localReset
    %0 = asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
    %localReset = node sym @theReset %0 {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    // CHECK-NEXT: %localReset = wire sym @theReset
    // CHECK-NEXT: [[RV:%.+]] = constant 0
    // CHECK-NEXT: %reg = regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: %0 = asAsyncReset %ui1
    // CHECK-NEXT: strictconnect %localReset, %0
  }
}

// -----

// Same test as above, ensure works w/forceable node.
firrtl.circuit "UnmovableForceableNodeShouldDominate" {
  // CHECK-LABEL: module @UnmovableForceableNodeShouldDominate
  module @UnmovableForceableNodeShouldDominate(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    %reg = reg %clock : !firrtl.clock, !firrtl.uint<8> // gets wired to localReset
    %0 = asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
    %localReset, %ref = node sym @theReset %0 forceable {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    // CHECK-NEXT: %localReset, %{{.+}} = wire sym @theReset
    // CHECK-NEXT: [[RV:%.+]] = constant 0
    // CHECK-NEXT: %reg = regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: %0 = asAsyncReset %ui1
    // CHECK-NEXT: strictconnect %localReset, %0
  }
}

// -----

// Move of local async resets should work across blocks.
firrtl.circuit "MoveAcrossBlocks1" {
  // CHECK-LABEL: module @MoveAcrossBlocks1
  module @MoveAcrossBlocks1(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    // <-- should move reset here
    when %ui1 : !firrtl.uint<1> {
      %reg = reg %clock : !firrtl.clock, !firrtl.uint<8> // gets wired to localReset
    }
    when %ui1 : !firrtl.uint<1> {
      %0 = asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
      %localReset = node sym @theReset %0 {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    }
    // CHECK-NEXT: %localReset = wire
    // CHECK-NEXT: when %ui1 : !firrtl.uint<1> {
    // CHECK-NEXT:   [[RV:%.+]] = constant 0
    // CHECK-NEXT:   %reg = regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: }
    // CHECK-NEXT: when %ui1 : !firrtl.uint<1> {
    // CHECK-NEXT:   [[TMP:%.+]] = asAsyncReset %ui1
    // CHECK-NEXT:   strictconnect %localReset, [[TMP]]
    // CHECK-NEXT: }
  }
}

// -----

firrtl.circuit "MoveAcrossBlocks2" {
  // CHECK-LABEL: module @MoveAcrossBlocks2
  module @MoveAcrossBlocks2(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    // <-- should move reset here
    when %ui1 : !firrtl.uint<1> {
      %0 = asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
      %localReset = node sym @theReset %0 {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    }
    when %ui1 : !firrtl.uint<1> {
      %reg = reg %clock : !firrtl.clock, !firrtl.uint<8> // gets wired to localReset
    }
    // CHECK-NEXT: %localReset = wire
    // CHECK-NEXT: when %ui1 : !firrtl.uint<1> {
    // CHECK-NEXT:   [[TMP:%.+]] = asAsyncReset %ui1
    // CHECK-NEXT:   strictconnect %localReset, [[TMP]]
    // CHECK-NEXT: }
    // CHECK-NEXT: when %ui1 : !firrtl.uint<1> {
    // CHECK-NEXT:   [[RV:%.+]] = constant 0
    // CHECK-NEXT:   %reg = regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: }
  }
}

// -----

firrtl.circuit "MoveAcrossBlocks3" {
  // CHECK-LABEL: module @MoveAcrossBlocks3
  module @MoveAcrossBlocks3(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    // <-- should move reset here
    %reg = reg %clock : !firrtl.clock, !firrtl.uint<8> // gets wired to localReset
    when %ui1 : !firrtl.uint<1> {
      %0 = asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
      %localReset = node sym @theReset %0 {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    }
    // CHECK-NEXT: %localReset = wire
    // CHECK-NEXT: [[RV:%.+]] = constant 0
    // CHECK-NEXT: %reg = regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: when %ui1 : !firrtl.uint<1> {
    // CHECK-NEXT:   [[TMP:%.+]] = asAsyncReset %ui1
    // CHECK-NEXT:   strictconnect %localReset, [[TMP]]
    // CHECK-NEXT: }
  }
}

// -----

firrtl.circuit "MoveAcrossBlocks4" {
  // CHECK-LABEL: module @MoveAcrossBlocks4
  module @MoveAcrossBlocks4(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    // <-- should move reset here
    when %ui1 : !firrtl.uint<1> {
      %reg = reg %clock : !firrtl.clock, !firrtl.uint<8> // gets wired to localReset
    }
    %0 = asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
    %localReset = node sym @theReset %0 {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.asyncreset
    // CHECK-NEXT: %localReset = wire
    // CHECK-NEXT: when %ui1 : !firrtl.uint<1> {
    // CHECK-NEXT:   [[RV:%.+]] = constant 0
    // CHECK-NEXT:   %reg = regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: }
    // CHECK-NEXT: [[TMP:%.+]] = asAsyncReset %ui1
    // CHECK-NEXT: strictconnect %localReset, [[TMP]]
  }
}

// -----

firrtl.circuit "SubAccess" {
  module @SubAccess(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %init: !firrtl.uint<1>, in %in: !firrtl.uint<8>, in %extraReset: !firrtl.asyncreset ) attributes {
    // CHECK-LABEL: module @SubAccess
    portAnnotations = [[],[],[],[],[{class = "firrtl.transforms.DontTouchAnnotation"}, {class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
    %c1_ui8 = constant 1 : !firrtl.uint<2>
    %arr = wire : !firrtl.vector<uint<8>, 1>
    %reg6 = regreset %clock, %init, %c1_ui8 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>
    %2 = subaccess %arr[%reg6] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<2>
    strictconnect %2, %in : !firrtl.uint<8>
    // CHECK:  %reg6 = regreset %clock, %extraReset, %c0_ui2  : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<2>, !firrtl.uint<2>
    // CHECK-NEXT: %0 = mux(%init, %c1_ui2, %reg6)
    // CHECK: strictconnect %reg6, %0
    // CHECK-NEXT:  %[[v0:.+]] = subaccess %arr[%reg6] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<2>
    // CHECK-NEXT:  strictconnect %[[v0]], %in : !firrtl.uint<8>

  }
}

// -----

// This is a regression check to ensure that a zero-width register gets a proper
// reset value.
// CHECK-LABEL: module @ZeroWidthRegister
firrtl.circuit "ZeroWidthRegister" {
  module @ZeroWidthRegister(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) attributes {
    portAnnotations = [[],[{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]]} {
    %reg = reg %clock : !firrtl.clock, !firrtl.uint<0>
    // CHECK-NEXT: [[TMP:%.+]] = constant 0 : !firrtl.uint<0>
    // CHECK-NEXT: %reg = regreset %clock, %reset, [[TMP]]
  }
}

// -----

// Check that unaffected fields ("data") are not being affected by width
// inference. See https://github.com/llvm/circt/issues/2857.
// CHECK-LABEL: module @ZeroLengthVectorInBundle1
firrtl.circuit "ZeroLengthVectorInBundle1"  {
  module @ZeroLengthVectorInBundle1(out %out: !firrtl.bundle<resets: vector<reset, 0>, data flip: uint<3>>) {
    %0 = subfield %out[resets] : !firrtl.bundle<resets: vector<reset, 0>, data flip: uint<3>>
    %invalid = invalidvalue : !firrtl.vector<reset, 0>
    strictconnect %0, %invalid : !firrtl.vector<reset, 0>
    // CHECK-NEXT: %0 = subfield %out[resets] : !firrtl.bundle<resets: vector<uint<1>, 0>, data flip: uint<3>>
    // CHECK-NEXT: %invalid = invalidvalue : !firrtl.vector<uint<1>, 0>
    // CHECK-NEXT: strictconnect %0, %invalid : !firrtl.vector<uint<1>, 0>
  }
}

// -----

// CHECK-LABEL: module @ZeroLengthVectorInBundle2
firrtl.circuit "ZeroLengthVectorInBundle2"  {
  module @ZeroLengthVectorInBundle2(out %out: !firrtl.bundle<resets: vector<bundle<a: reset>, 0>, data flip: uint<3>>) {
    %0 = subfield %out[resets] : !firrtl.bundle<resets: vector<bundle<a: reset>, 0>, data flip: uint<3>>
    %invalid = invalidvalue : !firrtl.vector<bundle<a: reset>, 0>
    strictconnect %0, %invalid : !firrtl.vector<bundle<a: reset>, 0>
    // CHECK-NEXT: %0 = subfield %out[resets] : !firrtl.bundle<resets: vector<bundle<a: uint<1>>, 0>, data flip: uint<3>>
    // CHECK-NEXT: %invalid = invalidvalue : !firrtl.vector<bundle<a: uint<1>>, 0>
    // CHECK-NEXT: strictconnect %0, %invalid : !firrtl.vector<bundle<a: uint<1>>, 0>
  }
}

// -----

// Resets nested underneath a zero-length vector should infer to `UInt<1>`.
// CHECK-LABEL: module @ZeroVecBundle
// CHECK-SAME: in %a: !firrtl.vector<bundle<x: uint<1>>, 0>
// CHECK-SAME: out %b: !firrtl.vector<bundle<x: uint<1>>, 0>
firrtl.circuit "ZeroVecBundle"  {
  module @ZeroVecBundle(in %a: !firrtl.vector<bundle<x: uint<1>>, 0>, out %b: !firrtl.vector<bundle<x: reset>, 0>) {
    %w = wire : !firrtl.vector<bundle<x: reset>, 0>
    strictconnect %b, %w : !firrtl.vector<bundle<x: reset>, 0>
    // CHECK-NEXT: %w = wire : !firrtl.vector<bundle<x: uint<1>>, 0>
    // CHECK-NEXT: strictconnect %b, %w : !firrtl.vector<bundle<x: uint<1>>, 0>
  }
}

// -----

// Resets directly in a zero-length vector should infer to `UInt<1>`.
// CHECK-LABEL: module @ZeroVec
// CHECK-SAME: in %a: !firrtl.bundle<x: vector<uint<1>, 0>>
// CHECK-SAME: out %b: !firrtl.bundle<x: vector<uint<1>, 0>>
firrtl.circuit "ZeroVec"  {
  module @ZeroVec(in %a: !firrtl.bundle<x: vector<reset, 0>>, out %b: !firrtl.bundle<x: vector<reset, 0>>) {
    strictconnect %b, %a : !firrtl.bundle<x: vector<reset, 0>>
    // CHECK-NEXT: strictconnect %b, %a : !firrtl.bundle<x: vector<uint<1>, 0>>
  }
}

// -----

// CHECK-LABEL: "RefReset"
firrtl.circuit "RefReset" {
  // CHECK-LABEL: module private @SendReset
  // CHECK-SAME: in %r: !firrtl.asyncreset
  // CHECK-SAME: out %ref: !firrtl.probe<asyncreset>
  // CHECK-NEXT: send %r : !firrtl.asyncreset
  // CHECK-NEXT: probe<asyncreset>
  module private @SendReset(in %r: !firrtl.reset, out %ref: !firrtl.probe<reset>) {
    %ref_r = ref.send %r : !firrtl.reset
    ref.define %ref, %ref_r : !firrtl.probe<reset>
  }
  // CHECK-LABEL: module @RefReset
  // CHECK-NEXT: in r: !firrtl.asyncreset
  // CHECK-SAME: out ref: !firrtl.probe<asyncreset>
  // CHECK-NEXT: !firrtl.asyncreset, !firrtl.asyncreset
  // CHECK-NEXT: %s_ref : !firrtl.probe<asyncreset>
  module @RefReset(in %r: !firrtl.asyncreset) {
    %s_r, %s_ref = instance s @SendReset(in r: !firrtl.reset, out ref: !firrtl.probe<reset>)
    connect %s_r, %r : !firrtl.reset, !firrtl.asyncreset
    %reset = ref.resolve %s_ref : !firrtl.probe<reset>
  }
}

// -----

// Check resets are inferred through references to bundles w/flips.

// CHECK-LABEL: "RefResetBundle"
firrtl.circuit "RefResetBundle" {
  // CHECK-LABEL: module @RefResetBundle
  // CHECK-NOT: reset
  module @RefResetBundle(in %driver: !firrtl.asyncreset, out %out: !firrtl.bundle<a: reset, b: reset>) {
  %r = wire : !firrtl.bundle<a: reset, b flip: reset> 
  %ref_r = ref.send %r : !firrtl.bundle<a: reset, b flip: reset>
  %reset = ref.resolve %ref_r : !firrtl.probe<bundle<a: reset, b: reset>>
  strictconnect %out, %reset : !firrtl.bundle<a: reset, b: reset>

   %r_a = subfield %r[a] : !firrtl.bundle<a: reset, b flip: reset>
   %r_b = subfield %r[b] : !firrtl.bundle<a: reset, b flip: reset>
   connect %r_a, %driver : !firrtl.reset, !firrtl.asyncreset
   connect %r_b, %driver : !firrtl.reset, !firrtl.asyncreset
  }
}

// -----

// Check resets are inferred through ref.sub.

// CHECK-LABEL: "RefResetSub"
firrtl.circuit "RefResetSub" {
  // CHECK-LABEL: module @RefResetSub
  // CHECK-NOT: reset
  module @RefResetSub(in %driver: !firrtl.asyncreset, out %out_a : !firrtl.reset, out %out_b: !firrtl.vector<reset,2>) {
  %r = wire : !firrtl.bundle<a: reset, b flip: vector<reset, 2>> 
  %ref_r = ref.send %r : !firrtl.bundle<a: reset, b flip: vector<reset, 2>>
  %ref_r_a = ref.sub %ref_r[0] : !firrtl.probe<bundle<a: reset, b : vector<reset, 2>>>
  %reset_a = ref.resolve %ref_r_a : !firrtl.probe<reset>

  %ref_r_b = ref.sub %ref_r[1] : !firrtl.probe<bundle<a: reset, b : vector<reset, 2>>>
  %reset_b = ref.resolve %ref_r_b : !firrtl.probe<vector<reset, 2>>

  strictconnect %out_a, %reset_a : !firrtl.reset
  strictconnect %out_b, %reset_b : !firrtl.vector<reset, 2>

   %r_a = subfield %r[a] : !firrtl.bundle<a: reset, b flip: vector<reset, 2>>
   %r_b = subfield %r[b] : !firrtl.bundle<a: reset, b flip: vector<reset, 2>>
   %r_b_0 = subindex %r_b[0] : !firrtl.vector<reset, 2>
   %r_b_1 = subindex %r_b[1] : !firrtl.vector<reset, 2>
   connect %r_a, %driver : !firrtl.reset, !firrtl.asyncreset
   connect %r_b_0, %driver : !firrtl.reset, !firrtl.asyncreset
   connect %r_b_1, %driver : !firrtl.reset, !firrtl.asyncreset
  }
}

// -----

// CHECK-LABEL: "ConstReset"
firrtl.circuit "ConstReset" {
  // CHECK-LABEL: module private @InfersConstAsync(in %r: !firrtl.const.asyncreset)
  module private @InfersConstAsync(in %r: !firrtl.const.reset) {}

  // CHECK-LABEL: module private @InfersConstSync(in %r: !firrtl.const.uint<1>)
  module private @InfersConstSync(in %r: !firrtl.const.reset) {}

  // CHECK-LABEL: module private @InfersAsync(in %r: !firrtl.asyncreset)
  module private @InfersAsync(in %r: !firrtl.reset) {}

  // CHECK-LABEL: module private @InfersSync(in %r: !firrtl.uint<1>)
  module private @InfersSync(in %r: !firrtl.reset) {}

  module @ConstReset(in %async: !firrtl.const.asyncreset, in %sync: !firrtl.const.uint<1>) {
    %constAsyncTarget = instance infersConstAsync @InfersConstAsync(in r: !firrtl.const.reset)
    %constSyncTarget = instance infersConstSync @InfersConstSync(in r: !firrtl.const.reset)
    %asyncTarget = instance infersAsync @InfersAsync(in r: !firrtl.reset)
    %syncTarget = instance infersSync @InfersSync(in r: !firrtl.reset)

    connect %constAsyncTarget, %async : !firrtl.const.reset, !firrtl.const.asyncreset
    connect %constSyncTarget, %sync : !firrtl.const.reset, !firrtl.const.uint<1>
    connect %asyncTarget, %async : !firrtl.reset, !firrtl.const.asyncreset
    connect %syncTarget, %sync : !firrtl.reset, !firrtl.const.uint<1>
  }
}

// -----

// Check resets are inferred for forceable ops.

// CHECK-LABEL: "InferToRWProbe"
firrtl.circuit "InferToRWProbe" {
  // CHECK-LABEL: module @InferToRWProbe
  // CHECK-NOT: reset
  module @InferToRWProbe(in %driver: !firrtl.asyncreset, out %out: !firrtl.bundle<a: reset, b: reset>) {
  %r, %r_rw = wire forceable : !firrtl.bundle<a: reset, b flip: reset>, !firrtl.rwprobe<bundle<a: reset, b : reset>>
  %reset = ref.resolve %r_rw : !firrtl.rwprobe<bundle<a: reset, b: reset>>
  strictconnect %out, %reset : !firrtl.bundle<a: reset, b: reset>

   %r_a = subfield %r[a] : !firrtl.bundle<a: reset, b flip: reset>
   %r_b = subfield %r[b] : !firrtl.bundle<a: reset, b flip: reset>
   connect %r_a, %driver : !firrtl.reset, !firrtl.asyncreset
   connect %r_b, %driver : !firrtl.reset, !firrtl.asyncreset
  }
}
