// RUN: circt-opt --pass-pipeline="builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-matches)))" %s | FileCheck %s

firrtl.circuit "LowerMatches" {

// CHECK-LABEL: module @EmptyEnum
firrtl.module @EmptyEnum(in %enum : !firrtl.enum<>) {
  match %enum : !firrtl.enum<> {
  }
// CHECK-NEXT: }
}

// CHECK-LABEL: module @OneVariant
firrtl.module @OneVariant(in %enum : !firrtl.enum<a: uint<8>>, out %out : !firrtl.uint<8>) {
  // CHECK: %0 = subtag %enum[a] : !firrtl.enum<a: uint<8>>
  // CHECK: strictconnect %out, %0 : !firrtl.uint<8>
  match %enum : !firrtl.enum<a: uint<8>> {
    case a(%arg0) {
      strictconnect %out, %arg0 : !firrtl.uint<8>
    }
  }
}

// CHECK-LABEL: module @LowerMatches
firrtl.module @LowerMatches(in %enum : !firrtl.enum<a: uint<8>, b: uint<8>, c: uint<8>>, out %out : !firrtl.uint<8>) {

   // CHECK-NEXT: %0 = istag %enum a : !firrtl.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT: when %0 : !firrtl.uint<1> {
   // CHECK-NEXT:   %1 = subtag %enum[a] : !firrtl.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT:   strictconnect %out, %1 : !firrtl.uint<8>
   // CHECK-NEXT: } else {
   // CHECK-NEXT:   %1 = istag %enum b : !firrtl.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT:   when %1 : !firrtl.uint<1> {
   // CHECK-NEXT:     %2 = subtag %enum[b] : !firrtl.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT:     strictconnect %out, %2 : !firrtl.uint<8>
   // CHECK-NEXT:   } else {
   // CHECK-NEXT:     %2 = subtag %enum[c] : !firrtl.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT:     strictconnect %out, %2 : !firrtl.uint<8>
   // CHECK-NEXT:   }
   // CHECK-NEXT: }
  match %enum : !firrtl.enum<a: uint<8>, b: uint<8>, c: uint<8>> {
    case a(%arg0) {
      strictconnect %out, %arg0 : !firrtl.uint<8>
    }
    case b(%arg0) {
      strictconnect %out, %arg0 : !firrtl.uint<8>
    }
    case c(%arg0) {
      strictconnect %out, %arg0 : !firrtl.uint<8>
    }
  }

}

// CHECK-LABEL: module @ConstLowerMatches
firrtl.module @ConstLowerMatches(in %enum : !firrtl.const.enum<a: uint<8>, b: uint<8>, c: uint<8>>, out %out : !firrtl.const.uint<8>) {

   // CHECK-NEXT: %0 = istag %enum a : !firrtl.const.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT: when %0 : !firrtl.const.uint<1> {
   // CHECK-NEXT:   %1 = subtag %enum[a] : !firrtl.const.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT:   strictconnect %out, %1 : !firrtl.const.uint<8>
   // CHECK-NEXT: } else {
   // CHECK-NEXT:   %1 = istag %enum b : !firrtl.const.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT:   when %1 : !firrtl.const.uint<1> {
   // CHECK-NEXT:     %2 = subtag %enum[b] : !firrtl.const.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT:     strictconnect %out, %2 : !firrtl.const.uint<8>
   // CHECK-NEXT:   } else {
   // CHECK-NEXT:     %2 = subtag %enum[c] : !firrtl.const.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT:     strictconnect %out, %2 : !firrtl.const.uint<8>
   // CHECK-NEXT:   }
   // CHECK-NEXT: }
  match %enum : !firrtl.const.enum<a: uint<8>, b: uint<8>, c: uint<8>> {
    case a(%arg0) {
      strictconnect %out, %arg0 : !firrtl.const.uint<8>
    }
    case b(%arg0) {
      strictconnect %out, %arg0 : !firrtl.const.uint<8>
    }
    case c(%arg0) {
      strictconnect %out, %arg0 : !firrtl.const.uint<8>
    }
  }

}
}