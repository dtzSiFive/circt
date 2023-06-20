//===- LowerProperties.cpp - "FIRRTL properties to OM" ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerProperties pass, which takes the module tree in
// FIRRTL, and creates a parallel tree of OM classes, where each property in
// FIRRTL becomes a field in OM.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/FieldRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

using namespace circt;
using namespace firrtl;


//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class LowerPropertiesPass : public LowerPropertiesBase<LowerPropertiesPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void LowerPropertiesPass::runOnOperation() {}

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerPropertiesPass() {
  return std::make_unique<LowerPropertiesPass>();
}
